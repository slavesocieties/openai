import cv2
import numpy as np

def detect_spine_column(
    image_path,
    column_width=5,
    dark_fraction_threshold=0.5,
    folio_side="recto",
    edge_margin_fraction=0.25
):
    """
    Searches for a narrow vertical strip of width ~ column_width
    where each column is at least dark_fraction_threshold black.
    If folio_side="recto", we limit to left edge;
    if folio_side="verso", we limit to right edge.
    Returns the center x-coordinate of the first matching spine strip, or None if none found.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Could not load image: {image_path}")

    img_h, img_w = gray.shape

    # Threshold (Otsu)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # We assume the background is black => 0, folio is white => 255, no invert needed. 
    # If your data is reversed, do: thresh = cv2.bitwise_not(thresh)

    # Fraction of black pixels per column
    is_black = (thresh == 0)
    col_black_counts = np.sum(is_black, axis=0)  # shape=(img_w,)
    col_black_fraction = col_black_counts / float(img_h)

    # Limit search region to the left or right side if desired
    left_limit = 0
    right_limit = img_w
    if folio_side == "recto":
        right_limit = int(img_w * edge_margin_fraction)
    else:  # verso
        left_limit = int(img_w * (1.0 - edge_margin_fraction))

    left_limit = max(0, left_limit)
    right_limit = min(img_w, right_limit)

    # Slide a window of width column_width across [left_limit, right_limit]
    for start_col in range(left_limit, right_limit - column_width + 1):
        window = col_black_fraction[start_col : start_col + column_width]
        if np.all(window >= dark_fraction_threshold):
            # Found a potential spine
            center_col = start_col + column_width // 2
            return center_col

    return None

def classify_components_single_pass(
    image_path,
    bboxes,
    output_path,
    folio_side="recto",
    nbins=50,
    percentile=50,
    # The user says margin is always on the LEFT side of main text region:
    # no separate "left_is_margin" needed.
    facing_folio_fraction=0.15,
    narrow_width_fraction=0.2,
    tall_ratio=5.0,
    color_map=None
):
    """
    Classifies bounding boxes into:
      - "text_region"
      - "margin"
      - "facing_folio"
      - "other"

    Key Rules:
      1) Attempt spine detection => label boxes across the spine as "facing_folio".
      2) Use a column histogram for the rest => define [leftmost_main_x, rightmost_main_x].
      3) If center < leftmost_main_x => "margin", else => "text_region",
         THEN re-check if "margin" center >= leftmost_main_x => becomes "text_region".
      4) If no spine found => fallback fraction approach for facing_folio.
      5) In a final pass, any bounding box that is extremely tall (height > tall_ratio * width)
         OR is narrower than (narrow_width_fraction * average_text_width) => "other".
    
    :param image_path: Path to the original image.
    :param bboxes: List of (x, y, w, h).
    :param output_path: Where to save the final annotated image.
    :param folio_side: "recto" or "verso".
    :param nbins: # of horizontal bins for the column histogram.
    :param percentile: e.g. 50 => median threshold for "non-sparse" bins.
    :param facing_folio_fraction: fallback fraction if spine not found.
    :param narrow_width_fraction: e.g. 0.2 => if box width < 0.2 * avg_text_width => "other"
    :param tall_ratio: e.g. 5 => if box height > 5 * width => "other"
    :param color_map: optional dict => category : (B,G,R)
    :return: A list of bounding boxes labeled "text_region".
    """
    # Default colors if not provided
    if color_map is None:
        color_map = {
            "text_region": (0, 255, 0),
            "margin": (0, 215, 255),
            "facing_folio": (255, 0, 0),
            "other": (128, 128, 128)
        }

    # 1) Detect spine
    spine_x = None
    for cw in range(5, 0, -1):
        attempt = detect_spine_column(
            image_path=image_path,
            column_width=cw,
            dark_fraction_threshold=0.5,
            folio_side=folio_side,
            edge_margin_fraction=0.25
        )
        if attempt is not None:
            spine_x = attempt
            break

    # Load the image for drawing
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load {image_path}")
    img_h, img_w = image.shape[:2]

    classifications = ["" for _ in bboxes]

    # 2) If we have spine => label bounding boxes across it as facing_folio
    if spine_x is not None:
        if folio_side == "recto":
            # everything left => facing_folio
            for i, (bx, by, bw, bh) in enumerate(bboxes):
                cx = bx + bw/2.0
                if cx < spine_x:
                    classifications[i] = "facing_folio"
        else:
            # verso => everything right => facing_folio
            for i, (bx, by, bw, bh) in enumerate(bboxes):
                cx = bx + bw/2.0
                if cx > spine_x:
                    classifications[i] = "facing_folio"

    # 3) Column histogram for non-facing_folio => define main region
    bins = np.zeros(nbins, dtype=int)
    unclassified_idxs = []
    for i, (bx, by, bw, bh) in enumerate(bboxes):
        if classifications[i] == "facing_folio":
            continue
        cx = bx + bw/2.0
        bin_idx = int((cx / float(img_w)) * nbins)
        bin_idx = max(0, min(bin_idx, nbins - 1))
        bins[bin_idx] += 1
        unclassified_idxs.append(i)

    sorted_counts = np.sort(bins)
    cutoff_index = int((percentile / 100.0) * nbins)
    cutoff_index = max(0, min(cutoff_index, nbins - 1))
    threshold_value = sorted_counts[cutoff_index]
    non_sparse_bins = [b for b, c in enumerate(bins) if c >= threshold_value]

    if not non_sparse_bins:
        leftmost_bin = 0
        rightmost_bin = nbins - 1
    else:
        leftmost_bin = min(non_sparse_bins)
        rightmost_bin = max(non_sparse_bins)

    leftmost_main_x = leftmost_bin * (img_w / float(nbins))
    rightmost_main_x = (rightmost_bin + 1) * (img_w / float(nbins))

    # 4) If no spine => fallback fraction approach for facing_folio
    if spine_x is None:
        for i in unclassified_idxs:
            (bx, by, bw, bh) = bboxes[i]
            cx = bx + bw/2.0
            if folio_side == "recto":
                if cx < leftmost_main_x * (1 - facing_folio_fraction):
                    classifications[i] = "facing_folio"
            else:
                if cx > rightmost_main_x * (1 + facing_folio_fraction):
                    classifications[i] = "facing_folio"

    # 5) Among remaining, label margin vs text_region
    #    margin = center < leftmost_main_x
    #    text_region = otherwise
    plausible_text_widths = []
    for i in unclassified_idxs:
        if classifications[i] == "":
            bx, by, bw, bh = bboxes[i]
            cx = bx + bw/2.0
            if cx < leftmost_main_x:
                classifications[i] = "margin"
            else:
                classifications[i] = "text_region"
                plausible_text_widths.append(bw)

    # Re-check margin: if center >= leftmost_main_x => label text_region
    for i, cat in enumerate(classifications):
        if cat == "margin":
            (bx, by, bw, bh) = bboxes[i]
            cx = bx + bw/2.0
            if cx >= leftmost_main_x:
                classifications[i] = "text_region"
                plausible_text_widths.append(bw)

    # 6) Final pass: label "other" if box is extremely tall (bh > tall_ratio * bw)
    #    or narrower than narrow_width_fraction * average_text_width
    if plausible_text_widths:
        avg_text_width = np.mean(plausible_text_widths)
    else:
        avg_text_width = 1.0

    for i, cat in enumerate(classifications):
        if cat not in ("facing_folio", "other"):
            # We'll override margin or text_region with "other" if too tall or too narrow
            (bx, by, bw, bh) = bboxes[i]
            # check ratio
            if bw > 0:
                height_ratio = bh / float(bw)
                if height_ratio > tall_ratio:
                    classifications[i] = "other"
                    continue
            # check narrowness
            if bw < avg_text_width * narrow_width_fraction:
                classifications[i] = "other"

        # If you also want to override "facing_folio" if it's extremely tall, 
        # you could do the same checks for cat == "facing_folio" etc.

    # 7) Draw final boxes
    final_img = image.copy()
    for (box, cat) in zip(bboxes, classifications):
        (bx, by, bw, bh) = box
        color = color_map.get(cat, (255, 255, 255))
        cv2.rectangle(final_img, (bx, by), (bx + bw, by + bh), color, 2)

    cv2.imwrite(output_path, final_img)

    # Return only text_region
    text_region_bboxes = [
        b for b, cat in zip(bboxes, classifications) if cat == "text_region"
    ]
    return text_region_bboxes

def extract_connected_components(image_path, min_area=50, max_area=50000):
    # 1) Load grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Could not load image: {image_path}")

    # 2) Optional: Enhance local contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # 3) Adaptive threshold
    binary = cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,   # blockSize
        30    # C
    )

    # 4) Small morphological close to unify strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5) Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # 6) Filter by area
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            bboxes.append((x, y, w, h))

    return bboxes

def draw_bounding_boxes(image_path, bboxes, output_path, color=(0, 255, 0)):
    """
    Draws the given bounding boxes on the image at `image_path` and writes
    the result to `output_path`.

    :param image_path: Path to the input image file.
    :param bboxes: List of bounding boxes, each in the form (x, y, w, h).
    :param output_path: File path to save the annotated image.
    :param color: A (B, G, R) tuple specifying the color for the bounding boxes;
                  defaults to green = (0, 255, 0).
    """
    # 1) Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image from path: {image_path}")

    # 2) Draw each bounding box
    for (x, y, w, h) in bboxes:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(image, top_left, bottom_right, color, thickness=2)

    # 3) Save the annotated result
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Could not write the annotated image to {output_path}")

    print(f"Bounding boxes drawn and saved to {output_path}")

def merge_bboxes(bboxes, merge_threshold):
    """
    Merges bounding boxes that overlap or lie within `merge_threshold` pixels (horizontally or vertically).
    Returns a new list of merged bounding boxes.
    """
    if not bboxes:
        return []

    merged = True
    while merged:
        merged = False
        new_bboxes = []
        taken = [False] * len(bboxes)

        for i in range(len(bboxes)):
            if taken[i]:
                continue
            (Ax, Ay, Aw, Ah) = bboxes[i]
            A_x2 = Ax + Aw
            A_y2 = Ay + Ah
            boxA = (Ax, Ay, Aw, Ah)

            for j in range(i + 1, len(bboxes)):
                if taken[j]:
                    continue
                (Bx, By, Bw, Bh) = bboxes[j]
                B_x2 = Bx + Bw
                B_y2 = By + Bh

                # Check if A & B overlap or are within `merge_threshold` in any direction
                if not (
                    Bx > A_x2 + merge_threshold or
                    B_x2 < Ax - merge_threshold or
                    By > A_y2 + merge_threshold or
                    B_y2 < Ay - merge_threshold
                ):
                    # Merge them into a bigger bounding box
                    newX = min(Ax, Bx)
                    newY = min(Ay, By)
                    newX2 = max(A_x2, B_x2)
                    newY2 = max(A_y2, B_y2)
                    boxA = (newX, newY, newX2 - newX, newY2 - newY)

                    # Update references to boxA for continuing merges
                    (Ax, Ay, Aw, Ah) = boxA
                    A_x2 = Ax + Aw
                    A_y2 = Ay + Ah

                    taken[j] = True
                    merged = True

            taken[i] = True
            new_bboxes.append(boxA)

        bboxes = new_bboxes

    return bboxes

def adaptive_merge_pipeline(
    bboxes,
    box_count_range=(2, 4),
    initial_merge_threshold=10,
    threshold_min=0,
    threshold_max=200,
    max_iterations=10
):
    """
    Iteratively adjusts 'merge_threshold' until the bounding-box count is in [min_count, max_count]
    or we exhaust max_iterations.

    :param bboxes: list of (x, y, w, h) bounding boxes
    :param box_count_range: (min_count, max_count) target range for final bounding-box count
    :param initial_merge_threshold: starting threshold
    :param threshold_min: lower bound for threshold
    :param threshold_max: upper bound for threshold
    :param max_iterations: max tries before giving up
    :return: (merged_bboxes, final_threshold, success_flag)
    """
    merge_threshold = initial_merge_threshold
    min_count, max_count = box_count_range
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1

        merged = merge_bboxes(bboxes, merge_threshold)
        n = len(merged)

        if min_count <= n <= max_count:
            # We hit the desired bounding-box count range => success
            return merged, merge_threshold, True

        # Otherwise, adjust threshold
        if n < min_count:
            # Too few => we over-merged => decrease threshold
            merge_threshold = max(threshold_min, merge_threshold - 5)
        elif n > max_count:
            # Too many => not merged enough => increase threshold
            merge_threshold = min(threshold_max, merge_threshold + 5)
        # Continue loop with updated threshold

    # If we exit the loop, return the last attempt
    final_merged = merge_bboxes(bboxes, merge_threshold)
    n = len(final_merged)
    success = (min_count <= n <= max_count)
    return final_merged, merge_threshold, success

def finalize_text_bounding_boxes(
    image_path,
    output_path,
    min_area=50,
    max_area=50000,
    # classification parameters
    folio_side="recto",
    nbins=50,
    percentile=50,
    marginal_width_fraction=0.2,
    facing_folio_fraction=0.15,
    # merging parameters
    box_count_range=(2, 4),
    initial_merge_threshold=10,
    max_iterations=10
):
    """
    1) Extract connected components with adaptive threshold for faint handwriting.
    2) Classify them (text_region vs. margin vs. facing_folio), 
       saving an intermediate annotated image if desired.
    3) Filter to only text_region bounding boxes.
    4) Adaptive-merge them to get 2-4 final bounding boxes.
    5) Annotate and return the final bounding boxes.
    """
    # -- A) Extract connected components (with adaptive threshold, per your updated function) --    
    all_bboxes = extract_connected_components(image_path, min_area, max_area)

    # -- B) Classify them (this saves an annotated image with margin/facing_folio/text) --    
    text_bboxes = classify_components_single_pass(
        image_path=image_path,
        bboxes=all_bboxes,
        output_path=output_path,  # e.g. "classified_components.png"
        folio_side=folio_side,
        nbins=nbins,
        percentile=percentile,
        narrow_width_fraction=marginal_width_fraction,
        facing_folio_fraction=facing_folio_fraction
    )

    # -- C) Now we only keep text_region bboxes => run adaptive merging
    merged_bboxes, final_thr, success = adaptive_merge_pipeline(
        bboxes=text_bboxes,
        box_count_range=box_count_range,
        initial_merge_threshold=initial_merge_threshold,
        max_iterations=max_iterations
    )

    # Optionally, we can annotate these final merged boxes in a new image
    final_img = cv2.imread(image_path)
    if final_img is None:
        raise IOError("Could not load original image")
    # Draw final boxes in e.g. green
    for (x, y, w, h) in merged_bboxes:
        cv2.rectangle(final_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    merged_output_path = "final_merged_output.png"
    cv2.imwrite(merged_output_path, final_img)

    print(f"Adaptive merge success={success}, final threshold={final_thr}, final box count={len(merged_bboxes)}")
    print(f"Final merged annotation saved to {merged_output_path}")

    return merged_bboxes

if __name__ == "__main__":    
    image_path = "images/239746-0020.jpg"
    output_path = "classification_with_spine_detection.png"

    text_boxes = finalize_text_bounding_boxes(
        image_path,
        output_path,
        min_area=50,
        max_area=50000,
        # classification parameters
        folio_side="verso",
        nbins=50,
        percentile=50,
        marginal_width_fraction=0.2,
        facing_folio_fraction=0.15,
        # merging parameters
        box_count_range=(2, 4),
        initial_merge_threshold=10,
        max_iterations=10
    )

    print(f"Found {len(text_boxes)} bounding boxes in main text region. Classification visualization saved to {output_path}")


