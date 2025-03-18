import cv2
import numpy as np
from scipy.signal import find_peaks

def detect_vertical_line_paragraph_breaks(
    image_path,
    text_bboxes,
    min_lines=20,
    max_lines=40,
    paragraph_gap_multiplier=2.0,
    smoothing_kernel=5,
    output_path="line_paragraph_breaks.png"
):
    """
    Identifies potential line breaks by finding local minima in a vertical 
    component-density curve. Also identifies extended low-density regions 
    as paragraph breaks if they exceed 'paragraph_gap_multiplier' times 
    the average line-gap height.

    1) For each bounding box in text_bboxes, increment row_density[row] 
       from y..y+h. 
    2) Smooth row_density with a small kernel (smoothing_kernel).
    3) Find local minima in the negative row_density (so we look for valleys).
       Adapt threshold to get ~ [min_lines, max_lines] minima, if possible.
    4) Measure average gap between consecutive minima => 
       identify any region of low density that is >= paragraph_gap_multiplier * that gap 
       as a 'paragraph break.'
    5) Visualize line breaks (thin horizontal lines) & paragraph breaks (thicker lines)
       on the original image, saving to output_path.

    :param image_path: Path to the original image.
    :param text_bboxes: List of (x, y, w, h) bounding boxes for the main text region.
    :param min_lines: desired lower bound on # lines
    :param max_lines: desired upper bound on # lines
    :param paragraph_gap_multiplier: e.g. 2.0 => if a gap is twice the average line-gap => paragraph break
    :param smoothing_kernel: size (in rows) of the moving-average smoothing.
    :param output_path: Where to save the annotated image.
    :return: (line_break_rows, paragraph_break_ranges)
             line_break_rows: list of y-coordinates (int) for potential line breaks
             paragraph_break_ranges: list of (start_y, end_y) for extended low-density regions
    """

    # 1) Load image & get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")
    img_h, img_w = image.shape[:2]

    # 2) Build a vertical density array
    row_density = np.zeros(img_h, dtype=np.int32)
    for (bx, by, bw, bh) in text_bboxes:
        top = max(0, by)
        bot = min(img_h, by + bh)
        row_density[top:bot] += 1

    # 3) (Optional) Smooth row_density with a small moving average or convolution
    if smoothing_kernel > 1:
        kernel = np.ones(smoothing_kernel) / smoothing_kernel
        smoothed = np.convolve(row_density, kernel, mode='same')
    else:
        smoothed = row_density.astype(np.float32)

    # We'll invert it (negate) so that line gaps => local minima become local *maxima* in negative space
    # but we can also directly find minima with find_peaks on (-smoothed).
    neg_smoothed = -smoothed

    # 4) Find local minima (peaks in neg_smoothed). We'll adapt the 'prominence' or 'distance' 
    #    threshold so we end up with ~ [min_lines, max_lines] peaks if possible.

    # A function to do the detection with adjustable parameters
    def detect_minima(neg_signal, distance=10, prominence=5):
        # find_peaks => local maxima in neg_signal => local minima in original smoothed
        # distance => minimal spacing between peaks
        # prominence => minimal difference from surrounding baseline
        peaks, properties = find_peaks(neg_signal, distance=distance, prominence=prominence)
        return peaks, properties

    # We'll do a simple iterative approach adjusting "prominence" until we get 
    # the # of peaks in [min_lines, max_lines], or we give up.
    # We'll start with some baseline guess for distance and prominence.
    distance_guess = 10
    prominence_guess = 5
    found_peaks = []
    max_iter = 20

    for _ in range(max_iter):
        peaks, _ = detect_minima(neg_smoothed, distance=distance_guess, prominence=prominence_guess)
        count = len(peaks)
        if min_lines <= count <= max_lines:
            found_peaks = peaks
            break
        # If too many lines => increase prominence => fewer peaks
        if count > max_lines:
            prominence_guess += 2
        elif count < min_lines and prominence_guess > 0:
            # too few => decrease prominence if possible
            prominence_guess = max(0, prominence_guess - 1)
        else:
            # if that fails to converge, we could also tweak 'distance_guess'
            # but let's keep it simple for now
            pass

    line_break_rows = found_peaks.tolist()  # potential line-break row indices
    line_break_rows.sort()

    # 5) Identify paragraph breaks: we measure average gap between line_break_rows.
    #    If there's a gap >= paragraph_gap_multiplier * average_gap => call that 
    #    region a paragraph break region.
    paragraph_break_ranges = []
    if len(line_break_rows) > 1:
        # measure gaps
        gaps = []
        for i in range(len(line_break_rows) - 1):
            gap = line_break_rows[i+1] - line_break_rows[i]
            gaps.append(gap)
        if gaps:
            avg_gap = np.mean(gaps)
            para_gap = paragraph_gap_multiplier * avg_gap

            # find consecutive row_breaks that differ by >= para_gap
            # the region between them is a "paragraph break region"
            for i in range(len(gaps)):
                if gaps[i] >= para_gap:
                    # paragraph break region from line_break_rows[i] 
                    # to line_break_rows[i+1]
                    start_y = line_break_rows[i]
                    end_y = line_break_rows[i+1]
                    paragraph_break_ranges.append((start_y, end_y))

    # 6) Visualize line breaks (thin green horizontal lines) & paragraph breaks (thick red lines).
    annotated = image.copy()

    # Draw line breaks
    for row_y in line_break_rows:
        cv2.line(annotated, (0, row_y), (img_w, row_y), (0, 255, 0), 1)

    # Draw paragraph breaks
    for (start_y, end_y) in paragraph_break_ranges:
        # We'll draw a thicker red line in the midpoint or at start or something.
        # E.g. let's draw a 2-pixel thick line at the start of the paragraph break region:
        cv2.line(annotated, (0, start_y), (img_w, start_y), (0, 0, 255), 2)

    # Or optionally fill a translucent rectangle from start_y to end_y
    # but that's more advanced. We'll keep it simple with lines for now.

    cv2.imwrite(output_path, annotated)

    return line_break_rows, paragraph_break_ranges

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
    # Existing pipeline parameters:
    min_area=50,
    max_area=50000,
    folio_side="recto",
    nbins=50,
    percentile=50,
    marginal_width_fraction=0.2,
    facing_folio_fraction=0.15,
    # New line/paragraph detection parameters:
    min_lines=20,
    max_lines=40,
    paragraph_gap_multiplier=2.0,
    smoothing_kernel=5,
    # Possibly pass in your color_map or other advanced parameters
):
    """
    1) Extract connected components from 'image_path'.
    2) Classify them (text vs. margin vs. facing_folio, etc.).
    3) Filter to only text-labeled bounding boxes => 'text_bboxes'.
    4) Run detect_vertical_line_paragraph_breaks(...) on text_bboxes to find
       line_break_rows + paragraph_break_ranges.
    5) Merge all text_bboxes into a single bounding rectangle. If no paragraph 
       breaks => keep as is. Otherwise, subdivide that rectangle at each 
       line break that occurs right after a paragraph break region.
    6) Return the resulting bounding boxes, also visualize them in a final image.
    """

    ##### Step A: Extract connected components #####    
    all_bboxes = extract_connected_components(
        image_path,
        min_area=min_area,
        max_area=max_area
    )

    ##### Step B: Classify bounding boxes #####    
    text_bboxes = classify_components_single_pass(
        image_path=image_path,
        bboxes=all_bboxes,
        output_path="classification_intermediate.png",  # intermediate result
        folio_side=folio_side,
        nbins=nbins,
        percentile=percentile,
        facing_folio_fraction=facing_folio_fraction,
        narrow_width_fraction=marginal_width_fraction,  
    )
    # 'text_bboxes' are the bounding boxes labeled "text_region" by the classifier

    if not text_bboxes:
        # No text => nothing to do
        final_img = cv2.imread(image_path)
        if final_img is not None:
            cv2.imwrite(output_path, final_img)
        print("No text regions found.")
        return []

    ##### Step C: Detect lines & paragraphs #####
    # We'll use your new function, e.g.: detect_vertical_line_paragraph_breaks    
    line_breaks, paragraph_breaks = detect_vertical_line_paragraph_breaks(
        image_path=image_path,
        text_bboxes=text_bboxes,
        min_lines=min_lines,
        max_lines=max_lines,
        paragraph_gap_multiplier=paragraph_gap_multiplier,
        smoothing_kernel=smoothing_kernel,
        output_path="line_paragraph_annotated.png"  # debug visualization
    )

    ##### Step D: Merge all text boxes into one big bounding region #####
    # (If you prefer to handle multiple disjoint blocks, do so. For simplicity, 
    #  we unify them all into minX..maxX, minY..maxY.)
    min_x = min(bx for (bx,by,bw,bh) in text_bboxes)
    max_x = max(bx + bw for (bx,by,bw,bh) in text_bboxes)
    min_y = min(by for (bx,by,bw,bh) in text_bboxes)
    max_y = max(by + bh for (bx,by,bw,bh) in text_bboxes)

    big_region = (min_x, min_y, max_x - min_x, max_y - min_y)

    # If no paragraph breaks => we skip subdividing
    if not paragraph_breaks:
        # Just return this single region
        final_boxes = [big_region]
    else:
        # Subdivide: For each paragraph break range (start_y, end_y),
        # we find the line break row that is strictly > end_y => 
        # that's our horizontal cut. 
        # We'll gather these cut rows, then produce sub-bounding boxes from 
        # top->cut1, cut1->cut2, etc.
        cut_rows = []
        for (start_y, end_y) in paragraph_breaks:
            # find the first line break row that is > end_y
            next_line_break = None
            for lb in line_breaks:
                if lb >= end_y:
                    next_line_break = lb
                    break
            if next_line_break is not None:
                cut_rows.append(next_line_break)

        # Now produce sub-bounding boxes from min_y..cut1, cut1..cut2, ... last..max_y
        cut_rows = sorted(set(cut_rows))  # unique & sorted
        final_boxes = []
        current_top = min_y
        for cr in cut_rows:
            if cr <= current_top or cr >= max_y:
                # skip invalid or out-of-bounds
                continue
            sub_height = cr - current_top
            final_boxes.append((min_x, current_top, max_x - min_x, sub_height))
            current_top = cr
        # remainder
        if current_top < max_y:
            final_boxes.append((min_x, current_top, max_x - min_x, max_y - current_top))

    ##### Step E: Visualize the final boxes #####
    final_img = cv2.imread(image_path)
    if final_img is None:
        raise IOError(f"Could not reload {image_path}")

    # Let's draw them in green
    for (bx,by,bw,bh) in final_boxes:
        cv2.rectangle(final_img, (bx,by), (bx+bw, by+bh), (0,255,0), 3)

    cv2.imwrite(output_path, final_img)

    print(f"Final subdivided text bounding boxes saved to {output_path}.")
    return final_boxes

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
        facing_folio_fraction=0.15
    )

    print(f"Found {len(text_boxes)} bounding boxes in main text region. Classification visualization saved to {output_path}")


