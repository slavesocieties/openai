import cv2
import numpy as np
import matplotlib.pyplot as plt

def drop_sparse_columns(bboxes, image_width, percentile=50, nbins=50):
    """
    Discard bounding boxes whose center lies in 'sparse' columns.
    Builds a 1D histogram of bounding-box centers across 'nbins' columns.
    Finds the bin-count threshold at 'percentile' and discards boxes
    in bins below that threshold.
    """
    if not bboxes:
        return bboxes

    bins = np.zeros(nbins, dtype=int)
    for x, y, w, h in bboxes:
        cx = x + w / 2.0
        bin_idx = int(cx * nbins / image_width)
        bin_idx = min(bin_idx, nbins - 1)
        bins[bin_idx] += 1

    sorted_counts = np.sort(bins)
    cutoff_index = int((percentile / 100.0) * nbins)
    cutoff_index = max(0, min(cutoff_index, nbins - 1))
    threshold_value = sorted_counts[cutoff_index]

    filtered = []
    for x, y, w, h in bboxes:
        cx = x + w / 2.0
        bin_idx = int(cx * nbins / image_width)
        bin_idx = min(bin_idx, nbins - 1)
        if bins[bin_idx] >= threshold_value:
            filtered.append([x, y, w, h])
    return filtered

def merge_bboxes_no_giant(
    bboxes,
    merge_threshold,
    image_width,
    image_height,
    max_fraction=2.0/3.0
):
    """
    Merges bounding boxes that overlap or come within 'merge_threshold' pixels
    (horizontally or vertically), EXCEPT if merging would produce a box
    whose area exceeds 'max_fraction' of the total image area.

    Returns a new list of merged bboxes.
    """
    if not bboxes:
        return []

    image_area = float(image_width * image_height)    

    merged = True
    while merged:
        merged = False
        new_bboxes = []
        taken = [False]*len(bboxes)

        for i in range(len(bboxes)):
            if taken[i]:
                continue
            boxA = bboxes[i]
            Ax, Ay, Aw, Ah = boxA
            A_x2 = Ax + Aw
            A_y2 = Ay + Ah

            for j in range(i+1, len(bboxes)):
                if taken[j]:
                    continue
                Bx, By, Bw, Bh = bboxes[j]
                B_x2 = Bx + Bw
                B_y2 = By + Bh

                # Check if A and B overlap or come within 'merge_threshold'
                if not (
                    Bx > A_x2 + merge_threshold or
                    B_x2 < Ax - merge_threshold or
                    By > A_y2 + merge_threshold or
                    B_y2 < Ay - merge_threshold
                ):
                    # Potential new bounding box if we merged them
                    newX = min(Ax, Bx)
                    newY = min(Ay, By)
                    newX2 = max(A_x2, B_x2)
                    newY2 = max(A_y2, B_y2)
                    newW = newX2 - newX
                    newH = newY2 - newY
                    newArea = newW * newH                    

                    # Check if this new bounding box is allowed (<= max_fraction * image_area)
                    if newArea <= max_fraction * image_area:                        
                        # Merge them
                        boxA = [newX, newY, newW, newH]
                        Ax, Ay, Aw, Ah = boxA
                        A_x2 = Ax + Aw
                        A_y2 = Ay + Ah

                        taken[j] = True
                        merged = True
                    # else: skip merging these two, proceed to check next box j
                    else:
                        print("bounding box too big")

            new_bboxes.append(boxA)
            taken[i] = True

        bboxes = new_bboxes

    return bboxes

def check_credible_result(bboxes, box_count_range):
    """
    Simple credibility check: the bounding-box count must be within box_count_range.
    """
    n = len(bboxes)
    min_count, max_count = box_count_range
    if n < min_count:
        return (False, "count_too_low")
    if n > max_count:
        return (False, "count_too_high")
    return (True, "ok")

def adaptive_merge_pipeline(
    bboxes,
    image_width,
    image_height,
    box_count_range=(2, 4),
    initial_merge_threshold=10,
    threshold_min=0,
    threshold_max=200,
    max_iterations=10
):
    """
    Adjusts 'merge_threshold' until the bounding-box count is in [min_count, max_count]
    or we exhaust max_iterations. Over-large merges are disallowed at the merge stage
    (merge_bboxes_no_giant).
    """
    merge_threshold = initial_merge_threshold
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1

        merged = merge_bboxes_no_giant(
            bboxes,
            merge_threshold,
            image_width,
            image_height,
            max_fraction=1
        )
        is_credible, reason = check_credible_result(merged, box_count_range)

        if is_credible:
            return merged, merge_threshold, True

        # Not credible => interpret reason
        if reason == "count_too_high":
            # Not merged enough => increase threshold
            merge_threshold = min(threshold_max, merge_threshold + 5)
        elif reason == "count_too_low":
            # Merged too aggressively => decrease threshold
            merge_threshold = max(threshold_min, merge_threshold - 5)
        else:
            # Just in case
            merge_threshold = max(threshold_min, merge_threshold - 5)

    # If we exit the loop, we return the final attempt
    final_merged = merge_bboxes_no_giant(
        bboxes,
        merge_threshold,
        image_width,
        image_height,
        max_fraction=1
    )
    is_credible, _ = check_credible_result(final_merged, box_count_range)
    return final_merged, merge_threshold, is_credible

def classify_facing_folio_bboxes(merged_boxes):
    """
    Identify facing-folio bounding boxes if:
      (1) The box's width is < 0.5 * (width of the widest merged box).
      (2) The box does not share x-coordinates with that widest box,
          i.e. it is completely to the left or completely to the right.

    Returns a list of the bounding boxes that satisfy these conditions.
    """
    if not merged_boxes:
        return []

    # 1) Find the bounding box with the maximum width
    max_width = 0
    max_index = -1
    for i, (bx, by, bw, bh) in enumerate(merged_boxes):
        if bw > max_width:
            max_width = bw
            max_index = i

    # Edge case: if no boxes found (shouldn't happen, but just in case)
    if max_index < 0:
        return []

    # Get the bounding box with the max width
    x_wide, y_wide, w_wide, h_wide = merged_boxes[max_index]
    left_wide = x_wide
    right_wide = x_wide + w_wide

    # 2) Classify boxes as facing folio if they meet both conditions
    folio_bboxes = []
    for (x, y, w, h) in merged_boxes:
        # Condition 1: width < 0.5 * w_wide
        width_condition = (w < 0.5 * w_wide)

        # Condition 2: no horizontal overlap with the widest box
        # i.e. the entire bounding box is left_of or right_of the widest box
        no_overlap_condition = ((x + w) < left_wide) or (x > right_wide)

        if width_condition and no_overlap_condition:
            folio_bboxes.append((x, y, w, h))

    return folio_bboxes

def remove_components_inside_bboxes(original_components, removal_bboxes):
    """
    Removes all connected-component bounding boxes that lie entirely within 
    any bounding box in 'removal_bboxes'.
    """
    if not removal_bboxes:
        return original_components

    filtered = []
    for (cx, cy, cw, ch) in original_components:
        top_left = (cx, cy)
        bottom_right = (cx + cw, cy + ch)
        inside = False

        for (rx, ry, rw, rh) in removal_bboxes:
            r_x2 = rx + rw
            r_y2 = ry + rh
            # Check if (cx, cy, cw, ch) is fully inside (rx, ry, rw, rh)
            if (cx >= rx and cy >= ry and
                (cx + cw) <= r_x2 and (cy + ch) <= r_y2):
                inside = True
                break

        if not inside:
            filtered.append([cx, cy, cw, ch])

    return filtered

def detect_internal_signature(image_array, num_consecutive_rows=50):
    """
    Find consecutive rows in a grayscale image where at least 90% of the pixels
    in the first half of each row and 90% in the last 10% of the row are white.
    Used to identify signature blocks in images likely to contain multiple 
    sacramental records but that layout analysis has failed to separate.
    
    Parameters:
    - image_array: a grayscale image as a numpy array, shape=(H, W).
    - num_consecutive_rows: The number of consecutive rows that need to meet the criteria.

    Returns:
    - A list of (start_idx, end_idx) row intervals (inclusive) representing
      signature regions.
    """
    image_array = np.asarray(image_array, dtype=np.uint8)

    # 1) Binarize the grayscale image via a quantile-based threshold
    binarization_quantile = 0.1
    bin_thresh = np.quantile(image_array, binarization_quantile)
    # Below threshold => 0 (black), else => 1 (white)
    bin_img = np.where(image_array <= bin_thresh, 0, 1)

    rows, cols = bin_img.shape

    # 2) Thresholds for the number of white pixels in the first half + last 10%
    first_half_threshold = 0.9 * (cols // 2)
    last_10_percent_threshold = 0.9 * (cols // 10)

    # 3) criteria_met[i] => row i meets the signature criteria
    criteria_met = np.zeros(rows, dtype=bool)

    for i in range(rows):
        first_half_count = np.sum(bin_img[i, : (cols // 2)])
        last_10percent_count = np.sum(bin_img[i, -int(np.ceil(cols * 0.1)) : ])
        if (first_half_count >= first_half_threshold) and (last_10percent_count >= last_10_percent_threshold):
            criteria_met[i] = True

    # 4) Identify sequences of >= num_consecutive_rows
    consecutive_sequences = []
    start_index = None

    for i in range(rows):
        if criteria_met[i] and start_index is None:
            start_index = i
        elif (not criteria_met[i]) and (start_index is not None):
            if i - start_index >= num_consecutive_rows:
                consecutive_sequences.append((start_index, i - 1))
            start_index = None    
    
    print(consecutive_sequences)
    return consecutive_sequences

def split_on_detected_signatures(
    bboxes,
    grayscale_image,
    min_consecutive_rows=50,
    bottom_margin=100
):
    """
    Given final bounding boxes from the second pass, use detect_internal_signature 
    to identify 'signature blocks' inside each bounding box. For each signature block, 
    we split the bounding box horizontally at the BOTTOM of that block, provided it's 
    not too close to the bounding box's bottom (i.e., 'internal').

    :param bboxes: list of (x, y, w, h) bounding boxes (final from pass 2).
    :param grayscale_image: the original grayscale image (not binarized) 
                            used for detect_internal_signature.
    :param min_consecutive_rows: num_consecutive_rows passed to detect_internal_signature.
    :param bottom_margin: skip splitting if the signature region extends 
                          too close to the bottom of the bounding box.

    :return: A new list of bounding boxes, where any containing an internal 
             signature region is split into top+bottom parts.
    """
    updated_bboxes = []
    img_h, img_w = grayscale_image.shape[:2]

    for (x, y, w, h) in bboxes:
        # Extract sub-image from the grayscale
        sub_gray = grayscale_image[y : y + h, x : x + w]

        # Find signature blocks in this sub-image
        sig_blocks = detect_internal_signature(sub_gray, num_consecutive_rows=min_consecutive_rows)

        if not sig_blocks:
            # No signatures => keep bounding box as-is
            updated_bboxes.append((x, y, w, h))
            continue

        # We'll store the y-splits for each valid signature region
        splits = []
        for (start_idx, end_idx) in sig_blocks:
            # The 'bottom' of the signature region is end_idx
            # Check if it is at least 'bottom_margin' rows from bounding box's bottom
            if end_idx < (h - bottom_margin):
                splits.append(end_idx)        
        
        if not splits:
            # If all signature blocks are near the bottom, we do not split
            updated_bboxes.append((x, y, w, h))
        else:
            # We create multiple sub-bounding boxes
            top_row = 0
            for s in sorted(splits):
                region_height = s - top_row
                updated_bboxes.append((x, y + top_row, w, region_height))
                # Next region starts right after s
                top_row = s + 1

            # Finally, append whatever is left from the last split to the bottom
            if top_row < h:
                region_height = h - top_row
                updated_bboxes.append((x, y + top_row, w, region_height))

    return updated_bboxes

def two_pass_pipeline(
    image_path,
    min_area=50,
    max_area=50000,
    column_percentile=50,
    nbins=50,
    initial_merge_threshold=10,
    box_count_range=(2, 4),
    max_iterations=10,
    remove_margin_notes=True
):
    """
    2-pass approach:
      1) Load image, threshold, find connected components in [min_area, max_area].
      2) (Optional) drop sparse columns => remove margin notes.
      3) Pass 1 => adaptive merge (with merges that exceed 2/3 image area disallowed).
      4) If pass1 merges contain a "big box" > some threshold (like 0.25 * image width),
         or other condition => remove facing folio (classify_facing_folio_bboxes + remove_components_inside_folio_bboxes).
      5) Pass 2 => adaptive merge again => final success/failure check.

    The difference from prior code:
      - We use merge_bboxes_no_giant in the adaptive merge pipeline,
        so no bounding box can exceed 2/3 of the total image area.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Cannot load {image_path}")
    img_h, img_w = image.shape[:2]    

    # --- Preprocess ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- Connected components ---
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    original_bboxes = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            original_bboxes.append([x, y, w, h])

    # --- (Optional) margin-note removal ---
    if remove_margin_notes:
        bboxes_pass1 = drop_sparse_columns(
            bboxes=original_bboxes,
            image_width=img_w,
            percentile=column_percentile,
            nbins=nbins
        )
    else:
        bboxes_pass1 = original_bboxes

    # --- Pass 1 => adaptive merge ---
    merged_pass1, thr_pass1, success_pass1 = adaptive_merge_pipeline(
        bboxes_pass1,
        image_width=img_w,
        image_height=img_h,
        box_count_range=box_count_range,
        initial_merge_threshold=initial_merge_threshold,
        max_iterations=max_iterations
    )    

    # Facing folio removal logic: if user so desires, check if there's a big box > 0.25 * image_width, etc.
    # For demonstration, let's do the same approach as before:
    found_big_box = any((bw > 0.25 * img_w) for (bx, by, bw, bh) in merged_pass1)
    if found_big_box:
        folio_boxes = classify_facing_folio_bboxes(merged_pass1)  # user-defined logic
        bboxes_pass2_input = remove_components_inside_bboxes(bboxes_pass1, folio_boxes)
    else:
        bboxes_pass2_input = bboxes_pass1

    # --- Pass 2 => adaptive merge ---
    merged_pass2, thr_pass2, success_pass2 = adaptive_merge_pipeline(
        bboxes_pass2_input,
        image_width=img_w,
        image_height=img_h,
        box_count_range=box_count_range,
        initial_merge_threshold=initial_merge_threshold,
        max_iterations=max_iterations
    )

    folio_boxes = classify_facing_folio_bboxes(merged_pass2)
    merged_pass2 = remove_components_inside_bboxes(merged_pass2, folio_boxes)
    
    final_boxes = split_on_detected_signatures(
        bboxes=merged_pass2,
        grayscale_image=gray,
        min_consecutive_rows=100  # tweak as needed        
    )
    # final_boxes = merged_pass2
    success_pass2, _ = check_credible_result(final_boxes, box_count_range)

    # Final success = pass2_success
    final_img = image.copy()
    color = (0, 255, 0) if success_pass2 else (0, 0, 255)
    for (bx, by, bw, bh) in final_boxes:
        cv2.rectangle(final_img, (bx, by), (bx + bw, by + bh), color, 2)

    return final_img, final_boxes, thr_pass2, success_pass2

import math

def multi_parameter_search(
    image_path,
    column_percentiles,
    max_iterations_list,
    box_count_range=(1, 4),
    min_area=50,
    max_area=50000,
    nbins=50,
    initial_merge_threshold=10
):
    """
    Tries each combination of 'column_percentile' and 'max_iterations' by calling 
    extract_paragraphs_with_facing_folio_filter. Returns the first combination that converges, 
    or if none converge, the combination whose bounding-box count is closest to the midpoint 
    of box_count_range, erring on the side of fewer boxes in case of a tie.    

    :param image_path: Path to the input image.
    :param column_percentiles: List of possible column_percentile values (e.g. [30, 50, 70]).
    :param max_iterations_list: List of possible max_iterations values (e.g. [5, 10, 15]).
    :param box_count_range: Desired bounding-box count range, (low, high).
    :param min_area: Minimum component area for inclusion.
    :param max_area: Maximum component area for inclusion.
    :param nbins: Number of bins for column-sparsity filtering.
    :param initial_merge_threshold: Starting threshold for merging in adaptive pipeline.
    :return: A tuple (final_image, final_bboxes, best_params, success_flag)
             best_params is (best_column_percentile, best_max_iterations)
    """
    best_result = None
    best_params = None
    best_diff = math.inf
    # For tie-breaking, we prefer fewer bounding boxes (over-merging).
    # We'll track the box_count as well for tie-break logic.

    # Compute the center of the desired range
    # E.g., if box_count_range = (2, 4), midpoint = 3.0
    range_low, range_high = box_count_range
    midpoint = (range_low + range_high) / 2.0    

    # We'll iterate over all combinations in the order they're given.
    for cp in column_percentiles:
        for mi in max_iterations_list:
            print(f"Now testing drop percentile of {cp} and max iterations of {mi}.")
            final_img, final_boxes, thr_used, success = two_pass_pipeline(
                image_path=image_path,
                min_area=min_area,
                max_area=max_area,
                column_percentile=cp,
                nbins=nbins,
                initial_merge_threshold=initial_merge_threshold,
                box_count_range=box_count_range,
                max_iterations=mi
            )

            if success:
                # If we got a successful result, return immediately
                return final_img, final_boxes, (cp, mi), True
            else:
                # Keep track of how close the bounding-box count is to the midpoint
                n = len(final_boxes)
                diff = abs(n - midpoint)
                if diff < best_diff:
                    best_diff = diff
                    best_result = (final_img, final_boxes, (cp, mi), success)
                elif math.isclose(diff, best_diff, abs_tol=1e-9):
                    # Tie: prefer fewer bounding boxes
                    best_n = len(best_result[1])  # bounding-box count in the current best
                    if n < best_n:
                        best_result = (final_img, final_boxes, (cp, mi), success)
                print(f"Second pass failed to converge. Final bounding box count: {len(final_boxes)}.")

    # If no combination converged, return the best we have
    if best_result is not None:
        return best_result
    else:
        # In theory, we should always have some best_result 
        # even if everything fails. But if for some reason 
        # no bounding boxes were processed, fallback:
        blank_img = None
        blank_boxes = []
        return blank_img, blank_boxes, (None, None), False


if __name__ == "__main__":
    # Example usage
    input_path = "images/239746-0078.jpg"
    final_img, final_boxes, thr_used, success = multi_parameter_search(
    input_path,
    [25, 35, 50],
    [10, 25, 50],
    box_count_range=(2, 4),
    min_area=50,
    max_area=50000,
    nbins=50,
    initial_merge_threshold=10
)

    out_path = "facing_folio_filtered_result.png"
    cv2.imwrite(out_path, final_img)
    if success:
        print(f"Converged with threshold={thr_used}, found {len(final_boxes)} bounding boxes.")
        print(final_boxes)
    else:
        print(f"Failed to converge. We ended up with {len(final_boxes)} bounding boxes.")
    print(f"Output saved to {out_path}")