import cv2
import numpy as np

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

def merge_bboxes(bboxes, merge_threshold):
    """
    Merges bounding boxes that overlap or come within 'merge_threshold'
    pixels (horizontally or vertically).
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

                # Check if A and B overlap or are within merge_threshold
                if not (
                    Bx > A_x2 + merge_threshold or
                    B_x2 < Ax - merge_threshold or
                    By > A_y2 + merge_threshold or
                    B_y2 < Ay - merge_threshold
                ):
                    # Merge them into a larger bounding box
                    newX = min(Ax, Bx)
                    newY = min(Ay, By)
                    newX2 = max(A_x2, B_x2)
                    newY2 = max(A_y2, B_y2)
                    boxA = [newX, newY, newX2 - newX, newY2 - newY]

                    # Update for continued merging within this pass
                    Ax, Ay, Aw, Ah = boxA
                    A_x2 = Ax + Aw
                    A_y2 = Ay + Ah
                    taken[j] = True
                    merged = True

            new_bboxes.append(boxA)
            taken[i] = True

        bboxes = new_bboxes
    return bboxes

def check_credible_result(bboxes, box_count_range=(2, 4)):
    """
    Simple check: are we in the desired number of bounding boxes?
    Coverage is no longer considered.
    """
    n = len(bboxes)
    if n >= box_count_range[0] and n <= box_count_range[1]:
        return True
    return False

def adaptive_merge_pipeline(
    bboxes,
    box_count_range=(2, 4),
    initial_merge_threshold=10,
    threshold_min=0,
    threshold_max=200,
    max_iterations=10
):
    """
    Iteratively adjusts 'merge_threshold' to converge on a bounding-box
    count in 'box_count_range'.
    """
    merge_threshold = initial_merge_threshold

    for _ in range(max_iterations):        
        merged_boxes = merge_bboxes(bboxes, merge_threshold)
        n = len(merged_boxes)        

        if check_credible_result(merged_boxes, box_count_range=box_count_range):
            # We have an acceptable box count
            return merged_boxes, merge_threshold, True

        # Adjust threshold
        if n < box_count_range[0]:
            # Too few boxes => we merged too aggressively => reduce threshold
            merge_threshold = max(threshold_min, merge_threshold - 2)
        elif n > box_count_range[1]:
            # Too many boxes => not merged enough => increase threshold
            merge_threshold = min(threshold_max, merge_threshold + 2)
        else:
            # It's in range for box count. 
            # If we're here, it means check_credible_result returned False 
            # for some reason, but that can't happen with only box count logic. 
            # We'll do a small nudge to threshold:
            merge_threshold = min(threshold_max, merge_threshold + 5)

    # If we fail to converge, return the last attempt
    merged_boxes = merge_bboxes(bboxes, merge_threshold)
    return merged_boxes, merge_threshold, False

def classify_facing_folio_bboxes(merged_boxes):
    """
    Instead of aspect ratio, we classify as facing folio any region whose width
    is less than half the width of the widest merged box.
    """
    if not merged_boxes:
        return []

    max_width = max(bw for (bx, by, bw, bh) in merged_boxes)
    folio_bboxes = []
    for (x, y, w, h) in merged_boxes:
        if w < 0.5 * max_width:
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

def extract_paragraphs_with_facing_folio_filter(
    image_path,
    min_area=50,
    max_area=50000,
    column_percentile=35,
    nbins=50,
    initial_merge_threshold=10,
    box_count_range=(2, 4),
    max_iterations=10
):
    """
    1) Load & binarize image, get connected components in [min_area, max_area].
    2) Drop sparse columns (first pass only).
    3) Adaptive merge => get bounding boxes (Pass 1).
    4) Classify "facing folio" boxes as those with width < half the widest box from pass 1.
    5) Remove original connected components that are inside these facing folio boxes.
    6) Adaptive merge again on the filtered set => final bounding boxes (Pass 2).
    7) Return final bounding boxes + visual output.
    """

    # --- Step A: Read & Threshold ---
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")
    img_h, img_w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Optional morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- Step B: Connected Components ---
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    original_components = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            original_components.append([x, y, w, h])

    # --- Step C: Drop Sparse Columns (one-time) ---
    bboxes_pass1 = drop_sparse_columns(original_components, img_w, percentile=column_percentile, nbins=nbins)

    # --- Step D: First Adaptive Merge Pass ---
    merged_pass1, thr_used_1, success_1 = adaptive_merge_pipeline(
        bboxes_pass1,
        box_count_range=box_count_range,
        initial_merge_threshold=initial_merge_threshold,
        max_iterations=max_iterations
    )

    # --- Step E: Identify & Remove Facing-Folio Components ---
    if success_1 and merged_pass1:
        folio_boxes = classify_facing_folio_bboxes(merged_pass1)
        # Remove from original_components the ones inside these folio boxes
        # BUT note: we've already removed margin boxes from original_components
        # to get bboxes_pass1, so we should pass 'bboxes_pass1' to the removal step
        # if we only want to remove from the already-sparse-filtered set.
        # However, the user explicitly said that "the bounding box set passed 
        # to remove_components_inside_folio_bboxes should already have components
        # that were filtered during the initial sparse columns check removed."
        # So let's remove from 'bboxes_pass1' not the entire 'original_components'.
        filtered_components = remove_components_inside_bboxes(bboxes_pass1, folio_boxes)
    else:
        # If the first pass didn't converge or no boxes found, we skip folio removal
        # filtered_components = bboxes_pass1
        print("First pass failed to converge.")
        folio_boxes = classify_facing_folio_bboxes(merged_pass1)
        # Remove from original_components the ones inside these folio boxes
        # BUT note: we've already removed margin boxes from original_components
        # to get bboxes_pass1, so we should pass 'bboxes_pass1' to the removal step
        # if we only want to remove from the already-sparse-filtered set.
        # However, the user explicitly said that "the bounding box set passed 
        # to remove_components_inside_folio_bboxes should already have components
        # that were filtered during the initial sparse columns check removed."
        # So let's remove from 'bboxes_pass1' not the entire 'original_components'.
        filtered_components = remove_components_inside_bboxes(bboxes_pass1, folio_boxes)

    # --- Step F: Second Adaptive Merge Pass (no sparse-column filter now) ---
    merged_pass2, thr_used_2, success_2 = adaptive_merge_pipeline(
        filtered_components,
        box_count_range=box_count_range,
        initial_merge_threshold=initial_merge_threshold,
        max_iterations=max_iterations
    )

    # --- Step G: Visualization ---
    final_img = image.copy()
    color = (0, 255, 0) if success_2 else (0, 0, 255)  # green if success, red if fail
    for bx, by, bw, bh in merged_pass2:
        cv2.rectangle(final_img, (bx, by), (bx + bw, by + bh), color, 2)

    return final_img, merged_pass2, thr_used_2, success_2

import math

def multi_parameter_search(
    image_path,
    column_percentiles,
    max_iterations_list,
    box_count_range=(2, 4),
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
            final_img, final_boxes, thr_used, success = extract_paragraphs_with_facing_folio_filter(
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
    else:
        print(f"Failed to converge. We ended up with {len(final_boxes)} bounding boxes.")
    print(f"Output saved to {out_path}")