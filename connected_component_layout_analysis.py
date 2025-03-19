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
    output_path="line_paragraph_breaks.png",
    low_line_density_threshold=0.25
):
    """
    Identifies line breaks (local minima) in vertical bounding-box density
    and then flags some breaks as paragraph breaks. Paragraph breaks
    are determined by:
      1) A large vertical gap >= paragraph_gap_multiplier * average_gap, or
      2) A line whose density is < (low_line_density_threshold * mean_line_density).
         The line break above that line is considered a paragraph break.
         If multiple consecutive line breaks qualify, only the bottom-most is used.

    Steps:
      1) Build a row_density array by incrementing for each row spanned by bounding boxes.
      2) Smooth row_density. Then find local minima => line breaks.
      3) Adapts prominence to achieve ~ [min_lines, max_lines] lines if possible.
      4) For each line i, measure line_density[i], the average row_density from line_break[i].. line_break[i+1].
         Compute mean_line_density across all lines.
      5) Mark line i as a potential paragraph break if:
           (gap[i] >= paragraph_gap_multiplier*avg_gap) OR
           (line_density[i+1] < low_line_density_threshold * mean_line_density)
         (the second condition means "the line immediately below line_break[i] has low density")
      6) If multiple consecutive line breaks are flagged, only the bottom-most is used.
      7) Draw line breaks (thin green) & paragraph breaks (thicker red lines) on the image, saving to output_path.

    :param image_path: path to the input image.
    :param text_bboxes: bounding boxes belonging to the main text region.
    :param min_lines, max_lines: target range for line count.
    :param paragraph_gap_multiplier: factor to mark big vertical gaps as paragraphs.
    :param smoothing_kernel: size for smoothing row_density.
    :param output_path: path to save the annotated result.
    :param low_line_density_threshold: fraction of average line density 
                                       below which a line is flagged as a paragraph break line.
                                       default 0.5 => 50%.
    :return: (line_break_rows, paragraph_break_rows)
             line_break_rows => list of row indices for line breaks
             paragraph_break_ranges => list of (start_y, end_y) for paragraphs.
    """

    # 1) Load image & get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")
    img_h, img_w = image.shape[:2]

    # 2) Build a vertical row_density array
    row_density = np.zeros(img_h, dtype=np.int32)
    for (bx, by, bw, bh) in text_bboxes:
        top = max(0, by)
        bot = min(img_h, by + bh)
        row_density[top:bot] += 1

    # 3) Smooth row_density with a small moving average
    if smoothing_kernel>1:
        kernel = np.ones(smoothing_kernel, dtype=np.float32)/smoothing_kernel
        smoothed = np.convolve(row_density, kernel, mode='same')
    else:
        smoothed = row_density.astype(np.float32)

    # We'll invert it to find line breaks as local maxima in -smoothed (valleys in smoothed).
    neg_smoothed = -smoothed

    # 4) Use find_peaks to detect line breaks as local maxima in neg_smoothed
    from scipy.signal import find_peaks
    def detect_minima_with_adaptation(neg_signal, min_lines, max_lines):
        # We'll adapt 'prominence' to get #peaks in [min_lines, max_lines] if possible
        distance_guess = 10
        prominence_guess = 5
        found_peaks = []
        max_iter = 20

        for _ in range(max_iter):
            peaks, _ = find_peaks(neg_signal, distance=distance_guess, prominence=prominence_guess)
            count = len(peaks)
            if min_lines <= count <= max_lines:
                return peaks
            # If too many => increase prominence => fewer peaks
            if count> max_lines:
                prominence_guess += 2
            elif count< min_lines and prominence_guess>0:
                # too few => reduce prominence
                prominence_guess = max(0, prominence_guess - 1)
            else:
                # not converging quickly => keep adjusting
                pass

        return peaks  # fallback

    line_breaks = detect_minima_with_adaptation(neg_smoothed, min_lines, max_lines)
    line_breaks = np.sort(line_breaks).tolist()  # row indices

    if len(line_breaks) <2:
        # trivial => no lines or 1 line => no paragraphs
        # we'll draw them & return
        annotated = image.copy()
        for row_y in line_breaks:
            cv2.line(annotated, (0,row_y), (img_w,row_y), (0,255,0), 1) # green line
        cv2.imwrite(output_path, annotated)
        return line_breaks, []

    # 5) Identify paragraph breaks. 
    #    We measure gap[i] = line_breaks[i+1] - line_breaks[i].
    #    We also measure the "line density" for each line => average row_density from 
    #    line_breaks[i].. line_breaks[i+1].
    #    Then compute mean_line_density, see which lines are < threshold => potential paragraph.

    # line gaps
    gaps = []
    for i in range(len(line_breaks)-1):
        gap = line_breaks[i+1] - line_breaks[i]
        gaps.append(gap)
    avg_gap = np.mean(gaps) if gaps else 0

    # line densities => for line i in [i, i+1)
    # We'll define line_density[i] => average smoothed or average row_density?
    # We'll do average row_density in that segment. 
    # We can do average(smoothed) or sum(row_density)/ length. 
    # We'll do sum(row_density)/ line_height for consistency.
    line_densities = []
    for i in range(len(line_breaks)-1):
        top = line_breaks[i]
        bot = line_breaks[i+1]
        segment = row_density[top:bot]  # integer
        if len(segment)>0:
            avg_line_val = np.mean(segment)
        else:
            avg_line_val = 0
        line_densities.append(avg_line_val)

    mean_line_density = np.mean(line_densities) if len(line_densities)>0 else 0

    # Potential paragraph breaks
    potential_parabreaks = set()
    for i in range(len(line_breaks)-1):
        # Condition A: gap >= paragraph_gap_multiplier * avg_gap
        big_gap = (gaps[i] >= paragraph_gap_multiplier * avg_gap) if avg_gap>0 else False

        # Condition B: line_densities[i+1] < low_line_density_threshold * mean_line_density
        #   i.e. the "line" immediately below break i has low density
        #   if i+1 == len(line_densities), there's no line after break i => skip
        low_density = False
        if i+1 < len(line_densities) and mean_line_density>0:
            if line_densities[i+1] < low_line_density_threshold* mean_line_density:
                low_density = True

        if big_gap or low_density:
            potential_parabreaks.add(i)

    # Now we only keep the bottom-most break in consecutive sequences
    # We treat line breaks in 'potential_parabreaks' as consecutive if i and i+1 are in the set
    # We'll do a pass scanning from top to bottom:
    # if we see consecutive i, i+1 => only keep i+1
    paragraph_break_indices = []
    sorted_pb = sorted(potential_parabreaks)
    skip_next = False
    for idx in range(len(sorted_pb)):
        if skip_next:
            skip_next = False
            continue
        current_i = sorted_pb[idx]
        if idx+1< len(sorted_pb):
            next_i = sorted_pb[idx+1]
            if next_i == current_i+1:
                # consecutive => only keep next_i => skip this one
                paragraph_break_indices.append(next_i)
                skip_next = True
            else:
                paragraph_break_indices.append(current_i)
        else:
            paragraph_break_indices.append(current_i)

    # paragraph_break_indices now are the line-break indexes (the i in line_breaks[i], line_breaks[i+1]) 
    # that mark paragraphs.

    # We define paragraph_break_rows in terms of (start_y, end_y). We interpret "the region from 
    # line_breaks[i].. line_breaks[i+1]" as a paragraph gap. We'll store each as (start_row, end_row).
    # The user previously wanted e.g. "start_y= line_break_rows[i], end_y= line_break_rows[i+1]" 
    # for paragraphs. We'll do that:
    paragraph_break_ranges = []
    for i in paragraph_break_indices:
        start_y = line_breaks[i]
        if i+1< len(line_breaks):
            end_y = line_breaks[i+1]
            paragraph_break_ranges.append((start_y, end_y))

    # 6) Visualize line breaks & paragraph breaks
    annotated = image.copy()

    # draw line breaks in green
    for row_y in line_breaks:
        cv2.line(annotated, (0,row_y), (img_w,row_y), (0,255,0), 1)

    # draw paragraph breaks => thicker red line at the top of the break range
    for (start_y, end_y) in paragraph_break_ranges:
        cv2.line(annotated, (0,start_y), (img_w,start_y), (0,0,255), 2)

    cv2.imwrite(output_path, annotated)

    return line_breaks, paragraph_break_ranges

def filter_large_and_contained_components(
    bboxes,
    classifications,
    max_component_size=50000
):
    """
    For each bounding box whose area > max_component_size, mark it as "other."
    Then, for every other bounding box that is fully contained by this large bounding box,
    also mark it as "other."

    :param bboxes: list of (x, y, w, h) bounding boxes.
    :param classifications: parallel list of labels (strings). This function modifies it in-place.
    :param max_component_size: area threshold above which a component is considered "large."
    """
    n = len(bboxes)
    # 1) Identify large bounding boxes
    large_indices = []
    for i, (bx, by, bw, bh) in enumerate(bboxes):
        area = bw * bh
        if area > max_component_size:
            # Mark it as other
            classifications[i] = "other"
            large_indices.append(i)

    # 2) For each large bounding box, mark any boxes fully contained by it as "other."
    for li in large_indices:
        (Lx, Ly, Lw, Lh) = bboxes[li]
        # bounding box corners for the large box
        Lx2 = Lx + Lw
        Ly2 = Ly + Lh

        for j, (bx, by, bw, bh) in enumerate(bboxes):
            # skip if j is already "other"
            if classifications[j] == "other":
                continue

            # check containment: top-left >= large box's top-left
            # and bottom-right <= large box's bottom-right
            jx2 = bx + bw
            jy2 = by + bh
            if (bx >= Lx) and (by >= Ly) and (jx2 <= Lx2) and (jy2 <= Ly2):
                # fully contained => mark other
                classifications[j] = "other"

def compute_xaxis_density(
    bboxes,
    image_width,
    nbins=50
):
    """
    Compute a normalized density distribution of bounding-box centers 
    across the x-axis, returning a density_array of length nbins in [0..1].
    1) We bin each bounding box's center x-coordinate in [0..image_width].
    2) We increment bin_counts[bin_idx].
    3) We scale the bin_counts by dividing by max_count so the resulting
       density_array is in [0..1].
    """
    bin_width = float(image_width)/nbins

    bin_counts = np.zeros(nbins, dtype=np.float32)
    for (bx, by, bw, bh) in bboxes:
        cx = bx + bw/2.0
        bin_idx = int(cx // bin_width)
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= nbins:
            bin_idx = nbins - 1
        bin_counts[bin_idx] += 1.0

    max_count = np.max(bin_counts)
    if max_count > 0:
        density_array = bin_counts / max_count
    else:
        # no bounding boxes or all zero => everything is 0
        density_array = bin_counts
    return density_array

def density_to_color(density):
    """
    Convert a density in [0..1] to a BGR color.
    We'll do a simple linear blend from blue -> red, ignoring green.
    0 => pure blue, 1 => pure red.
    """
    b = int((1.0 - density)*255)
    r = int(density*255)
    g = 0
    return (b, g, r)

def classify_components_xdensity_with_scenario(
    scenario,  # "unbound", "recto", or "verso"
    image_path,
    bboxes,
    image_width,
    image_height,
    nbins=50,
    smooth_kernel=5,
    sharpness_threshold=0.2,
    xdensity_heatmap_path=None,
    margin_label="margin",
    text_label="text_region",
    facing_label="facing_folio"
):
    """
    Classify bounding boxes based on x-axis density, using prior knowledge 
    of whether the image is "unbound", "recto", or "verso."

    - "unbound": expects exactly 1 disjuncture in the left portion => margin < xcut => text >= xcut
    - "recto": expects exactly 2 disjunctures in the left portion => 
               facing < xcut1 < margin < xcut2 < text
    - "verso": expects 1 disjuncture in left portion (margin->text), 
               plus 2 disjunctures in right portion (text->margin->facing).
               Then "all bounding boxes to the right of the second disjuncture (in the right portion) are facing."

    :param scenario: one of {"unbound","recto","verso"}
    :param image_path: path to the original image (for heatmap).
    :param bboxes: list of bounding boxes (x,y,w,h).
    :param image_width, image_height: size of the image in pixels.
    :param nbins: how many bins for x-axis density
    :param smooth_kernel: size of moving-average for smoothing
    :param sharpness_threshold: absolute diff threshold for a "sharp" disjuncture
    :param xdensity_heatmap_path: if provided, we save an intermediate bounding-box heatmap.
    :return: classifications, a list of labels in {margin_label,text_label,facing_label}
    """
    n = len(bboxes)
    classifications = [None]*n

    # Step 1) Compute x-density => [0..1]
    density_array = compute_xaxis_density(bboxes, image_width, nbins=nbins)

    # Step 2) Smooth
    if smooth_kernel>1:
        kernel = np.ones(smooth_kernel,dtype=np.float32)/smooth_kernel
        smoothed = np.convolve(density_array, kernel, mode='same')
    else:
        smoothed = density_array

    # Step 3) Diff array => find disjunctures where |diff[i]| >= sharpness_threshold
    diff = np.zeros(len(smoothed)-1, dtype=np.float32)
    for i in range(len(smoothed)-1):
        diff[i] = smoothed[i+1] - smoothed[i]

    bin_width = float(image_width)/ nbins

    # We'll define 3 helper subroutines:
    def find_biggest_disjuncture_in_range(start_bin, end_bin):
        """
        Return the bin index i in [start_bin..end_bin-1] that has the biggest |diff[i]|.
        Return (i, diff[i]) or (None, 0) if invalid.
        """
        i_best = None
        val_best = 0.0
        for i2 in range(start_bin, min(end_bin, len(diff))):
            if abs(diff[i2])> abs(val_best):
                i_best = i2
                val_best = diff[i2]
        return i_best, val_best

    def find_top2_disjunctures_in_range(start_bin, end_bin):
        """
        Return the top two distinct bin indices in [start_bin..end_bin-1] 
        by absolute diff value. If we can't find 2, we return fewer.
        """
        candidates = []
        for i2 in range(start_bin, min(end_bin, len(diff))):
            candidates.append( (i2, diff[i2]) )
        # sort by absolute value
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
        # pick top 2
        return candidates[:2]  # might have 1 if not enough

    def find_topN_disjunctures_in_range(start_bin, end_bin, N=2):
        candidates = []
        for i2 in range(start_bin, min(end_bin, len(diff))):
            candidates.append( (i2, diff[i2]) )
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
        return candidates[:N]

    # We'll define 'left_third' = image_width/3, 'right_third'= 2*image_width/3 => bin indexes
    left_third_bin = int(nbins*(1.0/3.0))
    right_third_bin = int(nbins*(2.0/3.0))

    # We'll define xcuts = []
    # Then label bounding boxes according to scenario.
    xcuts = []

    if scenario=="unbound":
        # we want exactly 1 disjuncture in ~left portion
        # for simplicity, let's search the left half or left third, whichever user wants.
        # let's do the left half => bin < nbins/2
        half_bin = nbins//2
        i_best, val_best = find_biggest_disjuncture_in_range(0, half_bin)
        if i_best is None:
            # fallback => everything text
            for i in range(n):
                classifications[i] = text_label
        else:
            xcut = (i_best+1)*bin_width
            # margin < xcut, text >= xcut
            for idx,(bx,by,bw,bh) in enumerate(bboxes):
                cx = bx + bw/2.0
                if cx< xcut:
                    classifications[idx] = margin_label
                else:
                    classifications[idx] = text_label
            xcuts.append(xcut)

    elif scenario=="recto":
        # we want 2 disjunctures in the left third => facing < xcut1 < margin < xcut2 < text
        # so let's find top2 in [0..left_third_bin], then sort them
        top2 = find_top2_disjunctures_in_range(0, left_third_bin)
        if len(top2)<2:
            # fallback => everything text
            for i in range(n):
                classifications[i] = text_label
        else:
            top2.sort(key=lambda x: x[0])  # sort by bin index
            i1, val1 = top2[0]
            i2, val2 = top2[1]
            xcut1 = (i1+1)*bin_width
            xcut2 = (i2+1)*bin_width
            if xcut2< xcut1:
                xcut1, xcut2 = xcut2, xcut1
            xcuts=[xcut1,xcut2]
            # classify
            for idx,(bx,by,bw,bh) in enumerate(bboxes):
                cx = bx + bw/2.0
                if cx< xcut1:
                    classifications[idx] = facing_label
                elif cx< xcut2:
                    classifications[idx] = margin_label
                else:
                    classifications[idx] = text_label

    elif scenario=="verso":
        # 1 disjuncture in left portion => margin < xcut1 => text >= xcut1
        # 2 disjunctures in right portion => bounding boxes > xcut2 => facing
        # ignoring xcut3 for labeling. 
        # We'll define 'left' portion => left_third_bin or half_bin?
        # user said "the first disjuncture in the leftmost third," "the two additional in the rightmost third."

        # 1 in [0..left_third_bin], 2 in [right_third_bin..(len(diff))]
        i_left, val_left = find_biggest_disjuncture_in_range(0, left_third_bin)
        top2_right = find_topN_disjunctures_in_range(right_third_bin, len(diff), N=2)

        if i_left is None or len(top2_right)<2:
            # fallback => everything text
            for i in range(n):
                classifications[i] = text_label
        else:
            # interpret
            xcut1 = (i_left+1)*bin_width
            top2_right.sort(key=lambda x: x[0]) 
            i2,val2 = top2_right[0]
            i3,val3 = top2_right[1]
            xcut2 = (i2+1)*bin_width
            xcut3 = (i3+1)*bin_width
            if xcut3< xcut2:
                xcut2, xcut3 = xcut3, xcut2
            xcuts=[xcut1,xcut2,xcut3]

            # margin < xcut1, text in [xcut1.. xcut2), facing > xcut2
            for idx,(bx,by,bw,bh) in enumerate(bboxes):
                cx = bx + bw/2.0                
                if cx< xcut1:
                    classifications[idx] = margin_label
                elif cx > xcut2:                                        
                    classifications[idx] = facing_label
                else:                    
                    classifications[idx] = text_label

    else:
        # fallback => everything text
        for i in range(n):
            classifications[i] = text_label

    # Step: produce heatmap if path is given
    if xdensity_heatmap_path is not None:
        heatmap_img = cv2.imread(image_path)
        if heatmap_img is None:
            # fallback => black canvas
            heatmap_img = np.zeros((int(image_height), int(image_width),3), dtype=np.uint8)

        # compute the raw bin-based density for each bounding box
        density_array = compute_xaxis_density(bboxes, image_width, nbins=nbins)
        bin_width = float(image_width)/ nbins
        for (bx,by,bw,bh) in bboxes:
            cx = bx + bw/2.0
            bin_idx = int(cx//bin_width)
            if bin_idx<0: bin_idx=0
            elif bin_idx>= nbins: bin_idx= nbins-1
            dens_val = density_array[bin_idx]
            color = density_to_color(dens_val)
            cv2.rectangle(
                heatmap_img,
                (int(bx), int(by)),
                (int(bx+bw), int(by+bh)),
                color, 2
            )

        # optional: also draw vertical lines at xcuts if found
        for xcut in xcuts:
            cv2.line(heatmap_img, (int(xcut),0), (int(xcut), int(image_height)), (0,255,255), 2)

        cv2.imwrite(xdensity_heatmap_path, heatmap_img)

    return classifications

def postprocess_classifications(
    bboxes,
    classifications,
    scenario="unbound",
    image_width=None,
    image_height=None,
    tile_size_ratio=0.05,
    tile_box_count_threshold=2
):
    """
    1) Reclassify text -> margin if fully left of margin_max_x
    2) If scenario == "verso", reclassify facing_folio -> text if half box is left of text_max_x
    3) New Tiling Step: only for the bounding region covering all text-labeled boxes.
       - Tile the region into tile_size_ratio steps (e.g. 0.05 => 20 tiles).
       - For each tile, count how many text-labeled bounding boxes partially/entirely intersect it.
       - If the count < tile_box_count_threshold => "low coverage tile."
       - Reclassify any text-labeled bounding box that intersects a low-coverage tile => "other."
    """

    # -----------------------------
    # (A) Margin pull-in
    margin_max_x = None
    for (bx, by, bw, bh), label in zip(bboxes, classifications):
        if label == "margin":
            right_edge = bx + bw
            if margin_max_x is None or right_edge > margin_max_x:
                margin_max_x = right_edge

    if margin_max_x is not None:
        for i, ((bx, by, bw, bh), label) in enumerate(zip(bboxes, classifications)):
            if label == "text_region":
                if (bx + bw) <= margin_max_x:
                    classifications[i] = "margin"

    '''# -----------------------------
    # (B) Verso scenario => push-out facing_folio -> text
    if scenario == "verso":
        text_max_x = None
        for (bx, by, bw, bh), label in zip(bboxes, classifications):
            if label == "text_region":
                right_edge = bx + bw
                if text_max_x is None or right_edge > text_max_x:
                    text_max_x = right_edge

        if text_max_x is not None:
            for i, ((bx, by, bw, bh), label) in enumerate(zip(bboxes, classifications)):
                if label == "facing_folio":
                    box_left = bx
                    box_right = bx + bw
                    overlap_start = max(box_left, 0)
                    overlap_end = min(box_right, text_max_x)
                    overlap = 0
                    if overlap_end > overlap_start:
                        overlap = overlap_end - overlap_start
                    if overlap >= bw / 2.0:
                        classifications[i] = "text_region"'''

    # -----------------------------
    # (C) Tiling Step with bounding region of text-labeled boxes,
    #     counting # of text boxes in each tile, not coverage area.

    if image_width is not None and image_height is not None:
        # 1) Gather all text-labeled boxes
        text_boxes = [
            (bx, by, bw, bh)
            for (bx, by, bw, bh), lab in zip(bboxes, classifications)
            if lab == "text_region"
        ]
        if text_boxes:
            min_x_text = min(bx for (bx, by, bw, bh) in text_boxes)
            max_x_text = max(bx + bw for (bx, by, bw, bh) in text_boxes)
            min_y_text = min(by for (bx, by, bw, bh) in text_boxes)
            max_y_text = max(by + bh for (bx, by, bw, bh) in text_boxes)

            region_w = max_x_text - min_x_text
            region_h = max_y_text - min_y_text

            if region_w > 0 and region_h > 0:
                tile_w = region_w * tile_size_ratio
                tile_h = region_h * tile_size_ratio

                num_tiles_x = int(1.0 / tile_size_ratio)  # e.g. 20 if 0.05
                num_tiles_y = int(1.0 / tile_size_ratio)

                # We'll store how many text boxes partially/entirely intersect each tile
                tile_box_count = np.zeros((num_tiles_y, num_tiles_x), dtype=int)

                def overlap_1d(a1, a2, b1, b2):
                    return max(0, min(a2, b2) - max(a1, b1))

                # 2) For each text-labeled box, increment the tile's box count if intersects
                for (bx, by, bw, bh), label in zip(bboxes, classifications):
                    if label == "text_region":
                        # region coords
                        rx1 = bx - min_x_text
                        ry1 = by - min_y_text
                        rx2 = rx1 + bw
                        ry2 = ry1 + bh

                        start_tx = int(rx1 // tile_w)
                        end_tx   = int(rx2 // tile_w)
                        start_ty = int(ry1 // tile_h)
                        end_ty   = int(ry2 // tile_h)

                        for ty in range(start_ty, end_ty + 1):
                            if 0 <= ty < num_tiles_y:
                                tile_top = ty * tile_h
                                tile_bottom = tile_top + tile_h
                                for tx in range(start_tx, end_tx + 1):
                                    if 0 <= tx < num_tiles_x:
                                        tile_left = tx * tile_w
                                        tile_right = tile_left + tile_w
                                        overlap_w = overlap_1d(rx1, rx2, tile_left, tile_right)
                                        overlap_h = overlap_1d(ry1, ry2, tile_top, tile_bottom)
                                        if overlap_w > 0 and overlap_h > 0:
                                            tile_box_count[ty, tx] += 1

                # 3) Identify "low coverage" tiles => tile_box_count < tile_box_count_threshold
                low_tiles = set()
                for ty in range(num_tiles_y):
                    for tx in range(num_tiles_x):
                        if tile_box_count[ty, tx] < tile_box_count_threshold:
                            low_tiles.add((ty, tx))

                # 4) Reclassify text-labeled boxes that intersect these tiles => "other"
                for i, ((bx, by, bw, bh), label) in enumerate(zip(bboxes, classifications)):
                    if label == "text_region":
                        rx1 = bx - min_x_text
                        ry1 = by - min_y_text
                        rx2 = rx1 + bw
                        ry2 = ry1 + bh

                        stx = int(rx1 // tile_w)
                        etx = int(rx2 // tile_w)
                        sty = int(ry1 // tile_h)
                        ety = int(ry2 // tile_h)

                        intersects_low_tile = False
                        for ty in range(sty, ety+1):
                            if 0 <= ty < num_tiles_y:
                                tile_top = ty*tile_h
                                tile_bottom = tile_top+tile_h
                                for tx in range(stx, etx+1):
                                    if 0 <= tx < num_tiles_x:
                                        if (ty, tx) in low_tiles:
                                            # check overlap is non-zero
                                            tile_left = tx* tile_w
                                            tile_right= tile_left+ tile_w
                                            overlap_w = overlap_1d(rx1, rx2, tile_left, tile_right)
                                            overlap_h = overlap_1d(ry1, ry2, tile_top, tile_bottom)
                                            if overlap_w>0 and overlap_h>0:
                                                intersects_low_tile = True
                                                break
                                if intersects_low_tile:
                                    break
                        if intersects_low_tile:
                            classifications[i] = "other"

    return classifications

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
    wide_ratio=5.0,
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

    # Load the image for drawing
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load {image_path}")
    h, w = image.shape[:2]
    max_component_size = .03 * h * w
    
    # 1) Call classify_components_by_xaxis_density
    classifications = classify_components_xdensity_with_scenario(
        folio_side,
        image_path,
        bboxes,
        w,
        h,
        nbins=50,
        smooth_kernel=0,
        sharpness_threshold=0.2,
        xdensity_heatmap_path="coverage_heatmap.png",
        margin_label="margin",
        text_label="text_region",
        facing_label="facing_folio"
    )    

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
            if bh > 0:
                width_ratio = bw / float(bh)
                if width_ratio > wide_ratio:
                    classifications[i] = "other"
                    continue                           

    filter_large_and_contained_components(bboxes=bboxes, classifications=classifications, max_component_size=max_component_size)

    classifications = postprocess_classifications(
        bboxes,
        classifications,
        scenario=folio_side,
        image_height=h,
        image_width=w,
        tile_box_count_threshold=2,
        tile_size_ratio=0.1
    )            

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

def extract_connected_components(image_path, min_area=100, max_area=50000):
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

    # 1) Load image & get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")
    img_h, img_w = image.shape[:2]
    min_area = .00001 * img_h * img_w
    max_area = .001 * img_h * img_w

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
        paragraph_gap_multiplier=1.5,
        smoothing_kernel=5,
        output_path="line_paragraph_annotated.png",  # debug visualization
        low_line_density_threshold=.25
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
    image_path = "images/585912-0092.jpg"
    output_path = "classification_with_spine_detection.png"

    text_boxes = finalize_text_bounding_boxes(
        image_path,
        output_path,        
        # classification parameters
        folio_side="recto",
        nbins=50,
        percentile=50,
        marginal_width_fraction=0.2,
        facing_folio_fraction=0.15
    )

    print(f"Found {len(text_boxes)} bounding boxes in main text region. Classification visualization saved to {output_path}")


