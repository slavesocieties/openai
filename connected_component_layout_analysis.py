import cv2
import numpy as np
from scipy.signal import find_peaks
import math

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
                elif cx> xcut2:
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

def postprocess_classifications(bboxes, classifications, scenario="unbound"):
    """
    :param bboxes: list of (x, y, w, h)
    :param classifications: parallel list of strings (e.g. "margin", "text_region", "facing_folio")
    :param scenario: one of {"unbound", "recto", "verso"}
    :return: modifies 'classifications' in-place for the described edge cases
    """

    # 1) Calculate margin_max_x => the maximum (x + w) among components labeled margin
    margin_max_x = None
    for (bx, by, bw, bh), label in zip(bboxes, classifications):
        if label == "margin":
            right_edge = bx + bw
            if margin_max_x is None or right_edge > margin_max_x:
                margin_max_x = right_edge

    # 1a) If we have a margin_max_x, re-classify "text" components that lie entirely to the left
    #     of that margin_max_x => margin
    if margin_max_x is not None:
        for i, ((bx, by, bw, bh), label) in enumerate(zip(bboxes, classifications)):
            if label == "text_region":
                if (bx + bw) <= margin_max_x:
                    classifications[i] = "margin"

    # 2) For scenario="verso", push-out some facing_folio => text
    if scenario == "verso":
        # Find text_max_x => maximum (x + w) among text-labeled components
        text_max_x = None
        for (bx, by, bw, bh), label in zip(bboxes, classifications):
            if label == "text_region":
                right_edge = bx + bw
                if text_max_x is None or right_edge > text_max_x:
                    text_max_x = right_edge

        if text_max_x is not None:
            # re-label facing_folio => text if >= half their width is to the left of text_max_x
            # i.e. overlap( [x, x+w], [0, text_max_x] ) >= w/2
            for i, ((bx, by, bw, bh), label) in enumerate(zip(bboxes, classifications)):
                if label == "facing_folio":
                    box_left = bx
                    box_right = bx + bw
                    # Overlap with [0, text_max_x]
                    overlap = 0.0
                    if box_right > 0 and box_left < text_max_x:
                        overlap_start = max(box_left, 0)
                        overlap_end = min(box_right, text_max_x)
                        if overlap_end > overlap_start:
                            overlap = overlap_end - overlap_start
                    if overlap >= bw/2.0:
                        classifications[i] = "text_region"

    # Return the modified classifications
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
        scenario=folio_side  # or "unbound"/"recto"
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
    image_path = "images/239746-0088.jpg"
    output_path = "classification_with_spine_detection.png"

    text_boxes = finalize_text_bounding_boxes(
        image_path,
        output_path,        
        # classification parameters
        folio_side="verso",
        nbins=50,
        percentile=50,
        marginal_width_fraction=0.2,
        facing_folio_fraction=0.15
    )

    print(f"Found {len(text_boxes)} bounding boxes in main text region. Classification visualization saved to {output_path}")


