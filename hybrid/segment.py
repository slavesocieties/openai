from pool_image import block_image

import numpy as np
import cv2
import matplotlib.pyplot as plt

def binarize_image(gray_image, binarization_quantile=0.1):
    """Binarize the grayscale image using a quantile threshold.

    Parameters
    ----------
    gray_image : numpy array
        Grayscale image.
    binarization_quantile : float
        Quantile to determine the binarization threshold (default is 0.1).

    Returns
    -------
    binary_image : numpy array
        Binarized image.
    """
    # Flatten the image data to compute the quantile threshold
    flattened_data = gray_image.flatten()
    
    # Compute the quantile threshold
    bin_thresh = np.quantile(flattened_data, binarization_quantile)
    
    # Apply the threshold to binarize the image
    binary_image = np.where(gray_image <= bin_thresh, 0, 255).astype(np.uint8)
    
    return binary_image

def segment_image(binary_image, min_area_threshold=50, max_area_threshold=5000, kernel_size=(3, 3), dilation_iterations=1, erosion_iterations=1):
    """Segment the binary image to identify text regions.

    Parameters
    ----------
    binary_image : numpy array
        Binarized image.
    min_area_threshold : int
        Minimum area threshold to filter out small components.
    max_area_threshold : int
        Maximum area threshold to filter out large components.
    kernel_size : tuple
        Size of the kernel for morphological operations.
    dilation_iterations : int
        Number of iterations for dilation.
    erosion_iterations : int
        Number of iterations for erosion.

    Returns
    -------
    segmented_image : numpy array
        Image with segmented text regions.
    """
    # Ensure binary image has only two unique values: 0 and 255
    if np.unique(binary_image).tolist() not in ([0, 255], [0], [255]):
        raise ValueError("The input image is not properly binarized.")
    
    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an output image to draw the detected components
    segmented_image = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):  # Skip the background label (0)
        x, y, w, h, area = stats[i]
        
        # Filter out components based on area thresholds
        if min_area_threshold < area < max_area_threshold:
            # Draw the bounding box around each component
            cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Fill the region with white
    
    # Apply morphological operations to refine the segmented image
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(segmented_image, kernel, iterations=dilation_iterations)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_iterations)
    
    return eroded_image

def sliding_window(image, window_size, step_size):
    """Slide a window across the image.

    Parameters
    ----------
    image : numpy array
        The input image.
    window_size : tuple
        The size of the window (width, height).
    step_size : int
        The step size for moving the window.

    Yields
    ------
    tuple
        The (x, y, window) where (x, y) is the top-left coordinate of the window,
        and window is the cropped region of the image.
    """
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def draw_regions(image, regions):
    """Draw the proposed regions on the image.

    Parameters
    ----------
    image : numpy array
        The input image.
    regions : list
        A list of tuples containing the coordinates of the regions (x, y, width, height).

    Returns
    -------
    numpy array
        The image with drawn regions.
    """
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 20)
    return image

def filter_regions(mask, regions, threshold=0.5):
    """Filter regions based on the amount of foreground pixels.

    Parameters
    ----------
    mask : numpy array
        The segmented image mask.
    regions : list
        A list of tuples containing the coordinates of the regions (x, y, width, height).
    threshold : float
        The minimum fraction of foreground pixels required to keep a region.

    Returns
    -------
    list
        The filtered list of regions.
    """
    filtered_regions = []
    for (x, y, w, h) in regions:
        window = mask[y:y + h, x:x + w]
        if np.mean(window) / 255 > threshold:
            filtered_regions.append((x, y, w, h))
    return filtered_regions

def merge_regions(regions, overlap_threshold=0.5):
    """Merge overlapping regions.

    Parameters
    ----------
    regions : list
        A list of tuples containing the coordinates of the regions (x, y, width, height).
    overlap_threshold : float
        The minimum overlap ratio to merge regions.

    Returns
    -------
    list
        The merged list of regions.
    """
    if not regions:
        return []

    # Sort regions by the x coordinate
    regions = sorted(regions, key=lambda r: r[0])

    merged_regions = []
    current_region = regions[0]

    for next_region in regions[1:]:
        x1, y1, w1, h1 = current_region
        x2, y2, w2, h2 = next_region

        # Calculate overlap
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        area1 = w1 * h1
        area2 = w2 * h2

        if overlap_area / area1 > overlap_threshold or overlap_area / area2 > overlap_threshold:
            # Merge regions
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y
            current_region = (new_x, new_y, new_w, new_h)
        else:
            merged_regions.append(current_region)
            current_region = next_region

    merged_regions.append(current_region)
    return merged_regions

# Example usage
image_path = 'hybrid/239746-0088.jpg'
gray_image = block_image(image_path)
binary_image = binarize_image(gray_image)
segmented_image = segment_image(binary_image, min_area_threshold=50, max_area_threshold=5000, kernel_size=(3, 3), dilation_iterations=1, erosion_iterations=1)

# Define window size and step size
window_size = (20, 20)  # Larger window size
step_size = 5  # Larger step size

# Generate region proposals
regions = []
for (x, y, window) in sliding_window(segmented_image, window_size, step_size):
    if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
        continue
    regions.append((x, y, window_size[0], window_size[1]))

# Filter regions based on content
filtered_regions = filter_regions(segmented_image, regions, threshold=0.1)  # Adjust threshold as needed

# Merge overlapping regions
merged_regions = merge_regions(filtered_regions, overlap_threshold=0.5)  # Adjust overlap threshold as needed

# Draw the proposed regions on the original image
image_with_regions = draw_regions(cv2.imread(image_path), merged_regions)

# Save the results
cv2.imwrite('hybrid/image_with_regions.png', image_with_regions)

# Display the image with regions
plt.imshow(cv2.cvtColor(image_with_regions, cv2.COLOR_BGR2RGB))
plt.title('Image with Region Proposals')
plt.show()