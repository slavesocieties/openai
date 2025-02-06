import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_largest_light_region(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Visualize grayscale image
    plt.figure()
    plt.title("Grayscale Image")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Normalize brightness and increase contrast
    normalized = cv2.normalize(gray, None, 50, 255, cv2.NORM_MINMAX)
    # Visualize normalized image
    plt.figure()
    plt.title("Normalized Image")
    plt.imshow(normalized, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Option 1: Use adaptive thresholding
    binary = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Option 2: Use fixed threshold with an adjusted value
    # _, binary = cv2.threshold(normalized, 100, 255, cv2.THRESH_BINARY)

    # Option 3: Threshold without normalization (directly from grayscale)
    # _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Visualize binary image
    plt.figure()
    plt.title("Binary Image")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Invert binary image to make light regions white (255) and dark regions black (0)
    inverted_binary = cv2.bitwise_not(binary)
    # Visualize inverted binary image
    plt.figure()
    plt.title("Inverted Binary Image")
    plt.imshow(inverted_binary, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Apply erosion and dilation to clean up small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_binary = cv2.erode(inverted_binary, kernel, iterations=1)
    cleaned_binary = cv2.dilate(cleaned_binary, kernel, iterations=1)
    # Visualize cleaned binary image
    plt.figure()
    plt.title("Cleaned Binary Image (After Erosion & Dilation)")
    plt.imshow(cleaned_binary, cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Find connected components (to identify distinct regions)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_binary, connectivity=8)
    
    # Visualize connected components
    label_viz = (labels * (255 / (num_labels - 1))).astype("uint8")  # Scale labels for visualization
    plt.figure()
    plt.title("Connected Components")
    plt.imshow(label_viz, cmap="jet")
    plt.axis("off")
    plt.show()
    
    # Filter by area and get the largest region
    min_area = 10000  # Minimum area to consider a valid folio
    largest_area = 0
    largest_label = 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area and area > min_area:
            largest_area = area
            largest_label = label
    
    if largest_area == 0:
        raise ValueError("No suitable folio region found!")
    
    # Get the bounding box of the largest light region
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    
    # Add padding
    padding = 20
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, gray.shape[1] - x)
    h = min(h + 2 * padding, gray.shape[0] - y)
    
    # Crop the largest region
    cropped = image[y:y+h, x:x+w]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped)
    print(f"Cropped image saved to {output_path}")

    # Display original and cropped images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Cropped Image")
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.show()

crop_largest_light_region("images/760017-0008.jpg", "images/7600017-0008_cropped.jpg")