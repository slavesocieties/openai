import cv2
from pool_image import block_image

def detect_page_edges(image_path, display_width=800):
    # Load the image
    # image = cv2.imread(image_path)

    image = block_image(image_path)
    
    # Get the current dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the new dimensions while maintaining aspect ratio
    new_width = display_width
    new_height = int(height * (new_width / width))
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    gray = resized_image
    
    # Convert to grayscale
    # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blur, 100, 200)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours and find the largest one (assuming it's the page)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Draw the page contour on the resized image
    page_edges = resized_image.copy()
    cv2.drawContours(page_edges, [largest_contour], 0, (0, 255, 0), 2)
    
    # Display the resized image with page edges
    cv2.imshow("Page Edges", page_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_page_edges("images/585912-0092.jpg", display_width=1000)