import cv2
import numpy as np

def crop_folio(image_path):
    """
    Crops an image to focus on the document/folio region.
    
    Args:
        image_path (str): Path to the input image file.
        
    Returns:
        numpy.ndarray: Cropped image containing the document/folio region.
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance the document region
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour corresponds to the document/folio region
    folio_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle of the document/folio region
    x, y, w, h = cv2.boundingRect(folio_contour)
    
    # Crop the original image based on the bounding rectangle
    cropped = img[y:y+h, x:x+w]
    
    return cropped

cropped_image = crop_folio("images/239746-0134.jpg")
cv2.imwrite('cropped_image.jpg', cropped_image)