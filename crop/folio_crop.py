import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)    
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    display_image(blurred, title="Image")
    return blurred

def detect_edges(image):
    edges = cv2.Canny(image, 50, 150)    
    return edges

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    return contours

def filter_contours(contours, image_shape):
    page_contour = None
    max_area = 0
    image_area = image_shape[0] * image_shape[1]    
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0.2 * image_area and area > max_area:  # Threshold to remove small contours
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)            
            if len(approx) == 4:  # Looking for a quadrilateral
                page_contour = approx
                max_area = area   
                
    return page_contour

def crop_image(image, contour):
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cropped = image[y:y+h, x:x+w]
        return cropped
    return image

def display_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def process_image(image_path):
    image = load_image(image_path)
    preprocessed = preprocess_image(image)
    edges = detect_edges(preprocessed)
    contours = find_contours(edges)
    page_contour = filter_contours(contours, image.shape)
    cropped_image = crop_image(image, page_contour)
    display_image(cropped_image, title="Cropped Image")

# Example usage
image_path = "images/760007-0013.jpg"
process_image(image_path)
