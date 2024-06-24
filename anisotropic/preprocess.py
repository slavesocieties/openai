import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Adaptive binarization
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    
    # Noise removal using connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    avg_area = np.mean(stats[1:, cv2.CC_STAT_AREA])
    filtered_image = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= avg_area:
            filtered_image[labels == i] = 255
    
    return filtered_image

def block_image(im_file):
    """Function to create a grayscale version of the image with enhanced contrast and normalized background variations

    Parameters
    ----------
    im_file : string
        a string representing the path to the original, full-color version of the image

    Returns
    -------
    im : numpy array
        a grayscaled, contrast-enhanced, and normalized version of the image
    """
    im = cv2.imread(im_file)     

    # full-color image is split into its three basic color channels
    # each channel is treated as a separate grayscale image representing the intensity of that color in the image
    rgb_planes = cv2.split(im)    
    result_norm_planes = []
    
    for plane in rgb_planes:
        # each color channel is dilated using a 7x7 kernel of ones
        # dilation increases the object area and accentuates features
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        # blurs the dilated image using a median filter with a 21x21 kernel to remove noise
        bg_img = cv2.medianBlur(dilated_img, 21)
        # the absolute difference between the original color channel and the blurred background image is calculated and subtracted from 255
        # to enhance the contrast of the foreground objects against the background
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        # the difference image is normalized to stretch its intensity values to span the full [0, 255] range
        # to further enhance the visibility of features
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)                
        result_norm_planes.append(norm_img)
    
    # the normalized images for each color channel are merged back together to form a single image.
    im = cv2.merge(result_norm_planes)

    # the merged image is converted to grayscale using OpenCV's cvtColor function with the COLOR_BGR2GRAY code
    # this conversion reduces the image to a single channel, making it a true grayscale image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
    
    return im

'''image_path = 'images/239746-0088.jpg'
preprocessed_image = preprocess_image(image_path)
im = Image.fromarray(preprocessed_image)
im.show()
blocked_image = block_image(image_path)
im = Image.fromarray(blocked_image)
im.show()'''