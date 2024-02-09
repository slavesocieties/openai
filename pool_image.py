import numpy as np
import cv2
from PIL import Image

def block_image(im_file):
    """Function to binarize the given image

    Parameters
    ----------
    im_file : string
        a string representing the path to the image

    Returns
    -------
    im : numpy array
        the binarized image
    """
    im = cv2.imread(im_file,1)     

    rgb_planes = cv2.split(im)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    im = cv2.merge(result_norm_planes)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    """pooled_img = Image.fromarray(im)
    pooled_img = pooled_img.resize((960, 1280))
    pooled_img.show()"""
    
    return im