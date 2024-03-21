from PIL import Image
import numpy as np
from find_pixels import *


# Constants
prominance = 3000 # for finding crop boundaries


def rotate_block(block, degree=None):
    """Function to convert input image into grayscale and rotate it appropriately

    Parameters
    ----------
    block : Image
        a block of text from the image

    Returns
    -------
    data : numpy array
        rotated and greyscaled image as a numpy array       
    """    
    if degree is None:        
        degree = get_degree(block)
        block = block.rotate(degree, fillcolor=255)
    else:
        block = block.rotate(degree, fillcolor=0)    
    
    data = np.asarray(block)
    
    return degree, data


def get_degree(orig_image_gray):
    """Function to compute the degree needed to rotate the image

    Parameters
    ----------
    orig_image_gray : Image
        the original grayscaled, non-rotated image

    Returns
    -------
    minDeg : int
        the degree needed to rotate the image
    """
    dict = {}
    orig_image_gray = orig_image_gray.resize((orig_image_gray.size[0]//3, orig_image_gray.size[1]//3))
    for i in range(-10, 10):
        # image_file = orig_image_gray
        image_file = orig_image_gray.rotate(i, fillcolor=1)
        data = np.asarray(image_file)
        pixel_counts = np.sum(data, axis=1, keepdims=True)
        array = [] #TODO can convert to np.sum for performance boost
        for val in pixel_counts:  # flatten the numpy array
            array.append(data.shape[1] - val[0])
        crop_pixels = scipy.signal.find_peaks(array, prominence=prominance)[0]
        peaks = []
        for pixel in crop_pixels:
            peaks.append(array[pixel])
        peaks_dif = []
        for j in range(len(crop_pixels)-1):
            peaks_dif.append(crop_pixels[j+1]-crop_pixels[j])

        pixel_counts = np.sum(data, axis=1, keepdims=True)
        array = []
        for val in pixel_counts:  # flatten the numpy array
            array.append(val[0])
        crop_pixels = scipy.signal.find_peaks(array, prominence=prominance)[0]
        troughs = []
        for pixel in crop_pixels:
            troughs.append(data.shape[1] - array[pixel])
        troughs_dif = []
        for j in range(len(crop_pixels)-1):
            troughs_dif.append(crop_pixels[j + 1] - crop_pixels[j])
        if len(peaks_dif) <= 1 or len(troughs_dif) <= 1:
            #set to arbitrarily large values
            dif = 10000
            std_vals = 10000
        else:
            dif = (np.std(peaks_dif)+np.std(troughs_dif))
            std_vals = (np.std(peaks) + np.std(troughs))

        #formula
        dict[i] = (std_vals)*(dif**3) / (((np.mean(peaks) - np.mean(troughs))**3)*(len(peaks)**4)*(len(troughs)**4))

    minDeg = 0
    minVal = 99999
    for degree in dict:
        if abs(dict[degree]) < minVal:
            minVal = abs(dict[degree])
            minDeg = degree
    return minDeg



