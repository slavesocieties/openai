import numpy as np
import scipy.signal


def find_pixels(data, prominance):
    """Find pixel boundaries to crop the images into lines using pixel-histogram analysis

    Parameters
    ----------
    data : numpy array
        rotated image as a numpy array
    prominance : float
        the prominance parameter to define the find_peaks method

    Returns
    -------
    crop_pixels : list
        list of indices indicate where to crop the image
    """
    pixel_counts = np.sum(data, axis=1, keepdims=True) #sum pixels along the horizontal axis
    array = []
    for val in pixel_counts: #flatten the numpy array
        array.append(val[0])
    crop_pixels = scipy.signal.find_peaks(array, prominence=prominance)[0] #find pixel boundaries
    return crop_pixels


