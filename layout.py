import numpy as np
from PIL import Image, ImageDraw
import sys
sys.setrecursionlimit(1000000000)

def get_pixel_histogram(data, compress_height, compress_width):
    """Function to compute a 1D pixel histogram where each value is the count of pixels across the vertical axis
    Note that this function will compute the histogram based on a compressed version of data

    Parameters
    ----------
    data : numpy array
        the image from which we will compute the pixel histogram
    compress_height : int
        the amount of pixels to be compressed into 1 pixel in the vertical axis
    compress_width : int
        the amount of pixels to be compressed into 1 pixel in the horizontal axis

    Returns
    -------
    series : list
        the 1D pixel histogram
    """
    h = len(data)
    w = len(data[0])
    img_tmp = []
    for i in range(0, h, compress_height):
        img_tmp.append([])
        for j in range(w):
            pixels = 0
            for k in range(compress_height):
                if i+k < len(data) and data[i+k][j] == 0:
                    pixels += 1
            # print(pixels)
            if pixels > compress_height/10:
                img_tmp[-1].append(0)
            else:
                img_tmp[-1].append(255)

    img_tmp2 = []
    for i in range(0, len(img_tmp[0]), compress_width):
        img_tmp2.append([])
        for j in range(len(img_tmp)):
            pixels = 0
            for k in range(compress_width):
                if i+k < len(img_tmp[j]) and img_tmp[j][i+k] == 0:
                    pixels += 1
            # print(pixels)
            if pixels > compress_width/10:
                img_tmp2[-1].append(0)
            else:
                img_tmp2[-1].append(255)
    array = np.array(img_tmp2).T
    series = []
    for i in range(len(array[0])):
        cur = 0
        for j in range(len(array)):
            if array[j][i] == 255:
                cur += 1
        series.append(cur)
    return series

def find_rotate_angle(data, compress_height, compress_width):
    """Function to compute the angle of rotation needed to correct the image

    Parameters
    ----------
    data : numpy array
        the image from which we will compute the angle of rotation
    compress_height : int
        the amount of pixels to be compressed into 1 pixel in the vertical axis
    compress_width : int
        the amount of pixels to be compressed into 1 pixel in the horizontal axis

    Returns
    -------
    maxAngle : int
        the angle of rotation needed to correct the image
    """
    img = Image.fromarray(data)
    maxStd = 0
    maxAngle = 0
    for i in range(-10, 10):
        tmp_img = img.rotate(i, fillcolor=1)
        series = get_pixel_histogram(np.asarray(tmp_img), compress_height, compress_width)
        std = np.std(series)
        if std > maxStd:
            maxStd = std
            maxAngle = i
    return maxAngle

# TODO explore approach that allows us to intelligently identify proper multiplier for a set of images
def compute_spaces(data, compress_height, compress_width, multiplier=1.2):
    """Function to compute the vertical spaces (that runs from top to bottom) between layout bounding boxes
    Note that this algorithm will compress the target image before performing analysis to reduce pixel noise

    Parameters
    ----------
    data : numpy array
        the image from which we will compute the spaces
    compress_height : int
        the amount of pixels to be compressed into 1 pixel in the vertical axis
    compress_width : int
        the amount of pixels to be compressed into 1 pixel in the horizontal axis
    multiplier : float
        a hyper-parameter to define the threshold for pixel histogram analysis (higher = less spaces)

    Returns
    -------
    spaces : list
        a list of computed spaces
    """
    series = get_pixel_histogram(data, compress_height, compress_width)
    threshold = np.mean(series)+multiplier*np.std(series)
    spaces = []
    left = 0
    is_higher = False
    for i in range(len(series)):
        if series[i] > threshold and not is_higher:
            is_higher = True
            if left != i:
                spaces.append([left*compress_width, i*compress_width])
        elif series[i] < threshold and is_higher:
            is_higher = False
            left = i
    return spaces

def layout_analyze(data, save_path = None, display = False):
    """Function to compute the layout bounding blocks that contain texts within an image

    Parameters
    ----------
    data : numpy array
        a grayscale image that has been normalized from a color original    
    save_path : string (optional)
        the path to which we want to save the layout-analyzed image (With bounding box overlays)
        
    Returns
    -------
    crops_orig_vertical : list(Image)
        a list of images of text blocks    
    coordinates : list
        a list of coordinates of the text blocks within the original image in (left, top, right, bottom) format
    """    
    # data is the pooled image, resized to 960x1280    
    binarization_quantile = 0.1
    bin_thresh = np.quantile(data, binarization_quantile)
    #print("Dynamic binarization threshold = "+str(bin_thresh))
    orig_img = data.copy()
    orig_img = Image.fromarray(orig_img)

    data = np.where(data <= bin_thresh, 0, 255)

    # data is still grayscale but has been pseudo-binarized (each pixel is either 0 or 255)
    # TODO should we actually binarize here?    
        
    img = Image.fromarray(data)   
    angle = find_rotate_angle(data, 20, 20)    
    img = img.rotate(angle, fillcolor=255)    
    # fill color has been changed here, I think that the original version assumed the image was binarized
    orig_img = orig_img.rotate(angle, fillcolor=255)
    # orig_img.save("pooled.jpg")    
    data = np.asarray(img)
    # data is rotated pseudo-binarized image
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img, "RGBA")
    # img has been converted back to color (so we can draw boxes on it?)

    spaces_horizontal = compute_spaces(data, 20, 20) # hyperparams
    crops_horizontal = []
    crops_orig_horizontal = []
    indices_horizontal = []

    for space in spaces_horizontal:
        crops_horizontal.append(img.crop((space[0], 0, space[1], img.size[1])))
        indices_horizontal.append(space)
        crops_orig_horizontal.append(orig_img.crop((space[0], 0, space[1], img.size[1])))
    indices_horizontal.append(img.size[1])
    
    crops_vertical = []
    crops_orig_vertical = []
    coordinates = []
    for i,crop in enumerate(crops_horizontal):
        crop_data = np.asarray(crop).T[0]
        spaces_vertical = compute_spaces(crop_data, 20, 20) # hyperparams
        for space in spaces_vertical:
            # TODO we're still getting junk crops (ruler, parts of images were edges of a lot of pages appear, etc)
            if space[1]-space[0] > 50 and crop.size[0] > 50: # filter out garbage crops
                draw.rectangle((indices_horizontal[i][0], space[0], indices_horizontal[i][1], space[1]), outline= 'blue', fill=(0, 255, 0, 30))
                crops_vertical.append(crop.crop((0, space[0], crop.size[0], space[1])))
                # crops_orig_vertical.append(crops_orig_horizontal[i].crop((0, space[0], crop.size[0], space[1])))
                crops_orig_vertical.append(orig_img.crop((indices_horizontal[i][0], space[0], indices_horizontal[i][1], space[1])))
                coordinates.append((indices_horizontal[i][0], space[0], indices_horizontal[i][1], space[1]))

    if display:
        img.show()
    
    if save_path is not None:
        img.save(save_path)

    """print(f"{len(coordinates)} text regions found.")

    for x in crops_orig_vertical:
        x.show()"""
    
    return crops_orig_vertical, coordinates, angle

# print(detect_internal_signature("segmented/239746-0175-01-pooled.jpg"))