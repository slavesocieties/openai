import os
from PIL import Image, ImageDraw
import numpy as np
from pool_image import block_image
from layout import layout_analyze
from binarize_data import gray_and_rotate
from find_pixels import find_pixels
from data_segmentation import data_segmentation
#import requests

def preprocess(path_to_image):
    """Function to preprocess the image

    Parameters
    ----------
    path_to_image : string
        a string representing the path to the image
    """
    while os.stat(path_to_image).st_size > 3000000:
        im = Image.open(path_to_image)
        width, height = im.size
        im = im.resize((int(round(width * .75)), int(round(height * .75))))
        im.save(path_to_image)
        im.close()

def filter_blocks(blocks, coordinates, thresh = .5):
    """Function to filter out noise blocks

    Parameters
    ----------
    blocks : list
        a list of blocks produced from the layout analyzer
    coordinates : list
        a list of coordinates of the blocks
    
    Returns
    -------
    entry_blocks : list
        a list of blocks that have noise blocks filtered
    entry_coords : list
        a list of coordinates of blocks that have noise blocks filtered
    """
    block_areas = []
    total_area = 0
    for block in blocks:
        block_areas.append(block.width * block.height)              
    for area in block_areas:
        total_area += area
    if len(block_areas) > 0:   
        avg_area = total_area / len(block_areas)
    else:
        return None, None
    entry_blocks = []
    entry_coords = []   
    for index, block in enumerate(blocks):        
        if block.width * block.height > thresh * avg_area:
            entry_blocks.append(block)
            entry_coords.append(coordinates[index])
    return entry_blocks, entry_coords

def segmentation_driver(path_to_image):   
    #preprocess(path_to_image)

    orig_img = Image.open(path_to_image)
    orig_img = orig_img.resize((960, 1280))
    #orig_img.show()
    pooled = block_image(path_to_image)
    pooled_img = Image.fromarray(pooled)
    pooled_img = pooled_img.resize((960, 1280))
    #pooled_img.show()
    pooled = np.array(pooled_img)    
    
    blocks, rotated_img, coordinates = layout_analyze(pooled, orig_img, save_path="layout.jpg")
    entry_blocks, entry_coords = filter_blocks(blocks, coordinates)

    if entry_blocks == None:
        return []
    
    all_coords = []
    counts = []

    tmp = rotated_img.convert('RGB')
    draw = ImageDraw.Draw(tmp, "RGBA")

    for entry_id, block in enumerate(entry_blocks):
        data, image_file, orig_image = gray_and_rotate(block)

        crop_pixels = find_pixels(data, 5000)
        data = np.array(orig_image)
        count, segment_coords = data_segmentation(data, crop_pixels, path_to_image, image_file, entry_id + 1, entry_coords[entry_id]) #cropping image and output
        for coord in segment_coords:
            draw.rectangle((coord[0], coord[1], coord[2], coord[3]), outline= 'blue', fill=(0, 255, 0, 30))
        all_coords.append(segment_coords)
        counts.append(count)

    """count = 0
    for file in os.scandir(f'./segmented/{path_to_image}'):
        with open(file.path, "rb") as f:
            img_data = f.read()
            headers = {"Content-Type":"image/jpeg"}
            requests.put("https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-htr-training/" + file.name , data=img_data, headers=headers)
            count += 1
        os.remove(file.path)    
    os.rmdir(f'./segmented/{path_to_image}')
    print("Done segmentation and upload")"""
    
    return counts

#print(segmentation_driver("239746-0037.jpg"))