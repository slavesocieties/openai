import os
from PIL import Image
import numpy as np
from pool_image import block_image
from layout import layout_analyze
from binarize_data import rotate_block
from find_pixels import find_pixels
from data_segmentation import data_segmentation
import json

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

def segmentation_driver(path_to_image, save_directory="segmented", verbose=True):    
    pooled = block_image(path_to_image)
    # pooled is a numpy array representing a grayscaled and normalized version of the image
    pooled_img = Image.fromarray(pooled)
    pooled_img = pooled_img.resize((960, 1280))    
    pooled = np.array(pooled_img)
    # pooled is a resized version of the same array from above
    if verbose:
        print("Image normalized.")

    # TODO do we actually need to do this resizing?    
    
    blocks, coordinates, angle = layout_analyze(pooled)

    # TODO this may be redundant if we can effectively filter junk crops elsewhere
    entry_blocks, entry_coords = filter_blocks(blocks, coordinates)    

    entries = []
    lines = 0

    if verbose:
        print("Layout analyzed.")
        if entry_blocks == None:
            print("No entries found.")
            return entries    

    for entry_id, block in enumerate(entry_blocks):
        entries.append({"id": f"{path_to_image[:path_to_image.find('.')]}-{entry_id + 1}", "angle": angle, "coords": entry_coords[entry_id]})
        rotated = rotate_block(block)
        crop_pixels = find_pixels(rotated, 5000)        
        entry_segments = data_segmentation(rotated, crop_pixels, path_to_image, entry_id + 1, save_dir=save_directory) #cropping image and output
        if len(entry_segments) > 1:
            entries[entry_id]["lines"] = []        
        for segment in entry_segments:
            # TODO figure out where int64s are coming from
            for x in range(len(segment["coords"])):
                segment["coords"][x] = int(segment["coords"][x])
            entries[entry_id]["lines"].append(segment)
            lines += 1

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

    if verbose:
        print(f"{lines} segmented lines from {len(entries)} regions of text saved to {save_directory}.")             
    
    return entries

"""with open("segmentation_test.json", "w") as f:
    json.dump(segmentation_driver("15834-0093.jpg"), f)"""