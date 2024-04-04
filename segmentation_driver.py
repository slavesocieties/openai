import os
from PIL import Image
import numpy as np
from pool_image import block_image
from layout import layout_analyze
from binarize_data import rotate_block
from find_pixels import find_pixels
from data_segmentation import data_segmentation

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

    # Step 1: Find the maximum width among images with height <= 3 * width
    max_width = max((img.width for img in blocks if img.height <= 3 * img.width), default=0)

    # Step 2: Filter out images with width at least 50 pixels larger or smaller than max_width
    entry_blocks = []
    entry_coords = []
    for i, img in enumerate(blocks):
        if max_width - 50 <= img.width <= max_width + 50:
            entry_blocks.append(img)
            entry_coords.append(coordinates[i])

    return entry_blocks, entry_coords

def segmentation_driver(path_to_image, save_directory="segmented", verbose=True, blocks_only=True):   
    pooled = block_image(path_to_image)    
    pooled_img = Image.fromarray(pooled)
    # TODO do we actually need to do this resizing? 
    pooled_img = pooled_img.resize((960, 1280))        
    pooled = np.array(pooled_img)
    
    if verbose:
        print("Image normalized.")       
    
    blocks, coordinates, angle = layout_analyze(pooled)    
    entry_blocks, entry_coords = filter_blocks(blocks, coordinates)    

    segments = []

    if verbose:
        print("Layout analyzed.")
        if entry_blocks == None:
            print("No entries found.")
            return segments

    if blocks_only:
        orig_img = Image.open(path_to_image)
        orig_img = orig_img.resize((960, 1280))
        orig_img = orig_img.rotate(angle, fillcolor=0)

    if "/" in path_to_image:
        path_to_image = path_to_image[path_to_image.rfind("/") + 1:]     

    for entry_id, block in enumerate(entry_blocks):        
        deg, block = rotate_block(block)                
        if blocks_only:            
            im_id = f"{path_to_image[:path_to_image.find('.')]}-{'0' * (2 - len(str(entry_id + 1)))}{entry_id + 1}"
            segments.append({"id": im_id, "coords": [entry_coords[entry_id][0], entry_coords[entry_id][1], entry_coords[entry_id][2], entry_coords[entry_id][3]]})            
                        
            orig_block = orig_img.crop(entry_coords[entry_id])            
            deg, orig_block = rotate_block(orig_block, degree=deg)
            orig_block = Image.fromarray(orig_block)                        
            orig_block.save(f"{save_directory}/{im_id}-color.jpg")                      
                       
            block = Image.fromarray(block)            
            block.save(f"{save_directory}/{im_id}-pooled.jpg")
            
            continue

        crop_pixels = find_pixels(block, 5000)        
        entry_segments = data_segmentation(block, crop_pixels, path_to_image, entry_id + 1, save_dir=save_directory) #cropping image and output        
        for segment in entry_segments:
            segments.append(segment)        

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

    if verbose and (not blocks_only):
        print(f"{len(segments)} segmented lines saved to {save_directory}.")
    elif verbose:
        print(f"{len(segments)} blocks of text saved to {save_directory}.")

    # TODO figure out where int64s are coming from
    if not blocks_only:
        for x, segment in enumerate(segments):
            for y in range(len(segment["coords"])):
                segments[x]["coords"][y] = int(segments[x]["coords"][y])            
    
    return segments

"""import json
with open("segmentation_test.json", "w") as f:
    json.dump(segmentation_driver("images/239746-0218.jpg"), f)"""