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
    widths = {}
    for block in blocks:
        # TODO if width equals width of largest region that isn't narrow, keep it, if not, don't
        block_areas.append(block.width * block.height)        
        if str(block.width) not in widths:
            widths[str(block.width)] = 1
        else:
            widths[str(block.width)] += 1              
    for area in block_areas:
        total_area += area
    if len(block_areas) > 0:   
        avg_area = total_area / len(block_areas)
    else:
        return None, None
    most_freq_width_count = 0
    for width in widths:
        if widths[width] > most_freq_width_count:
            most_freq_width_count = widths[width]
    most_freq_width = []
    for width in widths:
        if widths[width] == most_freq_width_count:
            most_freq_width.append(width)
    most_freq_width.sort(reverse=True)
    most_freq_width = most_freq_width[0]
    print(most_freq_width)

    entry_blocks = []
    entry_coords = []   
    for index, block in enumerate(blocks):
        print(block.width)        
        if (block.width * block.height > thresh * avg_area) and ((block.width * block.height > 75000) or (block.width == most_freq_width)):
            entry_blocks.append(block)
            entry_coords.append(coordinates[index])
    return entry_blocks, entry_coords

def segmentation_driver(path_to_image, save_directory="segmented", verbose=True, blocks_only=True):   
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

    for entry_id, block in enumerate(entry_blocks):        
        deg, block = rotate_block(block)                
        if blocks_only:
            # TODO allow this to handle nested directories
            im_id = f"{path_to_image[:path_to_image.find('.')]}-{entry_id + 1}"
            segments.append({"id": im_id, "coords": [entry_coords[entry_id][0], entry_coords[entry_id][1], entry_coords[entry_id][2], entry_coords[entry_id][3]]})            
                        
            orig_block = orig_img.crop(entry_coords[entry_id])            
            deg, orig_block = rotate_block(orig_block, degree=deg)
            orig_block = Image.fromarray(orig_block)                        
            orig_block.save(f"{save_directory}/{im_id}-color.jpg")

            #block = block_image(f"{save_directory}/{im_id}-color.jpg")            
                       
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

with open("segmentation_test.json", "w") as f:
    json.dump(segmentation_driver("239746-0027.jpg"), f)