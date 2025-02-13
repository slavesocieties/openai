
import os
from PIL import Image
import numpy as np
from pool_image import block_image
from layout import layout_analyze
from binarize_data import rotate_block
from find_pixels import find_pixels
from segment_lines import segment_lines
from pathlib import Path

def downscale_image(path_to_image):
    """reduces image size to 3MB or less by proportionally shrinking dimensions

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

def filter_blocks(blocks, coordinates):
    """Function to filter out noise blocks

    Parameters
    ----------
    blocks : list
        Images of regions of text segmented from an uncropped image
    coordinates : list
        pixel coordinates of these regions within uncropped image
    
    Returns
    -------
    entry_blocks : list
        regions of text likely to contain partial or whole sacramental records
    entry_coords : list
        pixel coordinates of these filtered text regions
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

def detect_internal_signature(image_array, num_consecutive_rows=100):
    """
    Find consecutive rows in an Image that meet the criteria that at least 90% of the pixels in the first half of the row
    and 90% in the last 10% of the row are white. Used to identify signature blocks in Images likely to contain multiple
    sacramental records but that layout analysis has failed to separate.
    
    Parameters:
    - image_array: a grayscale Image.
    - num_consecutive_rows: The number of consecutive rows that need to meet the criteria.

    Returns:
    - A list of tuples, each representing the start and end indices (inclusive) of consecutive rows meeting the criteria.
    """
    image_array = np.asarray(image_array)
    # image_array = np.array(image_array, copy=True)

    binarization_quantile = 0.1
    bin_thresh = np.quantile(image_array, binarization_quantile)
    image_array = np.where(image_array <= bin_thresh, 0, 1)

    rows, cols = image_array.shape
    # Calculate the thresholds for the number of white pixels
    first_half_threshold = 0.9 * (cols // 2)
    last_10_percent_threshold = 0.9 * (cols // 10)
    
    # Array to keep track of rows meeting the criteria
    criteria_met = np.zeros(rows, dtype=bool)
    
    # Check criteria for each row
    for i in range(rows):
        if (np.sum(image_array[i, :(cols // 2)]) >= first_half_threshold) and \
           (np.sum(image_array[i, -int(np.ceil(cols * 0.1)):]) >= last_10_percent_threshold):
            criteria_met[i] = True   
            
    # Identify and collect sequences of consecutive rows meeting the criteria
    consecutive_sequences = []
    start_index = None
    
    for i in range(rows):
        if criteria_met[i] and start_index is None:            
            start_index = i            
        elif (not criteria_met[i]) and (start_index is not None):            
            if i - start_index >= num_consecutive_rows:
                consecutive_sequences.append((start_index, i - 1))
                start_index = None
            else:
                start_index = None        
            
    # Check if the last sequence meets the criteria
    if start_index is not None and rows - start_index >= num_consecutive_rows:
        consecutive_sequences.append((start_index, rows - 1))
        
    return consecutive_sequences

def segmentation_driver(path_to_image, save_cropped_images=True, save_directory="segmented", verbose=True, blocks_only=True, display=False, orig_coords=False):
    """
    Main function for image segmentation. Segments uncropped image into blocks and/or lines of text for manual or automated transcription.

    Parameters
    ----------
    path_to_image : string
        path to image to be segmented
    save_directory : string
        directory to which segmented blocks and/or lines should be saved
    verbose : Boolean
        True if function call should produce explanatory output, false otherwise
    blocks_only : Boolean
        True if function call should only segment blocks, false for blocks and lines
    
    Returns
    -------
    segments : list
        dictionaries containing ids and coordinates for segmented blocks or lines
    """   
    pooled = block_image(path_to_image)    
    pooled_img = Image.fromarray(pooled)
    if display:
        pooled_img.show()
    # TODO do we actually need to do this resizing? 
    pooled_img = pooled_img.resize((960, 1280))        
    pooled = np.array(pooled_img)
    
    if verbose:
        print("Image normalized.")       
    
    blocks, coordinates, angle = layout_analyze(pooled, display=display)    
    entry_blocks, entry_coords = filter_blocks(blocks, coordinates)

    Path(save_directory).mkdir(exist_ok=True)    

    """for i, block in enumerate(entry_blocks):
        #deg, block = rotate_block(block)
        #block = Image.fromarray(block)
        block.show()        
        signatures = detect_internal_signature(block, num_consecutive_rows=50)        

        if len(signatures) == 0:
            final_blocks.append(block)
            final_coords.append(entry_coords[i])
        else:
            sig = 0
            top = 0
            while sig < len(signatures):                
                final_blocks.append(block.crop((0, top, block.width, signatures[sig][0])))
                final_coords.append((entry_coords[i][0], entry_coords[i][1] + top, entry_coords[i][2], entry_coords[i][1] + signatures[sig][0]))
                top += signatures[sig][1]                
                sig += 1                
            final_blocks.append(block.crop((0, top, block.width, block.height)))
            final_coords.append((entry_coords[i][0], entry_coords[i][1] + top, entry_coords[i][2], entry_coords[i][3])) """       

    im_record = {}

    if verbose:
        print("Layout analyzed.")
        if entry_blocks == None:
            print("No entries found.")
            return im_record

    if blocks_only:
        orig_img = Image.open(path_to_image)
        orig_img = orig_img.resize((960, 1280))
        orig_img = orig_img.rotate(angle, fillcolor=0)

    # Used to calculate the width and height ratios used to downscale the image to 960x1280. 
    # This is used when we want to evaluate the performance of the segmentation
    width, height = Image.open(path_to_image).size
    width_ratio = width / 960
    height_ratio = height / 1280

    path_to_image = path_to_image[path_to_image.rfind("/") + 1:]   
     

    for entry_id, block in enumerate(entry_blocks):                        
        deg, block = rotate_block(block)
        segment_id = f"{'0' * (2 - len(str(entry_id + 1)))}{entry_id + 1}"
        if not "image_id" in im_record:
            im_id = path_to_image[:path_to_image.find('.')]
            im_record["image_id"] = im_id
            im_record["text"] = []
            if orig_coords:
                im_record["original texts"] = []                            
        if blocks_only:           
            im_record["text"].append({"segment_id": segment_id, "coords": [int(entry_coords[entry_id][0]), int(entry_coords[entry_id][1]), int(entry_coords[entry_id][2]), int(entry_coords[entry_id][3])]})            
                        
            orig_block = orig_img.crop(entry_coords[entry_id])
            
            if orig_coords:
                im_record["original texts"].append({"segment_id": segment_id, "coords": [int(entry_coords[entry_id][0] * height_ratio), int(entry_coords[entry_id][1] * width_ratio), int(entry_coords[entry_id][2] * height_ratio), int(entry_coords[entry_id][3] * width_ratio)]})

            if display:
                orig_block.show()
            
            if save_cropped_images:            
                deg, orig_block = rotate_block(orig_block, degree=deg)
                orig_block = Image.fromarray(orig_block)                                   
                orig_block.save(f"{save_directory}/{im_id}-{segment_id}-color.jpg")                      
                block = Image.fromarray(block)                        
                block.save(f"{save_directory}/{im_id}-{segment_id}-pooled.jpg")           
        
        # TODO update for line-based segmentation
        """crop_pixels = find_pixels(block, 5000)        
        entry_segments = segment_lines(block, crop_pixels, path_to_image, entry_id + 1, save_dir=save_directory) #cropping image and output        
        for segment in entry_segments:
            segments.append(segment)"""        

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

    """if verbose and (not blocks_only):
        print(f"{len(segments)} segmented lines saved to {save_directory}.")
    elif verbose:
        print(f"{len(segments)} blocks of text saved to {save_directory}.")"""
    
    if verbose and blocks_only and save_cropped_images:
        print(f"{len(im_record['text'])} blocks of text saved to {save_directory}.")        
    
    #print(im_record)
    return im_record

"""import json
with open("segmentation_test.json", "w") as f:
    json.dump(segmentation_driver("/Users/kevinchen/Desktop/SSDA/full size/239746-0021.jpg", display=True), f)"""
