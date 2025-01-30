import os
import json
from segmentation_driver import segmentation_driver

#Opens a folder of images and processes them all in the segmentation driver and puts them all into one json
def image_walk(path_to_json, path_to_folder):
    with open(path_to_json, "w") as f:
        for root, folders, files in os.walk(path_to_folder):
            results = []
            for file in files:
                json_obj = segmentation_driver(os.path.join(root, file), save_cropped_images=False, display=False, orig_coords=True)
                results.append(json_obj)
            json.dump(results, f, indent = 4)
