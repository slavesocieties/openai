import os
import json
import numpy as np
from PIL import Image, ImageDraw

def resize_image_and_mask(image, mask, new_size):
    image_resized = image.resize(new_size, Image.ANTIALIAS)
    mask_resized = mask.resize(new_size, Image.NEAREST)
    return image_resized, mask_resized

def create_mask(image_size, annotations):
    mask = Image.new('L', (image_size['width'], image_size['height']), 0)
    draw = ImageDraw.Draw(mask)
    for annotation in annotations:
        if annotation['name'] == 'folio':
            polygon = annotation['polygon']
            # Convert bounding box to list of points if necessary
            if len(polygon) == 4:
                left, top, right, bottom = polygon
                polygon_points = [(left, top), (right, top), (right, bottom), (left, bottom)]                
                draw.polygon(polygon_points, outline=255, fill=255)                                
            else:
                draw.polygon(polygon, outline=1, fill=1)
    return mask

def process_annotations(annotations_file, new_size):    
    with open(annotations_file, 'r') as file:
        annotations = json.load(file)

    # resized_data = []
    for image_info in annotations['images']:
        image_filename = image_info['filename']
        image_size = image_info['size']
        image = Image.open(f"images/original/{image_filename}")
        mask = create_mask(image_size, image_info['annotations'])
        
        # Resize image and mask
        image_resized, mask_resized = resize_image_and_mask(image, mask, new_size)
        
        # Save resized image and mask
        image_resized.save(f"images/resized/{image_filename}")
        mask_resized.save(f"masks/{image_filename.replace('.jpg', '.png')}")
        
        '''# Update annotations
        new_annotations = []
        scale_x = new_size[0] / image_size['width']
        scale_y = new_size[1] / image_size['height']
        for annotation in image_info['annotations']:
            if annotation['name'] == 'folio':
                new_polygon = [(int(x * scale_x), int(y * scale_y)) for x, y in annotation['polygon']]
                new_annotations.append({
                    "name": "folio",
                    "polygon": new_polygon
                })
        
        resized_data.append({
            "filename": f"resized_{image_filename}",
            "size": {
                "width": new_size[0],
                "height": new_size[1]
            },
            "annotations": new_annotations
        })

    return resized_data'''

# Example usage
annotations_file = 'annotations.json'
new_size = (192, 256)  # Resize to 1024x768 while preserving aspect ratio

process_annotations(annotations_file, new_size)

'''# Save updated annotations
with open('resized_annotations.json', 'w') as outfile:
    json.dump({"images": resized_data}, outfile, indent=4)

print("Images and masks resized successfully.")'''