import numpy as np
from PIL import Image, ImageDraw
import json

def create_training_mask(image_data, target_size=None):
    # Get image size          
    width = image_data['size']['width']
    height = image_data['size']['height']            
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create separate masks for "folio" and "text"
    folio_mask = Image.new('L', (width, height), 0)
    text_mask = Image.new('L', (width, height), 0)
    draw_folio = ImageDraw.Draw(folio_mask)
    draw_text = ImageDraw.Draw(text_mask)
    
    # Draw polygons on masks
    for annotation in image_data['annotations']:
        polygon = annotation['polygon']
        polygon = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        if annotation['name'] == 'folio':
            draw_folio.polygon(polygon, outline=1, fill=1)
        elif annotation['name'] == 'text':
            draw_text.polygon(polygon, outline=1, fill=1)
    
    # Convert masks to numpy arrays
    folio_mask = np.array(folio_mask)
    text_mask = np.array(text_mask)
    
    # Create final mask
    mask[(folio_mask == 1) & (text_mask == 1)] = 2
    mask[(folio_mask == 1) & (text_mask == 0)] = 1
    
    # Resize the mask if target size is provided
    if target_size:
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.resize(target_size, Image.NEAREST)
        mask = np.array(mask_image)
    
    return mask

def create_masks(path_to_annotations, target_size=(192, 256)):
    with open(path_to_annotations, 'r', encoding='utf-8') as f:
        data = json.load(f)

    masks = []

    for image in data['images']:
        masks.append(create_training_mask(image, target_size))