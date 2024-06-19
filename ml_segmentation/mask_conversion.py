import numpy as np

def mask_to_onehot(mask, num_classes):
    """
    Converts a grayscale segmentation mask to a one-hot encoded mask.
    
    Args:
        mask (numpy.ndarray): The grayscale segmentation mask, where pixel values represent class labels.
        num_classes (int): The number of classes in the segmentation task.
        
    Returns:
        numpy.ndarray: The one-hot encoded segmentation mask.
    """
    one_hot_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.uint8)
    for c in range(num_classes):
        one_hot_mask[:, :, c] = (mask == c).astype(np.uint8)
    return one_hot_mask

from PIL import Image

def load_and_binarize_mask(image_path):
    # Load the image
    mask_image = Image.open(image_path).convert('L')
    
    # Convert the image to a NumPy array
    mask_array = np.array(mask_image)
    
    # Scale non-zero values to 1
    binarized_mask = (mask_array > 0).astype(np.uint8)
    
    return binarized_mask

'''# Example usage
mask_image_path = '/mnt/data/masks/debug_mask_239746-0013.png'
binarized_mask = load_and_binarize_mask(mask_image_path)

print(binarized_mask)'''
