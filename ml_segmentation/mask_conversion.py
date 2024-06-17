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

# write transformer to convert annotation json to 2D class mask