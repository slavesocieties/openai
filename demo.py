from find_pixels import find_pixels
from segment_lines import segment_lines
from binarize_data import rotate_block
from segmentation_driver import segmentation_driver, detect_internal_signature
from PIL import Image
import json

# segment lines from an entry

# open image with PIL
img = Image.open('images/239746-0044-01.jpg').convert('L')

# rotate image for straightest possible lines
degree, img = rotate_block(img)

# identify y coordinates of line stop/start
crop_pixels = find_pixels(img, 5000)

# save the resulting segmented lines
entry_segments = segment_lines(img, crop_pixels, 'images/239746-0044-01.jpg', 1, save_dir='demo')

# identify coordinates of individual entries in an image
# with open("segmentation_test.json", "w") as f:
    # json.dump(segmentation_driver("images/239746-0088.jpg", display=True), f)
