import json
from PIL import Image
from segmentation_driver import *
from segmentation_performance import performance
import os

# reading a json file 
# with open("ml_segmentation/annotations.json") as f:
#     data = json.load(f)

# # print(data["images"])
# # print("hello world")

# for im in data["images"]:
#     print(im["filename"])

# with open("/Users/vip/openai/ml_segmentation/annotations.json") as f:
#     data = json.load(f)

# directory = "/Users/vip/Desktop/SSDA/ssda_images"
# nameList = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]


# checkName = []

# for filename in nameList:
#     check = False
#     for im in data["images"]:
#         if im["filename"] == filename:
#             check = True
        
#     if (check == False):
#         checkName.append(filename)

# print(checkName)



# Opens an image in RGB mode
im = Image.open(r"/Users/vip/Desktop/SSDA/ssda_images/239746-0013.jpg")  # Replace with your image path

# Size of the image in pixels (size of original image)
width, height = im.size

# Setting the points for the cropped image
left = 413
top = 88
right = 2556
bottom = 3528

# Cropped image of above dimension
im1 = im.crop((left, top, right, bottom))

# Shows the image in image viewer (optional)
im1.show()

# Save the cropped image to a file
destination = "/Users/vip/Desktop/SSDA/ssda_croppedIm/" + "239746-0013.jpg"
im1.save(destination) 

