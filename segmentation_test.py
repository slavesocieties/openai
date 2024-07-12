import json
from PIL import Image
from segmentation_driver import *
from segmentation_performance import performance
import os



#reading folio
#get the text coordinates that human made 
# return #
# folio: coords for cropped image
# textList: a list of lists, representing the coords of human made bonding boxes 
def uncroppedHmnCoords(filename):
    with open("/Users/vip/openai/ml_segmentation/annotations.json") as f:
        data = json.load(f)

    textList = []

    for im in data["images"]:
        if im["filename"] == filename:

            for annotation in im["annotations"]:
                if annotation["name"] == "folio":
                    folio = annotation["polygon"]

                if annotation["name"] == "text":
                    textList.append(annotation["polygon"])

    return folio, textList


#Run the segmentation drivr to get the computer made bonding box
#Then, adjust the coords of computer made to match the human made one
def uncroppedCptCoords(filename):
    filepath = "/Users/vip/Desktop/SSDA/ssda_images/" + filename  # enter the images file path

    with open("segmentation_test.json", "w") as f:
        json.dump(segmentation_driver(filepath, display=True), f)
        
    with open("/Users/vip/openai/segmentation_test.json") as f:
        data = json.load(f)
    
    computerList = []

    #960 *1280 computer made image
    widthRatio = 2736/960
    height = 3648/1280

    for im in data:
        im["coords"][0]  = im["coords"][0] * widthRatio
        im["coords"][1]  = im["coords"][1] * height
        im["coords"][2]  = im["coords"][2] * widthRatio
        im["coords"][3]  = im["coords"][3] * height
        computerList.append(im["coords"])


    return computerList



# Crop the image and save it to a different folder
def cropImage (folio, filename):
    path = "/Users/vip/Desktop/SSDA/ssda_images/" + filename    # enter file path of original images
    im = Image.open(path) 

    newIm = im.crop((folio[0], folio[1], folio[2], folio[3]))

    # im1.show()
    destination = "/Users/vip/Desktop/SSDA/ssda_croppedIm/" + filename # path where to save the cropped images
    newIm.save(destination) 

    width, height = newIm.size

    return width, height



def croppedCptCoords(width, height, filename):
    filepath = "/Users/vip/Desktop/SSDA/ssda_croppedIm/" + filename # file path of cropped images

    with open("segmentation_test.json", "w") as f:
        json.dump(segmentation_driver(filepath, display=True), f)
        
    with open("/Users/vip/openai/segmentation_test.json") as f:
        data = json.load(f)
    
    computerList = []

    widthRatio = width/960
    height = height/1280

    for im in data:
        im["coords"][0]  = im["coords"][0] * widthRatio
        im["coords"][1]  = im["coords"][1] * height
        im["coords"][2]  = im["coords"][2] * widthRatio
        im["coords"][3]  = im["coords"][3] * height
        computerList.append(im["coords"])


    return computerList



def croppedHmnCoords (humanText, folio):
    croppedHmnText = []

    for text in humanText:
        newText = [0, 0, 0, 0]
        newText[0] = text[0] - folio[0]
        newText[1] = text[1] - folio[1]
        newText[2] = text[2] - folio[0]
        newText[3] = text[3] - folio[1]
        croppedHmnText.append(newText)

    return croppedHmnText


#Add the metrics for one image to the structure
def addData(data, filename, fscore, precision, recall, C_fscore, C_precision, C_recall):
    image_data = {
        "filename": filename
    }
    
    image_data["uncropped_performance"] = {
        "f-score": fscore,
        "precision": precision,
        "recall": recall
    }

    image_data["cropped_performance"] = {
        "f-score": C_fscore,
        "precision": C_precision,
        "recall": C_recall
    }
    
    # Add the image data to the main data structure
    data["images"].append(image_data)


def Compare(filename, data, threshold):

    folio, humanText = uncroppedHmnCoords(filename)
    computerText = uncroppedCptCoords(filename)

    # print(humanText, computerText)

    fscore, precision, recall = performance(humanText, computerText, threshold)
    
    #Then start to crop image and compare cropped image.
    
    width, height = cropImage(folio, filename)
    croppedCptText = croppedCptCoords(width, height, filename)
    croppedHmnText = croppedHmnCoords(humanText, folio)

    print("Width: ", width)
    print("Height: ", height)
    print(croppedCptText, croppedHmnText)

    C_fscore, C_precision, C_recall = performance(croppedHmnText, croppedCptText, threshold)

    addData(data, filename, fscore, precision, recall, C_fscore, C_precision, C_recall)

    print(filename, " analysis finished.")
    return data





def main():

    directory = "/Users/vip/Desktop/SSDA/ssda_images"  # file path of uncropped images

    # Get the list of image filenames
    nameList = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]


    data = {"images": []}

    threshold = float(input("Please enter a threshold value: "))

    try:
        index = float(threshold)
        print("You have entered the threshold value:", index)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

    
    # print(nameList)
    

    for filename in nameList:
        data = Compare(filename, data, threshold)

    # data = Compare("239746-0013.jpg", data, threshold)

    with open("segmentation_performance.json", 'w') as f:
        json.dump(data, f, indent=4)

main()