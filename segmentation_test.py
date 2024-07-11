import json
from PIL import Image
from segmentation_driver import *
from segmentation_performance import performance
import os




# with open("segmentation_test.json", "w") as f:
#     json.dump(segmentation_driver("/Users/vip/Desktop/SSDA/ssda_images/239746-0013.jpg", display=True), f)





#reading folio
#get the text coordinates that human made 
def getHumanCoordinates(filename):
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

# tmp = getCoordinates("239746-0013.jpg")
# print(tmp)

def getCptCoordinates(filename):
    filepath = "/Users/vip/Desktop/SSDA/ssda_images/" + filename

    with open("segmentation_test.json", "w") as f:
        json.dump(segmentation_driver(filepath, display=True), f)
        
    with open("/Users/vip/openai/segmentation_test.json") as f:
        data = json.load(f)
    
    computerList = []

    for im in data:
        computerList.append(im["coords"])

    return computerList


#Add the metrics for one image to the structure
def addData(data, filename, fscore, precision, recall):
    image_data = {
        "filename": filename
    }
    
    image_data["uncropped_performance"] = {
        "f-score": fscore,
        "precision": precision,
        "recall": recall
    }
    
    # Add the image data to the main data structure
    data["images"].append(image_data)


def uncroppedCompare(filename, data, threshold):


    folio, humenText = getHumanCoordinates(filename)
    computerText = getCptCoordinates(filename)

    # print(humenText, computerText)

    fscore, precision, recall = performance(humenText, computerText, threshold)

    
    addData(data, filename, fscore, precision, recall)
    print(filename, " analysis finished.")
    return data





def main():

    directory = "/Users/vip/Desktop/SSDA/ssda_images"

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
        data = uncroppedCompare(filename, data, threshold)

    with open("segmentation_performance.json", 'w') as f:
        json.dump(data, f, indent=4)

main()