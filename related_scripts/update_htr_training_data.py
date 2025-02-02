#check list of objects for lines that have both image and text on s3 but do not appear in training data record and add their id/color scheme/text

import boto3
import json
from utility import check_binarized
from upload_htr_training_data import update_training_data
import os
from PIL import Image

def download_data(client, bucket="ssda-htr-training", key="ssda-htr-training-data.json"):
    try:
        client.download_file(bucket, key, key)
        with open(key, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def list_all_objects(s3_client, bucket_name):
    # Create an S3 client
    #s3_client = boto3.client('s3')
    
    # Initialize an empty list to hold all object names
    all_objects = []
    
    # Initialize the pagination marker
    continuation_token = None
    
    while True:
        # If it's the first call, don't use the ContinuationToken
        if continuation_token:
            response = s3_client.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        # Add the current batch of object names to the list
        if 'Contents' in response:
            for obj in response['Contents']:
                all_objects.append(obj['Key'])
        
        # Check if more objects are available
        if response['IsTruncated']:
            continuation_token = response['NextContinuationToken']
        else:
            break  # Exit the loop if no more objects are available
    
    return all_objects

def add_pixel_count(bucket="ssda-htr-training"):
    s3_client = boto3.client('s3')
    lines = download_data(s3_client)
    n = 0
    for index, line in enumerate(lines["images"]):
        if "pixels" not in line:
            s3_client.download_file(bucket, f"{line['id']}.jpg", "temp.jpg")
            with Image.open("temp.jpg") as im:        
                lines["images"][index]["pixels"] = im.size[0] * im.size[1]
            n += 1
            if n % 500 == 0:
                print(f"Pixel count added for {n} lines.")

    with open("ssda-htr-training-data.json", "w") as f:
        json.dump(lines, f)

    s3_client.upload_file("ssda-htr-training-data.json", bucket, "ssda-htr-training-data.json", ExtraArgs={'ContentType': 'application/json'})
    os.unlink("ssda-htr-training-data.json")
    os.unlink("temp.jpg")

    return n

print(f"Pixel count added for {add_pixel_count()} images.")

def driver(bucket = "ssda-htr-training"):
    s3_client = boto3.client('s3')

    training_data = download_data(s3_client)

    if training_data is None:
        print("Failed to load training data.")
        return
    
    current_ids = []

    for im in training_data["images"]:
        current_ids.append(im["id"])
    
    objects = list_all_objects(s3_client, bucket)

    images = [file for file in objects if ".jpg" in file]
    texts = [file for file in objects if ".txt" in file]
    new_records = []
    adds = 0

    for image in images:
        image_id = image[:image.find(".")]
        if (f"{image_id}.txt" in texts) and (image_id not in current_ids):
            s3_client.download_file(bucket, image, "temp.jpg")
            add = {"id": image_id}
            add["color"] = check_binarized("temp.jpg")
            s3_client.download_file(bucket, f"{image_id}.txt", "temp.txt")
            with open("temp.txt", "r") as f:
                try:
                    for line in f:
                        add["text"] = line.replace("\n", "")
                except:
                    print(f"Failed to read text for {image}")
                    continue
            new_records.append(add)
            adds += 1           

    os.unlink("temp.txt")
    os.unlink("temp.jpg")
    update_training_data(new_records, bucket, s3_client)

    return adds

#print(f"{driver()} new HTR training data records added.")




