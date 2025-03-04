import json
import boto3
import os
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import sys

# Add the repository root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from pool_image import block_image
from binarize_data import *

def download_image(s3_client, bucket_name, key):
    """Download an image from S3 and return it as a PIL image."""
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    return Image.open(BytesIO(response['Body'].read()))

def upload_image(s3_client, bucket_name, key, image):
    """Upload an image to S3."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket_name, Key=key, Body=buffer, ContentType="image/jpeg")

def pil_to_cv2(image):
    """Convert a PIL image to an OpenCV image."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def process_annotations(annotation_file, source_bucket, destination_bucket):
    """Process the JSON annotation file and crop/upload images accordingly."""
    s3_client = boto3.client('s3')
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    for image_info in data["images"]:
        filename = image_info["filename"]
        annotations = image_info["annotations"]
        
        # Download image
        image = download_image(s3_client, source_bucket, filename)                
        
        # Filter relevant text annotations (non-partial, non-margin)
        text_annotations = [
            anno for anno in annotations 
            if anno["name"] == "text" and not anno["attributes"]["partial"] and not anno["attributes"]["margin"]
        ]
        
        # Sort annotations by vertical position (top coordinate)
        text_annotations.sort(key=lambda x: x["polygon"][1])
        
        # Crop and upload text regions
        for idx, anno in enumerate(text_annotations, start=1):
            left, top, right, bottom = anno["polygon"]
            cropped_image = image.crop((left, top, right, bottom))
            cv_image = pil_to_cv2(cropped_image)
            pooled_image = block_image(cv_image, data=True)
            pooled_image = Image.fromarray(pooled_image, mode="L")            
            degree, pooled_image = rotate_block(pooled_image)
            degree, cropped_image = rotate_block(cropped_image, degree=degree)
            cropped_image = Image.fromarray(cropped_image)
            pooled_image = Image.fromarray(pooled_image).convert("RGB")
            new_filename = f"{os.path.splitext(filename)[0]}-{idx:02d}-color.jpg"
            upload_image(s3_client, destination_bucket, new_filename, cropped_image)
            print(f"Uploaded {new_filename} to {destination_bucket}")            
            new_filename = f"{os.path.splitext(filename)[0]}-{idx:02d}-pooled.jpg"
            upload_image(s3_client, destination_bucket, new_filename, pooled_image)
            print(f"Uploaded {new_filename} to {destination_bucket}")

# Example usage
annotation_file = "image_annotations/annotations.json"
source_s3_bucket = "ssda-production-jpgs"
destination_s3_bucket = "ssda-fine-tuning"
process_annotations(annotation_file, source_s3_bucket, destination_s3_bucket)
