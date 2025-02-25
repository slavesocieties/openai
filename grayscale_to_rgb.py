import json
import boto3
from PIL import Image
import io
import os

# Configure S3
BUCKET_NAME = "ssda-openai-test"  # Replace with your S3 bucket name
s3_client = boto3.client("s3")

# Load the JSONL file
jsonl_file_path = "fine_tune.jsonl"  # Update with correct path
updated_jsonl_file_path = "fine_tune_updated.jsonl"

grayscale_images = []

# Read the JSONL file and find grayscale images
updated_examples = []

# Read the JSONL file and find grayscale images
with open(jsonl_file_path, "r") as file:
    for line in file:        
        example = json.loads(line)        
        updated_example = {"messages": []}        
        for message in example["messages"]:            
            try:
                image_url = message["content"][2]["image_url"]["url"]                
                grayscale_images.append(image_url[image_url.rfind('/') + 1:])
                # Generate new key for the RGB image
                base_name, _ = os.path.splitext(image_url)  # Remove file extension
                new_image_url = f"{base_name}-rgb.jpg"
                message["content"][2]["image_url"]["url"] = new_image_url
            except:
                continue
            updated_example["messages"].append(message)
        
        # Store all examples to modify later
        updated_examples.append(example)

# Process each grayscale image
for image_key in grayscale_images:
    try:
        # Download the image from S3
        print(f"Processing: {image_key}")
        s3_object = s3_client.get_object(Bucket=BUCKET_NAME, Key=image_key)
        image_data = s3_object["Body"].read()
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_data))

        # Check if image is grayscale
        if image.mode != "RGB":
            print(f"Converting {image_key} from {image.mode} to RGB...")
            image = image.convert("RGB")

            # Generate new key for the RGB image
            base_name, _ = os.path.splitext(image_key)  # Remove file extension
            new_image_key = f"{base_name}-rgb.jpg"

            # Save to a bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)

            # Upload new RGB image to S3
            s3_client.put_object(Bucket=BUCKET_NAME, Key=new_image_key, Body=buffer, ContentType="image/jpeg")
            print(f"Uploaded converted image: {new_image_key}")            

    except Exception as e:
        print(f"Error processing {image_key}: {e}")

# Save the updated JSONL file
with open(updated_jsonl_file_path, "w") as file:
    for example in updated_examples:
        file.write(json.dumps(example) + "\n")

print(f"Updated JSONL file saved to: {updated_jsonl_file_path}")
print("Processing complete.")