import json
import boto3
from PIL import Image
import io

# Configure S3
BUCKET_NAME = "ssda-openai-test"  # Replace with your S3 bucket name
# s3_client = boto3.client("s3")

# Load the JSONL file
jsonl_file_path = "fine_tune.jsonl"  # Update with correct path
grayscale_images = []

# Read the JSONL file and find grayscale images
with open(jsonl_file_path, "r") as file:
    for line in file:
        example = json.loads(line)
        for message in example:
            """try:
                grayscale_images.append(message["content"][2]["image_url"])
            except:
                continue"""

print(grayscale_images)

"""# Process each grayscale image
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

            # Save to a bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Upload back to S3
            s3_client.put_object(Bucket=BUCKET_NAME, Key=image_key, Body=buffer, ContentType="image/png")
            print(f"Uploaded converted image: {image_key}")

    except Exception as e:
        print(f"Error processing {image_key}: {e}")

print("Processing complete.")
"""