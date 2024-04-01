from block_transcribe import *
from driver import process_transcription
from segmentation_driver import segmentation_driver
import boto3

def process_image(image_id):
    output = transcribe_block(image_id)
    entries = [build_entry(output)]
    volume_id = int(image_id.split("-")[0])    
    write_volume(volume_id, entries, output_path=f"testing/{volume_id}_transcription.json")    
    process_transcription(f"testing/{volume_id}.json", "instructions.json", "training_data.json",
                          training_keywords = {"type": "baptism", "country": "Cuba"}, mode = "and", out_path = f"testing/{volume_id}_demo_output.json")    

#process_image("239746-0076-01")
    
def actual_full_demo(image_id, bucket_name="ssda-openai-test", local_file_path="images"):
    # Create an S3 client
    s3 = boto3.client('s3')
    
    # Download the file
    s3_file_key = f"{image_id}.jpg"
    local_file_path = f"{local_file_path}/{s3_file_key}"
    s3.download_file(bucket_name, s3_file_key, local_file_path)

    segments = segmentation_driver(local_file_path)
    entries = []

    for segment in segments:
        segment_id = segment["id"]
        s3.upload_file(f"segmented/{segment_id}-color.jpg", bucket_name, f'{segment_id}-color.jpg')
        s3.upload_file(f"segmented/{segment_id}-pooled.jpg", bucket_name, f'{segment_id}-pooled.jpg')
        output = transcribe_block(segment_id)
        entries.append(build_entry(output))

    volume_id = int(image_id.split("-")[0])    
    write_volume(volume_id, entries, output_path=f"testing/{volume_id}_full_demo_transcription.json")    
    process_transcription(f"testing/{volume_id}.json", "instructions.json", "training_data.json",
                          training_keywords = {"type": "baptism", "country": "Cuba"}, mode = "and", out_path = f"testing/{volume_id}_full_demo_output.json")
    
actual_full_demo("239746-0088")