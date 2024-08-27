from block_transcribe import *
from driver import process_transcription
from segmentation_driver import segmentation_driver
import boto3
from eval_entry import *
from utility import load_volume_metadata

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

    segments = segmentation_driver(local_file_path, display=True)
    entries = []

    for segment in segments["text"]:
        segment_id = f"{image_id}-{segment['segment_id']}"
        s3.upload_file(f"segmented/{segment_id}-color.jpg", bucket_name, f'{segment_id}-color.jpg', ExtraArgs={'ContentType': "image/jpeg"})
        s3.upload_file(f"segmented/{segment_id}-pooled.jpg", bucket_name, f'{segment_id}-pooled.jpg', ExtraArgs={'ContentType': "image/jpeg"})
        output = transcribe_block(segment_id)
        entries.append(build_entry(output))

    volume_id = int(image_id.split("-")[0])    
    write_volume(volume_id, entries, output_path=f"testing/{volume_id}_full_demo_transcription.json")    
    process_transcription(f"testing/{volume_id}_full_demo_transcription.json", "instructions.json", "training_data.json",
                          training_keywords = {"type": "baptism", "country": "Cuba"}, mode = "and", out_path = f"testing/{volume_id}_full_demo_output.json")
    
actual_full_demo("239746-0088")

def volume_demo(volume_id, image_bucket="ssda-openai-test", local_image_dir="images"):
    volume_metadata = load_volume_metadata(volume_id)
    
    # Create an S3 client
    s3 = boto3.client('s3')

    entries = []

    start = 1
    if 'start_image' in volume_metadata['fields']:
        start = volume_metadata['fields']['start_image']

    end = volume_metadata['fields']['images'] + 1
    if 'end_image' in volume_metadata['fields']:
        end = volume_metadata['fields']['end_image'] + 1      
    
    for im in range(start, end):
        image_file_name = f'{volume_id}-{("0" * (4 - len(str(im))))}{str(im)}.jpg'
        print(f'Now segmenting and transcribing {image_file_name}.')
        local_file_path = f"{local_image_dir}/{image_file_name}"
        s3.download_file(image_bucket, image_file_name, local_file_path)
        segments = segmentation_driver(local_file_path)        
        for segment in segments["text"]:
            segment_id = f'{image_file_name[:image_file_name.find(".")]}-{segment["segment_id"]}'
            s3.upload_file(f"segmented/{segment_id}-color.jpg", image_bucket, f'{segment_id}-color.jpg', ExtraArgs={'ContentType': "image/jpeg"})
            s3.upload_file(f"segmented/{segment_id}-pooled.jpg", image_bucket, f'{segment_id}-pooled.jpg', ExtraArgs={'ContentType': "image/jpeg"})
            output = transcribe_block(segment_id)
            entries.append(build_entry(output))

    for index, entry in enumerate(entries):        
        entries[index]['eval'] = check_entry(entry)

    entries_to_delete = []
    for index, entry in enumerate(entries):
        if ('beginning' in entry['eval'].lower()) and ('end' in entries[index + 1]['eval'].lower()):
            print(f'Found a partial entry starting at {entry["id"]}.')
            entries[index]['raw'] += ' ' + entries[index + 1]['raw']
            entries_to_delete.append(index + 1)

    entries = [element for index, element in enumerate(entries) if index not in entries_to_delete]
    entries = [element for element in entries if ('end' not in element['eval'].lower())]
    for entry in entries:
        del entry['eval']
    write_volume(volume_id, entries, output_path=f"testing/{volume_id}_demo_transcription.json")    
    process_transcription(f"testing/{volume_id}_demo_transcription.json", "instructions.json", "training_data.json",
                          training_keywords = {"type": "baptism", "country": "Cuba"}, mode = "and", out_path = f"testing/{volume_id}_demo_output.json")            

# volume_demo(239746)