from block_transcribe import *
from driver import process_transcription

def process_image(image_id):
    output = transcribe_block(image_id)
    entries = [build_entry(output)]
    volume_id = int(image_id.split("-")[0])    
    write_volume(volume_id, entries, output_path=f"testing/{volume_id}_transcription.json")    
    process_transcription(f"testing/{volume_id}.json", "instructions.json", "training_data.json",
                          training_keywords = {"type": "baptism", "country": "Cuba"}, mode = "and", out_path = f"testing/{volume_id}_demo_output.json")    

process_image("239746-0076-01")