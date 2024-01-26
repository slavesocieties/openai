from parse import parse_xml
from normalize import *
from extract import *
from aggregate import *

def process_transcription(volume_record_path, instructions_path, training_data_path, training_keywords, out_path = None, mode = "or", shots = 1000, parse = False, testing = False):
    if parse:
        volume_record_path = parse_xml(volume_record_path)
        print("Parsing complete.")
        
    normalized = normalize_volume(volume_record_path, instructions_path, training_data_path, training_keywords, match_mode = mode, max_shots = shots, ld = not parse)

    print("Normalization complete.")

    extracted = extract_data_from_volume(normalized, instructions_path, training_data_path, training_keywords, output_path = None, match_mode = mode, max_shots = shots, ld = False)
    
    if testing:
        with open("extracted.json", "w", encoding="utf-8") as f:
            json.dump(extracted, f)

    print("Extraction complete.")

    volume_record = aggregate_entry_records(extracted, output_path=out_path, ld=False)

    print("Volume record saved.")

    return volume_record

process_transcription("testing\\166470_lg_sample.json", "instructions.json", "training_data.json", {"type": "baptism", "country": "Cuba"}, out_path = "testing\\166470_lg_sample_output.json", mode = "and")