from parse import parse_xml
from normalize import *
from extract import *

def process_transcription(volume_record_path, instructions_path, training_data_path, training_keywords, out_path = None, mode = "or", shots = 1000, parse = False):
    if parse:
        volume_record_path = parse_xml(volume_record_path)
        print("Parsing complete.")
        
    normalized = normalize_volume(volume_record_path, instructions_path, training_data_path, training_keywords, match_mode = mode, max_shots = shots, ld = not parse)

    print("Normalization complete.")

    extracted = extract_data_from_volume(normalized, instructions_path, training_data_path, training_keywords, output_path = out_path, match_mode = mode, max_shots = shots, ld = False)

    print("Extraction complete.")

    return extracted

process_transcription("testing\\FHL_007548705_sample.json", "instructions.json", "training_data.json", {"language": "Spanish", "country": "United States"}, out_path = "FHL_007548705.json", mode = "and")