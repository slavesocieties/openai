from parse import parse_xml
from normalize import *
from extract import *

def process_transcription(volume_record_path, instructions_path, training_data_path, out_path = None, strategy = None, shots = 1000, parse = False):
    if parse:
        volume_record_path = parse_xml(volume_record_path)
        print("Parsing complete.")
        
    normalized = normalize_volume(volume_record_path, instructions_path, training_data_path, output_path = None, few_shot_strategy = strategy, max_shots = shots, ld = not parse)

    print("Normalization complete.")

    extracted = extract_data_from_volume(normalized, instructions_path, training_data_path, output_path = out_path, few_shot_strategy = None, max_shots = 1000, ld = False)

    print("Extraction complete.")

    return extracted

process_transcription("testing\\FHL_007548705_sample.json", "instructions.json", "training_data.json", out_path = "FHL_007548705.json", strategy = "i")