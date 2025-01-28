import datetime
import os
import traceback

from aggregate import aggregate_entry_records
from extract import extract_data_from_volume
from normalize import normalize_volume
import json

## os.environ["OPENAI_API_KEY"] = ""

# with open('invalid_jsons.json', 'r') as f:
#     str = f.read()
#     json_str = fr'{str}'
#     print(dict(json.loads(json_str)))
#     print(validate_extracted_json(str))

print(datetime.datetime.now())

json_name = "1795"
# rec = process_transcription(f"json/{json_name}.json", "instructions.json", "training_data.json",
#                           training_keywords = {"type": "marriage", "country": "Cuba"}, mode = "and", out_path = f"sam_test/output_{json_name}.json")            

try:
    # normalized = normalize_volume(f"json/{json_name}.json", "instructions.json", "training_data.json", keywords = {"type": "marriage", "country": "Cuba"}, match_mode = "and", max_shots = 1000)

    # with open(f'big_test/normalized_intermediate_{datetime.datetime.now()}.json', 'w') as f:
    #     json.dump(normalized, f)

    with open(f'big_test/normalized_intermediate_2024-11-07 17:44:16.351828.json', 'r') as f:
        normalized = json.load(f)

    extracted = extract_data_from_volume(normalized, "instructions.json", "training_data.json", keywords = {"type": "marriage", "country": "Cuba"}, match_mode = "and", max_shots = 1000, output_path=f"sam_test/extracted_{json_name}.json")

    with open(f'big_test/extracted_intermediate_{datetime.datetime.now()}.json', 'w') as f:
        json.dump(extracted, f)

    volume_record = aggregate_entry_records(extracted, output_path=f"big_test/output_{json_name}_{datetime.datetime.now()}.json")
    
except Exception as error:
    print("FAILED!")
    print(error)
    print(traceback.format_exc())

print(datetime.datetime.now())
