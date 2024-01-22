from openai import OpenAI    
import json
from utility import *


def extract_data_from_volume(volume_record_path, instructions_path, training_data_path, keywords, output_path = None, match_mode = "or", max_shots = 1000, ld = True):
    data, volume_metadata = parse_volume_record(volume_record_path, load=ld)

    examples = generate_training_data(training_data_path, keywords, match_mode=match_mode, max_shots=max_shots)
    instructions = collect_instructions(instructions_path, volume_metadata, "extraction")
    
    for x, entry in enumerate(data["entries"]):
        info = extract_data_from_entry(entry, volume_metadata, examples, instructions)
        data["entries"][x]["data"] = json.loads(info)    
    
    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return data

def extract_data_from_entry(entry, volume_metadata, examples, instructions):    
    client = OpenAI()
    
    record_type = parse_record_type(volume_metadata)

    conversation = []

    for instruction in instructions:
        conversation.append(
            {
                    "role": "system",
                    "content": instruction["text"]
            }
        )    
    
    for example in examples:
        conversation.append(
            {
                "role": "user",
                "content": f"Please extract information from this transcription of a {volume_metadata['language']} {record_type}: `" + example["normalized"] + "`"
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": json.dumps(example["data"])
            }
        )

    conversation.append(
        {
            "role": "user",
            "content": "Please extract information from this transcription of a Spanish baptismal register: `" + entry["normalized"] + "`"
        }
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages = conversation        
    )

    return response.choices[0].message.content