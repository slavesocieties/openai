import json
from utility import *
from openai import OpenAI

def normalize_volume(volume_record_path, instructions_path, training_data_path, output_path = None, few_shot_strategy = None, max_shots = 1000, ld = True):
    data, volume_metadata = parse_volume_record(volume_record_path, load=ld)

    examples = generate_training_data(training_data_path, volume_metadata, few_shot_strategy, max_shots)
    instructions = collect_instructions(instructions_path, volume_metadata, "normalization")
    
    for x, entry in enumerate(data["entries"]):
        norm = normalize_entry(entry, volume_metadata, examples, instructions)
        data["entries"][x]["normalized"] = norm

    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return data

def normalize_entry(entry, volume_metadata, examples, instructions):    
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
                "content": f"Please normalize this transcription of a {volume_metadata['language']} {record_type}: `" + example["raw"] + "`"
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": example["normalized"]
            }
        )

    conversation.append(
        {
                "role": "user",
                "content": f"Please normalize this transcription of a {volume_metadata['language']} {record_type}: `" + entry["raw"] + "`"
        }
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",    
        messages = conversation
    )

    return response.choices[0].message.content