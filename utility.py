import json
from random import sample

def generate_training_data(training_data_path, keywords, match_mode="or", max_shots=1000):    
    examples = []

    with open(training_data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    if match_mode == "or":
        for example in training_data["examples"]:
            for key in keywords:
                if example[key] == keywords[key]:
                    examples.append(example)
                    break                    
    else:
        for example in training_data["examples"]:
            match = True
            for key in keywords:
                if example[key] != keywords[key]:
                    match = False
            if match:
                examples.append(example)
    
    if len(examples) > max_shots:
        examples = sample(examples, max_shots)
    
    return examples

def parse_volume_record(volume_record_path, load = True):
    if load:
        with open(volume_record_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = volume_record_path

    if data["country"] in ["Colombia", "Cuba", "United States"]:
        language = "Spanish"
    else:
        language = "Portuguese"

    volume_metadata = {"type": data["type"], "language": language, "institution": data["institution"], "id": data["id"]}

    return data, volume_metadata

def parse_record_type(volume_metadata):
    if volume_metadata["type"] == "baptism":
        record_type = "baptismal register"
    elif volume_metadata["type"] == "marriage":
        record_type = "marriage register"
    elif volume_metadata["type"] == "burial":
        record_type = "burial register"
    else:
        record_type = "sacramental record"

    return record_type

def collect_instructions(instructions_path, volume_metadata, mode):
    with open(instructions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keywords = [mode, volume_metadata["language"], volume_metadata["type"]]

    instructions = []

    for instruction in data["instructions"]:
        match = True
        for keyword in instruction["cases"]:
            if keyword not in keywords:
                match = False
        if match:
            instructions.append(instruction)

    return sorted(instructions, key=lambda x: x["sequence"])