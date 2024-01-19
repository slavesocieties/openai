import json
from random import sample

def generate_training_data(training_data_path, volume_metadata, few_shot_strategy, max_shots):    
    examples = []

    with open(training_data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    if few_shot_strategy == "v":
        examples = [x for x in training_data["examples"] if x["id"] == volume_metadata["id"]]        
    elif few_shot_strategy == "l":
        examples = [x for x in training_data["examples"] if x["language"] == volume_metadata["language"]]
    elif few_shot_strategy == "t":
        examples = [x for x in training_data["examples"] if x["type"] == volume_metadata["type"]]
    elif few_shot_strategy == "lt":
        examples = [x for x in training_data["examples"] if ((x["language"] == volume_metadata["language"]) and (x["type"] == volume_metadata["type"]))]
    elif few_shot_strategy == "i":
        examples = [x for x in training_data["examples"] if ((x["institution"] == volume_metadata["institution"]) and (x["type"] == volume_metadata["type"]))]
    else:
        examples = training_data["examples"]

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