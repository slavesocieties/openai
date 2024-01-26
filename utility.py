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

    if data["country"] in ["Colombia", "Cuba", "Mexico", "United States"]:
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

def parse_date(date):
    if "/" in date:
        dates = date.split("/")
        start = dates[0].split("-")
        end = dates[1].split("-")
        parts = []
        for part in start:
            parts.append(int(part))
        for part in end:
            parts.append(int(parts))        
    else:
        parts = date.split("-")
        for part in parts:
            part = int(part)
    
    return parts

def compare_dates(x, y):
    if x[0] < y[0]:
        return True
    elif x[0] > y[0]:
        return False
    else:
        if x[1] < y[1]:
            return True
        elif x[1] > y[1]:
            return False
        else:
            if x[2] < y[2]:
                return True
            elif x[2] > y[2]:
                return False
            else:
                return True
            
def complete_date(date, mode="m"):
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if type(date) == str:
        date = parse_date(date)

    if mode == "s":
        if len(date) == 1:
            return date[0], 1, 1
        else:
            return date[0], date[1], 1
    elif mode == "e":
        if len(date) == 1:
            return date[0], 12, 31
        else:
            return date[0], date[1], months[date[1] - 1]
    else:
        if len(date) == 1:
            return date[0], 1, 1, date[0], 12, 31
        else:
            return date[0], date[1], 1, date[0], date[1], months[date[1] - 1]
        
def disambiguate_people(x, y):
    for key in ["rank", "origin", "ethnicity", "age", "legitimate", "occupation", "phenotype", "free"]:
        if (key in x) and (key in y) and (x[key] != y[key]):
            return False    
    people = {"people": [x, y]}
    with open("disambiguate.json", "w", encoding="utf-8") as f:
        json.dump(people, f)

    match = input("Should these records be combined? (y/n)")

    if match == "y":
        return True
    
    return False

def merge_records(x, y):
    for key in ["rank", "origin", "ethnicity", "age", "legitimate", "occupation", "phenotype", "free"]:
        if (key in y) and (key not in x):
            x[key] = y[key]

    if ("titles" in x) and ("titles" in y):
        for title in y["titles"]:
            if title not in x["titles"]:
                x["titles"].append(title)
    elif "titles" in y:
        x["titles"] = y["titles"]

    if ("relationships" in x) and ("relationships" in y):
        for rel in y["relationships"]:            
            x["relationships"].append(rel)
    elif "relationships" in y:
        x["relationships"] = y["relationships"]

    if (type(x["id"]) == str) and (type(y["id"]) == str):
        x["id"] = [x["id"], y["id"]]
    elif type(x["id"]) == str:
        x["id"] = [x["id"]]
        for id in y["id"]:
            x["id"].append(id)
    elif type(y["id"]) == str:
        x["id"].append(y["id"])
    else:
        for id in y["id"]:
            x["id"].append(id)

    return x
    