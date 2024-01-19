import json
from utility import *
from openai import OpenAI

def normalize_volume(volume_record_path, training_data_path, output_path = None, few_shot_strategy = None, max_shots = 1000, ld = True):
    data, volume_metadata = parse_volume_record(volume_record_path, load=ld)

    examples = generate_training_data(training_data_path, volume_metadata, few_shot_strategy, max_shots)

    for x, entry in enumerate(data["entries"]):
        norm = normalize_entry(entry, volume_metadata, examples)
        data["entries"][x]["normalized"] = norm

    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return data

def normalize_entry(entry, volume_metadata, examples):    
    client = OpenAI()    

    record_type = parse_record_type(volume_metadata)

    conversation = [
        {
                "role": "system",
                "content": f"You are assisting a historian of the early modern Atlantic with a large collection of transcriptions of Catholic sacramental records written in {volume_metadata['language']}. " \
                f"The historian will provide you with a transcription of a {record_type} written in early modern {volume_metadata['language']} and ask you to normalize it by expanding abbreviations, " \
                "correcting idiosyncratic or archaic spellings, modernizing capitalization and punctuation, and correcting obvious transcription errors."
        }
    ] 

    if volume_metadata["language"] == "Spanish":
        conversation.append(
            {
                "role": "system",
                "content": "Expand abbreviated names as well as words. Commonly abbreviated first names include Antonio or Antonia, Domingo or Dominga, Francisco or Francisca, " \
                "or Juan or Juana. Commonly abbreviated last names include Fernandez, Gonzalez, Hernandez, or Rodriguez. These are not intended to be complete lists. Use context " \
                "to determine when a name has been abbreviated and your knowledge of Spanish names to determine what the abbreviated name is."
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