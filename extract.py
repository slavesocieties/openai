def extract_data_from_volume(volume_record_path, training_data_path, output_path = None, few_shot_strategy = None, max_shots = 1000):
    import json
    from utility import parse_volume_record   

    data, volume_metadata = parse_volume_record(volume_record_path)
    
    for x, entry in enumerate(data["entries"]):
        info = extract_data_from_entry(entry, volume_metadata, training_data_path, few_shot_strategy, max_shots)
        data["entries"][x]["data"] = json.loads(info)    
    
    if output_path == None:
        output_path = volume_record_path
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def extract_data_from_entry(entry, volume_metadata, training_data_path, few_shot_strategy, max_shots):
    from openai import OpenAI
    client = OpenAI()    
    import json
    from utility import generate_training_data, parse_record_type

    record_type = parse_record_type(volume_metadata)

    conversation = [
        {
            "role": "system",
            "content": f"You are assisting a historian of the early modern Atlantic with a large collection of transcriptions of Catholic sacramental records written in {volume_metadata['language']}. " \
            f"The historian will provide you with a transcription of a {record_type} and ask you to extract detailed information from that record. Output this information as JSON. "
        },
        {
            "role": "system",        
            "content": "These transcriptions contain information about people. You will create a record for each person, containing at minimum their name as it appears in the transcription. " \
            "Assign each person a unique identifier. These should take the form P##, where ## is a two-digit padded integer and P01 is assigned to the first person to appear in the transcription."            
        }
    ]

    if volume_metadata["language"] == "Spanish":
        instructions = [        
            {
                "role": "system",
                "content": "These records will sometimes contain information about honorific titles or military ranks earned by these people. When they do, please include this information in your output for the person in question. " \
                "Examples of honorific titles include (but are not limited to): Don, Doña, Doctor, Licenciado, or a variety of ecclesiastical titles such as Frai, Padre, or Hermano. List each title exactly as it appears and list multiple titles separately."
                "Examples of military ranks include (but are not limited to): Capitán, Sargento Mayor, or Alférez. List these ranks as their English equivalents (respectively for the listed examples: Captain, Sergeant Major, and Ensign)."
            },             
            {
                "role": "system",
                "content": "These records will sometimes contain information about the places of origin of these people. When they do, please include this information in your output for the person in question." \
                "The presence of this information will often be signalled by words such as `natural` for a single individual or `naturales` for multiple."
            },
            {
                "role": "system",
                "content": "These records will sometimes contain information about the ethnicities of these people. When they do, please include this information in your output for the person in question. " \
                "There are a large number of ethnonyms that might appear in these records, so their appearance should be determined contextually rather than relying on a controlled vocabulary."
            },
            {
                "role": "system",
                "content": "These records will sometimes contain information about the ages of these people. When they do, please include this information in your output for the person in question. " \
                "This information might appear in the forms of words including (but not limited to): parvulo/a (translate to `infant`), niño/a (translate to `child`), or adulto/a (translate to `adult`)."
            },
            {
                "role": "system",
                "content": "These records will sometimes contain information about the legitimacy of the birth of these people, particularly infants. " \
                "When they do, please include this information in your output for the person in question." \
                "This information will usually appear as either legítimo/a (for legitimate) or ilegítimo/a (for illegitimate)."
            },
            {
                "role": "system",
                "content": "These records will sometimes contain information about the occupations of these people. When they do, please include this information in your output for the person in question. " \
                "This information will most frequently appear as a variety of words including (but not limited to) religioso, eclesiástico, clérigo, or cura (translate to `cleric`)." \
                "Other possibilities include ingeniero/a (translate to `engineer`)."
            },
            {
                "role": "system",
                "content": "These records will sometimes contain information about the phenotypes of these people. When they do, please include this information in your output for the person in question. " \
                "This information might appear in the forms of words including (but not limited to): negro/a (list as `negro`), moreno/a (list as `moreno`), or pardo/a (list as `mixed-race`)."
            },
            {
                "role": "system",
                "content": "These records will sometimes contain information about the freedom status of these people. When they do, please include this information in your output for the person in question. " \
                "This information might appear explicitly as either esclavo/a (list as `enslaved`) or libre (list as `free`). " \
                "Freedom status can also be communicated implicitly, most notably when the enslaver of one or more individuals is listed."
            }
        ]

    instructions.append(
        {
        "role": "system",
        "content": "These records will sometimes contain information about relationships between people. " \
        "Each relationship that you identify should be between 2 people and appear twice in your output, once for each related person. " \
        "The relationship type that you list for each individual should indicate what the related person *is to them*." \
        "The possible values for relationship type are: `parent`, `child`, `spouse`, `enslaver`, `slave`, `godparent`, `godchild`, `grandparent`, and `grandchild`."
        }
    )

    if volume_metadata["type"] == "baptism":
        instructions.append(
            {
                "role": "system",
                "content": "Baptismal registers will usually contain information about exactly 1 baptism. They will sometimes also contain information about a birth, particularly when an infant is being baptized. " \
                "Baptisms and births are both events. For each event, record the type of the event (either `baptism` or `birth`), the unique identifier assigned to the principal (the person who was baptized or born), " \
                "and the date when the event took place. Represent dates in a YYYY-MM-DD format. If you can't find a complete date, include as much information as possible."
            }
        )

    for instruction in instructions:
        conversation.append(instruction)

    examples = generate_training_data(training_data_path, volume_metadata, few_shot_strategy, max_shots)

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

extract_data_from_volume("testing\\15834_sample.json", "training_data.json", output_path = "testing\\15834_modular_extraction.json", few_shot_strategy = None, max_shots = 1000)