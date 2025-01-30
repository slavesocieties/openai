import json
from random import sample
from openai import OpenAI
import boto3

client = OpenAI()

def generate_training_data(training_data_path, volume_metadata, few_shot_strategy, max_shots):    
    examples = []

    bucket_name = training_data_path["bucket"]
    file_key = training_data_path["key"]

    s3_client = boto3.client('s3')

    try:
        # Get the file object from S3
        file_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read the file content
        file_content = file_object['Body'].read().decode('utf-8')
        
        # Convert the file content to a dictionary
        training_data = json.loads(file_content)        
       
    except Exception as e:
        print(e)
        return {
            'statusCode': 500,
            'body': json.dumps("Error reading training data from S3.")
        }

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

def parse_volume_record(data):
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

def normalize_volume(volume_record_path, training_data_path, few_shot_strategy = None, max_shots = 1000):
    data, volume_metadata = parse_volume_record(volume_record_path)

    examples = generate_training_data(training_data_path, volume_metadata, few_shot_strategy, max_shots)

    for x, entry in enumerate(data["entries"]):
        norm = normalize_entry(entry, volume_metadata, examples)
        data["entries"][x]["normalized"] = norm

    return data, examples

def normalize_entry(entry, volume_metadata, examples):    
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

def extract_data_from_volume(volume_record_path, examples):
    data, volume_metadata = parse_volume_record(volume_record_path)
    
    for x, entry in enumerate(data["entries"]):
        info = extract_data_from_entry(entry, volume_metadata, examples)
        data["entries"][x]["data"] = json.loads(info)    
    
    return data

def extract_data_from_entry(entry, volume_metadata, examples):
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

def process_event(event):
    body = json.loads(event['body'])

    if "training_strategy" in body:
        strat = body["training_strategy"]
    else:
        strat = None

    if "num_examples" in body:
        num = body["num_examples"]
    else:
        num = 1000

    training_data_path = {"bucket": "ssda-nlp", "key": "training_data.json"}
    
    if "train_bucket" in body:
        training_data_path["bucket"] = body["train_bucket"]

    if "train_key" in body:
        training_data_path["key"] = body["train_key"]            

    normalized, examples = normalize_volume(body, training_data_path, few_shot_strategy = strat, max_shots = num)

    return extract_data_from_volume(normalized, examples) 

def lambda_handler(event, context):
    if (event['body']) and (event['body'] != None):
            message = process_event(event)
            return {
                'statusCode': 200,                
                'body': json.dumps(message)
            }
    else:
         return {
              'statusCode': 400
         }

"""event = {}

with open("test_request_body.json", "r", encoding="utf-8") as f:
    body = ""
    for line in f:
        body += line

event["body"] = body

output = lambda_handler(event, "fish")

output = json.loads(output["body"])

with open("test_request_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f)"""