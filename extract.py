"""Computationally extracts content from normalized text.

Uses natural language instructions and manually created examples
to construct a conversation history that is then passed to an LLM
for chat completion via an api. The api should respond with a str
representation of a json document containing extracted content.
"""

from openai import OpenAI    
import json
from utility import *
# from schema import schema
import re

def extract_data_from_volume(volume_record_path, instructions_path, training_data_path, keywords = None, match_mode = "or", max_shots = 1000, output_path = None, log_prefix = ""):
    """Extracts content from a series of transcribed entries from a historical document.

    Args:
        volume_record_path: either path to a json file containing a volume record or
        a dictionary containing this data

        instructions_path (str): path to a json file containing natural language instructions
        that will be passed to the llm as system messages

        training_data_path (str): path to a json file containing manually constructed examples
        of content extraction that will be used to train the llm

        keywords (list, optional): list of keywords that define a subset of training
        data to use in conjunction with the next parameter; if not included, all available
        training data will be used

        match_mode (str, optional): either `and` or `or`, defines subset of training data to use
        in conjunction with previous parameter

        max_shots (int, optional): defines maximum number of examples to include in conversation
        history supplied to llm
        
        out_path (str, optional): path to output volume record with extracted content to; volume
        record will not be saved if this is not included

        log_prefix (str, optional): a directory to log failed relationships (ending with /)

    Returns:
        Dict containing volume record and extracted content. 
    """
    data, volume_metadata = parse_volume_record(volume_record_path)

    examples = generate_training_data(training_data_path, keywords, match_mode=match_mode, max_shots=max_shots)
    instructions = collect_instructions(instructions_path, volume_metadata, "extraction")
    
    for x, entry in enumerate(data["entries"]):
        info = extract_data_from_entry(entry, volume_metadata, examples, instructions, log_prefix=log_prefix)

        if len(info) > 0:
            data["entries"][x]["data"] = json.loads(info)    
    
    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return data

def extract_data_from_entry(entry, volume_metadata, examples, instructions, log_fails=True, log_prefix=""):
    """Extracts content from a single transcribed entry from a historical document.

    Args:
        entry (dict): entry record minimally including normalized entry text as
        the value assigned to the key `normalized`

        volume_metadata (dict): dict containing basic volume metadata

        examples (list): training data to be used to fine-tune model

        instructions (list): instructions to be passed to model as system messages

        log_prefix (str, optional): a directory to log failed relationships (ending with /)

    Returns:
        str representation of a json document containing extracted content 
    """    
    client = OpenAI()
    
    record_type = parse_record_type(volume_metadata)

    conversation = []

    #add natural language instructions to conversation history as system messages
    for instruction in instructions:
        conversation.append(
            {
                    "role": "system",
                    "content": instruction["text"]
            }
        )    
    
    #add training examples to conversation history as responses to previous queries
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

    #add new query to conversation history
    conversation.append(
        {
            "role": "user",
            "content": "Please extract information from this transcription of a Spanish baptismal register: `" + entry["normalized"] + "`"
        }
    )
    
   
    #generate response from llm
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_format={"type": "json_object"},
        messages = conversation        
    )

    # result = response.choices[0].message.content
    
    if log_fails:
        init_result = response.choices[0].message.content

        # log invalid jsons
        if not check_valid_json(init_result) or 'people' not in init_result:
            log_failure(f"{log_prefix}invalid_jsons.json", init_result, entry)
            return ""
        
        # log invalid people
        people_result, people_valid = log_failed_people(init_result)
        if not people_valid:
            log_failure(f"{log_prefix}invalid_people.json", init_result, entry)
        
        # log invalid events
        event_result, people_valid = log_failed_people(people_result)
        if not people_valid:
            log_failure(f"{log_prefix}invalid_events.json", people_result, entry)

        # log invalid relationships
        rel_result, rel_valid = log_failed_relationships(event_result)
        if not rel_valid:
            log_failure(f"{log_prefix}invalid_relationships.json", event_result, entry)
        
        # log dropped property information
        result, valid_props = fill_nulls(rel_result)
        if not valid_props:
            log_failure(f"{log_prefix}invalid_properties.json", rel_result, entry)

    return result

def check_valid_json(data):
    json_str = fr'{data}'
    try:
        _ = json.loads(json_str)
        return True
    except:
        return False
    
def log_failed_people(data):
    json_str = fr'{data}'
    data = json.loads(json_str)
    
    success = True
    for p in data['people']:
        if 'id' not in p or 'name' not in p:
            success = False

    data['people'] = [p for p in data['people'] if 'id' in p and 'name' in p]
        
    return json.dumps(data), success

def log_failed_events(data):
    json_str = fr'{data}'
    data = json.loads(json_str)
    
    success = True
    if 'events' in data:
        for e in data['events']:
            if 'type' not in e or 'principals' not in e or not isinstance(e['principals'], list):
                success = False

        data['events'] = [e for e in data['events'] if 'type' in e and 'principals' in e and isinstance(e['principals'], list)]
        
    return json.dumps(data), success

def log_failed_relationships(data):
    """Logs any entries with invalid relationships.

    Args:
        data (string): json string response from the model

    Returns:
        tuple of (str, boolean) where the first element is a json string with no relationships to non-existant people 
    """    
    reciprocal_rels = {"parent": "child", "child": "parent", 
                       "enslaver": "slave", "slave": "enslaver", 
                       "spouse": "spouse",
                       "godparent": "godchild", "godchild": "godparent"}

    json_str = fr'{data}'
    data = json.loads(json_str)

    ids = [p['id'] for p in data['people']]

    valid = True

    relationships = {}
    for p in [p for p in data['people'] if 'relationships' in p]:
        relationships[p['id']] = {}
        for r in [p for p in p['relationships'] if 'relationship_type' in p and'related_person' in p]:
            relationships[p['id']][r['related_person']] = [r['relationship_type']  for r in p['relationships']]
        valid_rels = [r for r in p['relationships'] if 'related_person' in r and 
                              'relationship_type' in r and r['related_person'] in ids]
        if len(p['relationships']) != len(valid_rels):
            valid=False
            p['relationships'] = valid_rels

    for p1 in relationships:
        for p2, rel in p.items():
            if p2 not in relationships or relationships[p2][p1] !=  reciprocal_rels[rel]:
                valid = False
    
    if 'events' in data:
        valid_events = [e for e in data['events'] if 'principals' in e]
        if len(data['events']) != len(valid_events):
            valid=False
            data['events'] = valid_events
        
        for e in data['events']:
            valid_principals = [p for p in e['principals'] if p in ids]
            if len(e['principals']) != len(valid_principals):
                valid=False
                e['principals'] = valid_principals

    return json.dumps(data), valid

def log_failure(path, cur_result, entry):
    cur_entry = entry
    cur_entry["data"] = json.loads(cur_result)
    if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                    data["entries"].append(entry)
    else:
        data = {}
        data["entries"] = [entry] 
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def fill_nulls(data):
    json_str = fr'{data}'
    data = json.loads(json_str)

    nullable_props = ['rank', 'origin', 'ethnicity', 'age', 'legitimate', 'occupation', 'phenotype', 'free']
    people_props = nullable_props + ['id', 'name', 'titles', 'relationships']

    success = True
    for p in data['people']:
        p['titles'] = p.pop('titles') if 'titles' in p and isinstance(p['titles'], list) else []
        for prop in nullable_props:
            p[prop] = p.pop(prop) if prop in p else None
            if isinstance(p[prop], list):
                valid_values = [x for x in p[prop] if isinstance(x, str)]
                val = valid_values[0] if len(valid_values) > 0 else None
                p[prop] = val
                success = False
        p['relationships'] = p.pop('relationships') if 'relationships' in p and isinstance(p['relationships'], list) else []
        original_len = len(p)
        p = {k:v for k,v in p.items() if k in people_props}
        if len(p) != original_len:
            success=False
    event_props = ['type', 'principals', 'date']
    if 'events' in data:
        for e in data['events']:
            e['date'] = e.pop('date') if 'date' in e and isinstance(e['date'], str) else None
            original_len = len(e)
            e = {k:v for k,v in e.items() if k in event_props}
            if len(e) != original_len:
                success=False
        
    return json.dumps(data), success