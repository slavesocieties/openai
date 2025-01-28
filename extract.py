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

##CONSTANTS
RECIPROCAL_RELS = {"parent": "child", "child": "parent", 
                    "enslaver": "slave", "slave": "enslaver", 
                    "spouse": "spouse",
                    "godparent": "godchild", "godchild": "godparent"}
NULLABLE_PEOPLE_PROPS = ['rank', 'origin', 'ethnicity', 'age', 'legitimate', 'occupation', 'phenotype', 'free']
PEOPLE_PROPS = NULLABLE_PEOPLE_PROPS + ['id', 'name', 'titles', 'relationships']
EVENT_PROPS = ['type', 'principals', 'date']

def extract_data_from_volume(volume_record_path, instructions_path, training_data_path, keywords = None, match_mode = "or", max_shots = 1000, output_path = None, log_path = "failure_log.json"):
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
        info = extract_data_from_entry(entry, volume_metadata, examples, instructions, log_path=log_path)

        if len(info) > 0:
            data["entries"][x]["data"] = json.loads(info)    
    
    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return data

def extract_data_from_entry(entry, volume_metadata, examples, instructions, log_fails=True, log_path="failure_log.json"):
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
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages = conversation        
    )

    # result = response.choices[0].message.content
    
    if log_fails:
        init_result = response.choices[0].message.content

        # log invalid jsons
        if not check_valid_json(init_result, entry['id'], log_path):
            return ""
        
        # log invalid people
        people_result, people_valid = log_failed_people(init_result, entry['id'], log_path)
        
        # log invalid events
        event_result, event_valid = log_failed_events(people_result, entry['id'], log_path)

        # log invalid relationships
        rel_result, rel_valid = log_failed_relationships(event_result, entry['id'], log_path)

        # log dropped property information
        nulled_result, props_valid = fill_nulls(rel_result, entry['id'], log_path)

        # log relationship reciprocity fixing
        result, recip_valid = fix_relationshiops(nulled_result, entry['id'], log_path)

    return result

def log_failure(failure_id, path, failure_type, failure_msg, original_data):
    """Logs a failure in the extraction process.

    Args:
        failure_id: the id of the failed entry. 

        path: path to failure log

        failure_type: type of failure (e.g. json, people, relationships, etc.)

        failure_msg: a message describing the failure

        original_data: the output that triggered the failure
    """    
    print(f"Logging {failure_type} failure for {failure_id}: {failure_msg}")
    if "/" in path and not os.path.exists(path.rsplit("/",1)[0]):
        os.makedirs(path.rsplit("/",1)[0])

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}
        data["entries"] = []
        data["outputs"] = []
    
    if failure_id not in [x['id'] for x in data["outputs"]]:
        try:
            output = json.loads(original_data)
            data["outputs"].append(dict({"id": failure_id, "body": output}))
        except:
            data["outputs"].append(dict({"id": failure_id, "body": original_data}))
    
    data["entries"].append(dict({"id": failure_id, "type": failure_type, "message": failure_msg}))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def check_valid_json(data, id, path):
    """Checks if a json is formatted validly. Output fails this check if:
        - it is not valid Json format
        - it is missing the "people" property

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        True if the output is valid Json, False if not.
    """    
    json_str = fr'{data}'
    try:
        init_result = json.loads(json_str)
        if 'people' in init_result:
            return True
        else:
            log_failure(id, path , "json", f"Invalid json: missing 'people'", data)
            return False
    except:
        log_failure(id, path , "json", f"Invalid json", data)
        return False
    
def log_failed_people(data, id, path):
    """Checks if the "people" property has errors, including
        - a person is missing the property "id"
        - a person is missing the property "name"

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """    
    json_str = fr'{data}'
    data = json.loads(json_str)
    
    success = True
    valid_people = []
    for p in data['people']:
        if 'id' not in p or 'name' not in p:
            log_failure(id, path , "people", f"Missing 'name' or 'id' property", data)
            success = False
        else:
            valid_people.append(p)

    data['people'] = valid_people
        
    return json.dumps(data), success

def log_failed_events(data, id, path):
    """Checks if the "events" property has errors, including
        - an event is missing the property "type"
        - an event is missing the property "principals"
        - an event has invalid principals

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """    
    json_str = fr'{data}'
    data = json.loads(json_str)
    
    success = True
    if 'events' in data:
        valid_events = []
        for e in data['events']:
            if 'type' not in e:
                success=False
                log_failure(id, path , "events", f"Missing 'type' property", data)
            elif 'principals' not in e:
                success=False
                log_failure(id, path , "events", f"Missing 'principals' property", data)
            else:
                valid_events.append(e)

        data['events'] = valid_events
        
        ids = [p['id'] for p in data['people']]

        for e in data['events']:
            if isinstance(e['principals'], list):
                valid_principals = []
                for p in e['principals']:
                    if p not in ids:
                        success=False
                        log_failure(id, path , "events", f"Invalid principal: {p}", data)
                    else:
                        valid_principals.append(p)
                e['principals'] = valid_principals

    return json.dumps(data), success

def log_failed_relationships(data, id, path):
    """Checks for errors in the "relationships" properties, including
        - a relationship is missing the property "related_person"
        - a relationship is missing the property "relationship_type"
        - a relationship has an invalid related person
        - a relationship has an invalid relationship type
        - the relationship has unexpected properties
        Does NOT check for relationship reciprocity

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """   
    json_str = fr'{data}'
    data = json.loads(json_str)

    ids = [p['id'] for p in data['people']]

    success = True

    for p in [p for p in data['people'] if 'relationships' in p and isinstance(p['relationships'], list)]:
        valid_rels = []
        for r in [r for r in p['relationships'] if isinstance(r, dict)]:
            failure_msg = ""
            if 'related_person' not in r:
                failure_msg = "missing related person"
            elif 'relationship_type' not in r:
                failure_msg = "missing relationship type"
            elif r['relationship_type'] not in RECIPROCAL_RELS.keys():
                failure_msg = f"invalid relationship type {r['relationship_type']}"
            elif r['related_person'] not in ids:
                failure_msg = f"invalid related person {r['related_person']}"
            elif len(r.keys()) > 2:
                failure_msg = f"unexpected relationship properties: {r.keys()}"
            if len(failure_msg) > 0:
                success=False
                log_failure(id, path, "relationship",
                            f"Invalid relationship for {p['id']}: {failure_msg}", data)
            else:
                valid_rels.append(r)
        
        p['relationships'] = valid_rels

    return json.dumps(data), success

def fill_nulls(data, id, path):
    """Fills missing properties with nulls and empty lists,
        and checks for errors with properties, including:
        - the data type of a property is wrong
        - there is an unexpected property for a person or event

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """   
    json_str = fr'{data}'
    data = json.loads(json_str)

    success = True
    for p in data['people']:
        for prop in NULLABLE_PEOPLE_PROPS:
            p[prop] = p.pop(prop) if prop in p else None
            if isinstance(p[prop], list):
                valid_values = [x for x in p[prop] if isinstance(x, str)]
                val = valid_values[0] if len(valid_values) > 0 else None
                log_failure(id, path, "people", 
                            f"Unexpected list property type for {p['id']}: {prop} = {val}", data)
                success = False
                p[prop] = val
        
        p['titles'] = p.pop('titles') if 'titles' in p else []
        if not isinstance(p['titles'], list):
            log_failure(id, path, "people", 
                        f"Titles of {p['id']} must be a list", data)
            success = False
            p['titles'] = [str(p['titles'])]
        
        p['relationships'] = p.pop('relationships') if 'relationships' in p else []
        if not isinstance(p['relationships'], list):
            log_failure(id, path, "relationships", 
                        f"Relationships of {p['id']} must be a list", data)
            success = False
            p['relationships'] = []
        
        valid_props = {}
        for k,v in p.items():
            if k not in PEOPLE_PROPS:
                log_failure(id, path, "people", 
                            f"Invalid property for person {p['id']}: {k} = {v}", data)
                success=False
            else:
                valid_props[k] = v
        p = valid_props

    if 'events' in data:
        valid_events = []
        for e in data['events']:
            if 'date' in e and not isinstance(e['date'], str):
                log_failure(id, path, "events", 
                            f"Invalid date", data)
                success = False
                e['date'] = None
            
            if 'principals' in e and not isinstance(e['principals'], list):
                log_failure(id, path, "events", 
                            f"Principals must be a list", data)
                success = False
                e['principals'] = [str(e['principals'])]

            valid_props = {}
            for k,v in e.items():
                if k not in EVENT_PROPS:
                    log_failure(id, path, "events", 
                                f"Invalid event property: {k} = {v}", data)
                    success=False
                else:
                    valid_props[k] = v
            e = valid_props

            if e['type'] == 'marriage' and len(e['principals']) != 2:
                log_failure(id, path, "events", 
                            f"Invalid event principals: marriage must have 2 principals.", data)
                success=False
            elif e['type'] == 'baptism' and len(e['principals']) != 1:
                log_failure(id, path, "events", 
                            f"Invalid event principals: baptism must have 1 principal.", data)
                success=False
            else:
                valid_events.append(e)
    
        data['events'] = valid_events

    return json.dumps(data), success

def fix_relationshiops(data, id, path):
    """Checks for irreciprocal relationship errors and fixes them with the following assumptions:
        - a principal is always the child, slave, godchild, or spouse in the relationship
        - irreciprocal relationships involving two non-principals are un-fixable
        - an irreciprocal relationship involving two unrelated relationship types 
            (eg child and slave) is un-fixable
        - if a relationship exists in one direction, it should exist in the other as well
        (unidirectional relationships are not dropped, assume a miss rather than a hallucination)

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """       
    json_str = fr'{data}'
    data = json.loads(json_str)

    success = True

    relationships = {}
    for p in data['people']:
        relationships[p['id']] = {}
        for r in p['relationships']:
            relationships[p['id']][r['related_person']] = r['relationship_type']
    

    def del_relation(p1, p2):
        for rel_list in [p['relationships'] for p in data['people'] if p['id'] == p1]:
            rel_list = [r for r in rel_list if r['related_person'] != p2]

    def add_relation(p1, p2, type):
        del_relation(p1, p2)
        for rel_list in [p['relationships'] for p in data['people'] if p['id'] == p1]:
            rel_list.append(dict({"related_person": p2, "relationship_type": type}))

    def is_principal(p):
       return p in data['events'][0]['principals']

    for p1 in relationships.keys():
        for p2 in relationships[p1].keys():
            rel = relationships[p1][p2]
            if p1 not in relationships[p2]:
                success = False
                log_failure(id, path, "relationships", 
                            f"Irreciproal relationship for {p1} and {p2}: None and {rel}", data)
                add_relation(p2, p1, RECIPROCAL_RELS[rel])
            elif relationships[p2][p1] ==  RECIPROCAL_RELS[rel]:
                ##Valid reciprocal relationship
                pass
            elif relationships[p2][p1] !=  rel or (not is_principal(p1) and not is_principal(p2)):
                ##Give up
                success = False
                log_failure(id, path, "relationships", 
                            f"Unfixable irreciproal relationship for {p1} and {p2}: {relationships[p2][p1]} and {rel}", data)
                del_relation(p2, p1)
                del_relation(p1, p2)
            else:
                ##Fixable
                success = False
                log_failure(id, path, "relationships", 
                            f"Fixable irreciproal relationship for {p1} and {p2}: {relationships[p2][p1]} and {rel}", data)
                principal = p1 if is_principal(p1) else p2
                other = p2 if principal == p1 else p1
                if rel in ["child", "slave", "godchild"]:
                    add_relation(principal, other, RECIPROCAL_RELS[rel])
                    add_relation(other, principal, rel)
                else:
                    add_relation(other, principal, RECIPROCAL_RELS[rel])
                    add_relation(principal, other, rel)
                
    
    return json.dumps(data), success
