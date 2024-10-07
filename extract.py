"""Computationally extracts content from normalized text.

Uses natural language instructions and manually created examples
to construct a conversation history that is then passed to an LLM
for chat completion via an api. The api should respond with a str
representation of a json document containing extracted content.
"""

from openai import OpenAI    
import json
from utility import *

def extract_data_from_volume(volume_record_path, instructions_path, training_data_path, keywords = None, match_mode = "or", max_shots = 1000, output_path = None):
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

    Returns:
        Dict containing volume record and extracted content. 
    """
    data, volume_metadata = parse_volume_record(volume_record_path)

    examples = generate_training_data(training_data_path, keywords, match_mode=match_mode, max_shots=max_shots)
    instructions = collect_instructions(instructions_path, volume_metadata, "extraction")
    
    for x, entry in enumerate(data["entries"]):
        info = extract_data_from_entry(entry, volume_metadata, examples, instructions)

        def load_extracted_data(info):
            try:
                json.loads(info)                
            except:
                return False
            
            return True

        while not load_extracted_data(info):
            info = extract_data_from_entry(entry, volume_metadata, examples, instructions)
        
        data["entries"][x]["data"] = json.loads(info)    
    
    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return data

def extract_data_from_entry(entry, volume_metadata, examples, instructions):
    """Extracts content from a single transcribed entry from a historical document.

    Args:
        entry (dict): entry record minimally including normalized entry text as
        the value assigned to the key `normalized`

        volume_metadata (dict): dict containing basic volume metadata

        examples (list): training data to be used to fine-tune model

        instructions (list): instructions to be passed to model as system messages

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

    return response.choices[0].message.content