"""Automatically extracts content from historical records.

Parses transcriptions into required format if necessary,
then normalizes each transcriptions, extracts content from
each transcription, and aggregates extracted content into a
single, volume-level record.
"""

from parse import parse_xml
from normalize import *
from extract import *
from aggregate import *

def process_transcription(volume_record_path, instructions_path, training_data_path, training_keywords = None, mode = "or", shots = 1000, parse = False, out_path = None, testing = False):
    """Automatically extracts content from historical records.
    
    Args:
        volume_record_path (str): path to an xml or json file containing a historical transcription

        instructions_path (str): path to a json file containing natural language instructions
        that will be passed to the llm as system messages

        training_data_path (str): path to a json file containing manually constructed examples
        of content extraction that will be used to train the llm

        training_keywords (list, optional): list of keywords that define a subset of training
        data to use in conjunction with the next parameter; if not included, all available
        training data will be used

        mode (str, optional): either `and` or `or`, defines subset of training data to use
        in conjunction with previous parameter

        shots (int, optional): defines maximum number of examples to include in conversation
        history supplied to llm

        parse (boolean, optional): true if input file is xml that needs to be parsed to json,
        false otherwise

        out_path (str, optional): path to output final volume record to; volume record will not
        be saved if this is not included

    Returns:
        The final volume record as a dict.
    """
    if parse:
        volume_record_path = parse_xml(volume_record_path)
        print("Parsing complete.")
        
    normalized = normalize_volume(volume_record_path, instructions_path, training_data_path, keywords = training_keywords, match_mode = mode, max_shots = shots)

    print("Normalization complete.")

    extracted = extract_data_from_volume(normalized, instructions_path, training_data_path, keywords = training_keywords, match_mode = mode, max_shots = shots)
    
    #temporary, to check quality of model output
    if testing:
        with open("extracted.json", "w", encoding="utf-8") as f:
            json.dump(extracted, f)

    print("Extraction complete.")

    volume_record = aggregate_entry_records(extracted, output_path=out_path)

    print("Volume record saved.")

    return volume_record

#process_transcription("testing\\166470_sample.json", "instructions.json", "training_data.json", training_keywords = {"type": "baptism", "country": "Cuba"}, mode = "and", out_path = "testing\\166470_sample_output.json")