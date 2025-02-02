"""Aggregates entry-level metadata into a document-level record.

Supports the natural language processing of entry-based historical records
by aggregating data from individual entries into a single, document-level
json file. Intermediate step between content extraction and integration
into a database.
"""

import json
from utility import parse_date, complete_date, disambiguate_people, merge_records

def aggregate_entry_records(path_to_entry_records, output_path=None):
    """Aggregates data from individual entries into document-level json file.
    
    Args:    
        path_to_entry_records: Either a path to a json file containing
        entry-level data to aggregate or a dictionary containing this data.

        output_path (str, optional): Path to output aggregated json to.
        Output will not be saved if this parameter is not included.

    Returns:
        A dict containing the aggregated data.
    """
    if type(path_to_entry_records) is str:
        with open(path_to_entry_records, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = path_to_entry_records

    #attempts to complete any missing dates
    #assumes that entries are in chronological order
    last_date = parse_date("1500-01-01")
    needs_date = []

    for x, entry in enumerate(data["entries"]):
        if not "events" in entry:
            continue
        
        for index, event in enumerate(entry["events"]):
            if "date" in event:
                date = parse_date(event["date"])
                if len(date) == 3:
                    if len(needs_date) != 0:
                        for i in needs_date:
                            data["entries"][i[0]]["events"][i[1]]["date"] = f"{last_date[0]}-{last_date[1]}-{last_date[2]}/{date[0]}-{date[1]}-{date[2]}"
                    needs_date = []
                    last_date = date                    
                else:
                    date = complete_date(date)
                    entry["events"]["index"]["date"] = f"{date[0]}-{date[1]}-{date[2]}/{date[3]}-{date[4]}={date[5]}"
            else:
                if event["type"] != "birth":
                    needs_date.append((x, index))

    #creates volume-level metadata
    volume = {}

    for key in ["type", "country", "state", "city", "institution", "id", "title"]:
        volume[key] = data[key]

    volume["entries"] = []
    volume["people"] = []    
    volume["events"] =[]

    #creates metadata about people and events 
    for entry in data["entries"]:        
        volume["entries"].append({"id": entry["id"], "text": entry["normalized"]})

        for key in ["people", "events"]:
            if key not in entry['data']:
                entry['data'][key] = ""        
        
        #adds entry info to each person id so that it is unique at the volume level
        for person in entry["data"]["people"]:
            person["id"] = f"{entry['id']}-{person['id']}"
            person["mentions"] = entry['id']
            if "relationships" in person:
                for rel in person["relationships"]:
                    #temporary fix to catch incorrectly structured model output
                    try:
                        rel["related_person"] = f"{entry['id']}-{rel['related_person']}"
                    except:
                        print(entry["id"])
                        if type(entry) == str:
                            print(f"Bad entry: {entry}")
                        elif type(rel) == str:
                            print(f"Bad relationship: {rel}")
            add = False
            #recursively disambiguates people who appear in the data
            for i, p in enumerate(volume["people"]):
                if (person["name"] == p["name"]) and disambiguate_people(person, p):
                    volume["people"][i] = merge_records(person, p)
                    add = True
            if not add:
                volume["people"].append(person)

        #adds entry info the person ids that appear in events, assigns unique ids to events
        if "events" not in entry["data"]:
            continue
        
        for index, event in enumerate(entry["data"]["events"]):
            if "principals" in event:
                for x, p in enumerate(event["principals"]):
                    event["principals"][x] =  f"{entry['id']}-{event['principals'][x]}"
            if "witnesses" in event:
                for x, w in enumerate(event["witnesses"]):
                    event["witnesses"][x] =  f"{entry['id']}-{event['witnesses'][x]}"
            event["id"] = f"{entry['id']}-E{'0' * (2 - len(str(index))) + str(index)}"
            volume["events"].append(event)

    #assigns volume-level unique id to each *disambiguated* person
    next_id = 1
    id_map = {}
    for person in volume["people"]:
        new_id = "P" + "0" * (5 - len(str(next_id))) + str(next_id)
        if type(person["id"]) == str:
            id_map[person["id"]] = f"{volume['id']}-{new_id}"            
        else:
            for id in person["id"]:
                id_map[id] = f"{volume['id']}-{new_id}"                
        next_id += 1

    """with open("people.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f)"""

    #applies volume-level unique ids to people records
    for i, person in enumerate(volume["people"]):
        if type(person["id"]) == str:
            volume["people"][i]["id"] = id_map[person["id"]]
        else:
            volume["people"][i]["id"] = id_map[person["id"][0]]
        if "relationships" in person:
            for j, rel in enumerate(person["relationships"]):
                #temporary fix to catch model overfitting to few-shot training
                try:
                    volume["people"][i]["relationships"][j]["related_person"] = id_map[volume["people"][i]["relationships"][j]["related_person"]]
                except:
                    print(f"Found bad related person identifier: {volume['people'][i]['relationships'][j]['related_person']}")

    #applies volume-level unique ids to event records
    for i, event in enumerate(volume["events"]):
        if len(event["principals"]) == 1:
            volume["events"][i]["principals"] = [id_map[volume["events"][i]["principals"][0]]]          
        elif len(event["principals"]) == 2:
            volume["events"][i]["principals"] = [id_map[volume["events"][i]["principals"][0]], id_map[volume["events"][i]["principals"][1]]]
        elif "witnesses" in event:
            for j in range(len(event["witnesses"])):
                volume["events"][i]["witnesses"][j] = id_map[volume["events"][i]["witnesses"][j]]

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(volume, f)

    return volume

#aggregate_entry_records("extracted.json", output_path="testing\\166470_lg_sample_agg.json")

        

