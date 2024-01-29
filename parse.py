"""Transforms xml representations of document transcriptions into json.

Supports the natural language processing of entry-based historical records
by transforming xml representations of document transcriptions into json,
which is required by downstream components of nlp pipeline.
"""

import json

def parse_xml(xml_path, output_path = None):
    """Parses standard SSDA xml into json.

    Args:
        xml_path (str): path to an xml representation of a volume transcription

        output_path (str, optional): path to output parsed json to; json will not
        be saved if this is not included

    Returns:
        Dict containing parsed xml record. 
    """
    images = 0
    volume = {}    
    in_entry = False
    in_partial = False
    btwn = False
    first_partial_close = False    
    entry = ""
    with open(xml_path, "r", encoding="utf-8") as f:        
        for line in f:
            #generates volume-level metadata
            if "<volume" in line:
                for key in ["type", "country", "state", "city", "institution", "id", "title"]:
                    start = line.find(key + "=")                    
                    volume[key] = line[start + len(key) + 2:line.find('"', start + len(key) + 2)]                                      
                    if key == "id":
                        volume[key] = int(volume[key])
                volume["entries"] = []
                continue
            #text pre-processing            
            strip = ["[margin]", "[roto]", "[signed]", "<margin>", "</margin>", "&lt;insert&gt;", "&lt;/insert&gt;", "&lt;blurry&gt;", "&lt;/blurry&gt;"]
            for x in strip:
                line = line.replace(x, "")
            if not "<" in line:
                punc = ["[", "]", "{", "}", "="]
                for x in punc:
                    line = line.replace(x, "")
            while line.find("  ") != -1:
                line = line.replace("  ", " ")
            if line[0] == " ":
                line = line[1:]
            #in-volume xml parsing 
            if "<entry" in line:
                in_entry = True
            elif "<image" in line:
                images += 1
                start = line.find("id")                
                image = int(line[start + len("id") + 2:line.find('"', start + len("id") + 2)])
                #fix for image numbers that were padded with a leading 1
                if (image > 1000) and (images < 500):
                    image = image % 1000
                if in_partial:
                    last_ent = entry_number
                entry_number = 1           
            elif "</entry" in line:
                in_entry = False
                while entry.find("  ") != -1:
                    entry = entry.replace("  ", " ")
                if len(entry) < 1:
                    entry = ""                    
                    continue
                if entry[-1] == " ":
                    entry = entry[:len(entry) - 1]
                num = "0" * (2 - len(str(entry_number))) + str(entry_number)
                im = "0" * (4 - len(str(image))) + str(image)
                volume["entries"].append({"id": f"{im}-{num}", "raw": entry})
                entry_number += 1
                entry = ""
            #combines entries that cross between images on the first image                
            elif "<partial id" in line:
                in_partial = True
            elif ("</partial" in line) and (not first_partial_close):
                first_partial_close = True
                btwn = True
            elif "</partial" in line:
                in_partial = False
                first_partial_close = False
                while entry.find("  ") != -1:
                    entry = entry.replace("  ", " ")
                if len(entry) < 1:
                    entry = ""                    
                    continue
                if entry[-1] == " ":
                    entry = entry[:len(entry) - 1]
                num = "0" * (2 - len(str(last_ent))) + str(last_ent)
                im = "0" * (4 - len(str(image - 1))) + str(image - 1)
                volume["entries"].append({"id": f"{im}-{num}", "raw": entry})                
                entry = "" 
            elif "<partial" in line:
                btwn = False          
            elif in_entry or (in_partial and (not btwn)):
                #logic to determine whether text is marginal notation
                if len(line) < 10:
                    continue
                elif len(line) < 30:
                    caps = 1
                    no_caps = 0
                    for word in line.split(" "):
                        if word.lower() == word:
                            no_caps += 1
                        else:
                            caps += 1                    
                    if caps >= no_caps:
                        continue
                if line[-2] == "-":                    
                    entry += line[:len(line) - 2]                    
                else:                                        
                    entry += line[:len(line) - 1] + " "

    if output_path != None:        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(volume, f)   

    return volume

def parse_nbu(txt_path, output_path = None):   
    """Parses Native Bound-Unbound output into json.

    Args:
        txt_path (str): path to a text representation of a volume transcription

        output_path (str, optional): path to output parsed json to; json will not
        be saved if this is not included

    Returns:
        Dict containing parsed txt record. 
    """ 
    volume = {}
    #volume-level metadata should be included in input if this ever enters production
    volume["type"] = "baptism"
    volume["country"] = "United States"
    volume["city"] = "Isleta Pueblo"
    volume["institution"] = "San Agustín de la Isleta"       
    entry = ""
    in_entry = False
    partial = False    
    with open(txt_path, "r", encoding="utf-8") as f:        
        for x, line in enumerate(f):
            if len(line) < 2:
                continue
            if x == 0:
                tokens = line.split("_")
                volume["id"] = int(tokens[1])
                volume["title"] = " ".join(tokens[2:len(tokens) - 1])            
                volume["entries"] = []
            #text pre-processing
            strip = ["ILL","(", ")", "[", "]", "*", "^"]
            for x in strip:
                line = line.replace(x, "")
            #in-volume text parsing           
            if "{left margin" in line:
                in_entry = True                
            elif ("{rubric" in line) and in_entry:
                in_entry = False
                if entry[-1] == " ":
                    entry = entry[:len(entry) - 1]
                if partial:
                    volume["entries"].append({"id": partial_im + partial_id, "raw": entry})
                    partial = False
                else:        
                    volume["entries"].append({"id": image + entry_ids[entry_index], "raw": entry})
                entry_index += 1
                entry = ""
            #combines entries that cross between images on the first image
            elif "IMG" in line:
                if in_entry:
                    partial = True
                    partial_id = entry_ids[entry_index]
                    partial_im = image
                tokens = line.replace(" ", "").replace("\n", "").split("_")
                image = tokens[3]                
                entry_ids = tokens[4:]
                entry_index = 0         
            elif in_entry:                
                if line[-2] == "-":                    
                    entry += line[:len(line) - 2]                    
                else:                                        
                    entry += line[:len(line) - 1] + " "

    if output_path != None:        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(volume, f)   

    return volume

def parse_jorge(txt_path, output_path = None):
    """Parses Jorge Delgadillo's transcriptions into json.

    Args:
        txt_path (str): path to a text representation of a volume transcription

        output_path (str, optional): path to output parsed json to; json will not
        be saved if this is not included

    Returns:
        Dict containing parsed txt record. 
    """ 
    volume = {}
    #volume-level metadata should be included in input if this ever enters production
    volume["type"] = "marriage"
    volume["country"] = "Cuba"
    volume["state"] = "La Habana"
    volume["city"] = "Havana"
    volume["institution"] = "Catedral de la Habana"
    volume["id"] = 6517
    volume["title"] = "Libro de Barajas: Matrimonios, 1584-1622"
    volume["entries"] = []
    entry = ""
    in_entry = False
    image = 1
    entry_number = 1

    with open(txt_path, "r", encoding="utf-8") as f:        
        for x, line in enumerate(f):            
            if "upper margin" in line:
                continue
            if len(line) < 2:
                continue
            #text pre-processing           
            strip = ["(", ")", "[ilegible]", "[roto]", "?", "¿"]
            for x in strip:
                line = line.replace(x, "")           
            #in-volume text parsing
            if "[left margin" in line:
                in_entry = True
                if len(line) > 100:
                    entry += line[line.find("margin") + 7:]               
            elif ("[signed]" in line) and in_entry:
                if len(line) > 50:
                    line = line[:line.find("[signed]")]
                    entry += line
                in_entry = False
                if len(entry) < 1:
                    print(entry_number)
                    print(image)
                    return               
                if entry[-1] == " ":
                    entry = entry[:len(entry) - 1]
                im = "0" * (4 - len(str(image))) + str(image)                      
                num = "0" * (2 - len(str(entry_number))) + str(entry_number)
                entry = entry.replace("[", "").replace("]", "")
                entry = entry.replace("\n", " ")
                while entry.find("  ") != -1:
                    entry = entry.replace("  ", " ")
                while entry[len(entry) - 1] in [" ", "–"]:
                    entry = entry[:len(entry) - 1]
                while entry[0] in [" ", ":"]:
                    entry = entry[1:]
                volume["entries"].append({"id": f"{im}-{num}", "raw": entry})
                entry_number += 1
                entry = "" 
            elif "[f." in line:               
                image += 1
                entry_number = 1 
            elif in_entry:
                if len(line) < 2:
                    continue                
                if line[-2] == "-":                    
                    entry += line[:len(line) - 2]                    
                else:                                        
                    entry += line[:len(line) - 1] + " "

    if output_path != None:        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(volume, f)

    return volume

#parse_nbu("txt\\FHL_007548705.txt", output_path = "json\\FHL_007548705.json")

#id = 239746
#parse_xml(f"xml\\{id}.xml", output_path = f"json\\{id}.json")

#parse_jorge("txt\\6517_clean.txt", output_path = "json\\6517.json")