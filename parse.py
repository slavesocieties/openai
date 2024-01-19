import json

def parse_xml(xml_path, output_path = None):
    images = 0
    volume = {}    
    in_entry = False
    in_partial = False
    btwn = False
    first_partial_close = False    
    entry = ""
    with open(xml_path, "r", encoding="utf-8") as f:        
        for line in f:
            if "<volume" in line:
                for key in ["type", "country", "state", "city", "institution", "id", "title"]:
                    start = line.find(key)                    
                    volume[key] = line[start + len(key) + 2:line.find('"', start + len(key) + 2)]                    
                    if key == "id":
                        volume[key] = int(volume[key])
                volume["entries"] = []            
            strip = ["[margin]", "[roto]", "[signed]", "<margin>", "</margin>"]
            for x in strip:
                line = line.replace(x, "")
            punc = ["[", "]", "{", "}", "="]
            for x in punc:
                line = line.replace(x, "")
            while line.find("  ") != -1:
                line = line.replace("  ", " ")
            if line[0] == " ":
                line = line[1:]
            if "<entry" in line:
                in_entry = True
            elif "<image" in line:
                images += 1
                start = line.find("id")
                image = int(line[start + len("id") + 2:line.find('"', start + len("id") + 2)])
                if (image > 1000) and (images < 500):
                    image = image % 1000
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
                num = "0" * (2 - len(str(entry_number))) + str(entry_number)
                volume["entries"].append({"id": f"{image}-{num}", "raw": entry})
                entry_number += 1
                entry = "" 
            elif "<partial" in line:
                btwn = False          
            elif in_entry or (in_partial and (not btwn)):
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