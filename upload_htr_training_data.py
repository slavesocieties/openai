import os

def driver(path_to_data):
    images = []
    for folder, subfolders, files in os.walk(path_to_data):       
        if ("logs.csv" not in files) or ("text.txt" not in files) or (folder == path_to_data):
            print(f"{folder} missing either logs or text")
        else:
            image_lines = parse_logs(os.path.join(path_to_data, folder, "logs.csv"))           
            image_lines = parse_text(os.path.join(path_to_data, folder, "text.txt"))
            #check color scheme of first two images, if they don't match assume gray
            #loop through images, upload to s3, upload text to s3
            #write coords and color scheme to log in s3            

def parse_logs(logs_path):
    lines = []
    with open(logs_path, "r", encoding="utf-8") as f:
        first = True                
        for line in f:
            if first:
                first = False
                continue
            data = line[line.rfind('"') + 2:].split(",")
            id = data[5]
            coords = data[6].replace("\n", "").split(" ")
            lines.append({"id": id, "coords": coords})

    return lines

def parse_text(text_path, lines):
    with open(text_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            lines[index]["text"] = line.replace("\n", "")

    return lines