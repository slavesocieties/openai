import os
from utility import check_binarized
import boto3
import json

def driver(path_to_data, bucket="ssda-htr-training", verbose=False):
    # remember to update credentials
    s3_client = boto3.client('s3')
    images = []
    for folder, subfolders, files in os.walk(path_to_data):       
        if (folder == path_to_data) or (".git" in folder):
            continue
        elif ("logs.csv" not in files) or ("text.txt" not in files):
            print(f"{folder} missing either logs or text")
        else:
            if verbose:
                print(f"Now working on {folder}.")            
            image_lines = parse_logs(os.path.join(folder, "logs.csv"))                      
            image_lines = parse_text(os.path.join(folder, "text.txt"), image_lines)
            if len(image_lines) == 1:
                image_lines[0]["color"] = check_binarized(os.path.join(folder, files[0]))
            else:                    
                checks = [check_binarized(os.path.join(folder, files[0])), check_binarized(os.path.join(folder, files[1]))]
                if checks[0] == checks[1]:
                    for line in image_lines:
                        line["color"] = checks[0]
                else:
                    for line in image_lines:
                        line["color"] = "gray"
            files.sort()            
            for i, file in enumerate(files):
                if "jpg" not in file:
                    continue
                s3_client.upload_file(os.path.join(folder, file), bucket, f"{image_lines[i]['id']}.jpg", ExtraArgs={'ContentType': 'image/jpeg'})
            for line in image_lines:
                images.append(line)
    update_training_data(images, bucket, s3_client)
    return len(images)

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

#print(parse_logs("htr_upload_test\\8186-0040\\logs.csv"))

def parse_text(text_path, lines):
    with open(text_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line =  line.replace("\n", "").replace("\t", " ")
            while line.find("  ") != -1:
                line = line.replace("  ", " ")
            lines[i]["text"] = line

    return lines

#print(parse_text("htr_upload_test\\8186-0040\\text.txt", parse_logs("htr_upload_test\\8186-0040\\logs.csv")))

def update_training_data(images, bucket, s3_client, verbose=True):
    s3_client.download_file(bucket, "ssda-htr-training-data.json", "ssda-htr-training-data.json")
    with open("ssda-htr-training-data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for image in images:
        data["images"].append(image)
    if verbose:
        print(f"{len(data['images'])} training lines now indexed.")
    with open("ssda-htr-training-data.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    s3_client.upload_file("ssda-htr-training-data.json", bucket, "ssda-htr-training-data.json", ExtraArgs={'ContentType': 'application/json'})
    os.unlink("ssda-htr-training-data.json")

#print(f"{driver(path)} new lines processed and uploaded.")
