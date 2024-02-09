from openai import OpenAI
from utility import *
from segmentation_driver import *
import json
import boto3
import os
import shutil
import numpy as np

#function that constructs training data
#create a simple list of metadata for all volumes that we have htr training data for, use keyword matching to select subset of examples to be used for training
#random sample can be constructed as before
#can include instructions in same file as those for nlp

def transcribe_line(image_url):
	client = OpenAI()

	conversation = []

	conversation.append(
		{
		  "role": "user",
		  "content": [
			{"type": "text", "text": "Please use the OpenAI Vision System to manually transcribe this image of a line from an early modern Spanish baptismal register. Your response should only include the transcribed text."},
			{
			  "type": "image_url",
			  "image_url": {
				"url": image_url,
				"detail": "high"
			  }
			}
		  ]
		}
	)
	
	response = client.chat.completions.create(
		model="gpt-4-vision-preview",	
		messages = conversation
	)

	return response.choices[0].message.content

def transcribe_entry(image_urls):
	entry_text = ""
	for url in image_urls:
		#print(url)
		try:
			entry_text += transcribe_line(url) + "\n"
		except:
			print(url)			
			continue
	
	return entry_text

def transcribe_volume(volume_id, volume_metadata_path = "volumes.json", source_bucket = "ssda-production-jpgs", image_bucket = "ssda-openai-test", transcription_bucket = "ssda-transcriptions", output_path = None):
	#remember to save AWS credentials as environmental variables
	s3_client = boto3.client('s3')  
	volume_metadata = load_volume_metadata(volume_id, volume_metadata_path = volume_metadata_path)

	if volume_metadata is None:
		print("No metadata found for that volume id.")
		return
	
	entries = []
	
	for image in range(volume_metadata["fields"]["images"]):
		#download image from S3
		im_id = "0" * (4 - len(str(image + 1))) + str(image + 1)
		s3_client.download_file(source_bucket, f"{volume_metadata['id']}-{im_id}.jpg", f"{volume_metadata['id']}-{im_id}.jpg")
		#segment locally with modified algorithm
		counts = segmentation_driver(f"{volume_metadata['id']}-{im_id}.jpg")

		os.unlink(f"{volume_metadata['id']}-{im_id}.jpg")
		#append a record to entries containing each entry id and number of lines of entry		
		for e, count in enumerate(counts):
			ent_id = "0" * (2 - len(str(e + 1))) + str(e + 1)
			entries.append({"id": f"{volume_metadata['id']}-{im_id}-{ent_id}", "lines": count})
	
	print("Images segmented.")

	for folder, subfolders, files in os.walk("segmented"):		
		for file in files:		
			s3_client.upload_file(os.path.join(folder, file), image_bucket, file, ExtraArgs={'ContentType': 'image/jpeg'})

	print("Image segments uploaded.")

	shutil.rmtree("segmented")
	
	for entry in entries:
		image_urls = []
		for line_id in range(entry["lines"]):
			padded_line = "0" * (2 - len(str(line_id + 1))) + str(line_id + 1)
			image_urls.append(f"https://{image_bucket}.s3.amazonaws.com/{entry['id']}-{padded_line}.jpg")
		entry["text"] = transcribe_entry(image_urls)
		with open("temp.txt", "w", encoding="utf-8") as f:
			f.write(entry["text"])
		s3_client.upload_file("temp.txt", transcription_bucket, f"entry['id'].txt")
		entry["text"] = entry["text"].replace("\n", " ")
		
	os.unlink("temp.txt")

	print("Entries transcribed.")

	record = {}

	#TODO add support for additional record types
	if "Baptisms" in volume_metadata["fields"]["subject"]:
		record["type"] = "baptism"
	elif "Marriages" in volume_metadata["fields"]["subject"]:
		record["type"] = "marriage"
	elif "Burials" in volume_metadata["fields"]["subject"]:
		record["type"] = "burial"
	else:
		record["type"] = "other"

	for key in ["country", "state", "city", "institution", "identifier", "title"]:
		record[key] = volume_metadata["fields"][key]

	record["entries"] = []
	
	for entry in entries:
		record["entries"].append({"id": entry["id"], "raw": entry["text"]})

	if output_path is None:
		output_path = f"{record['identifier']}.json"

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(record, f)

	return record	

#transcribe_volume(740005, volume_metadata_path = "demo.json", source_bucket = "ssda-openai-test")

#print(transcribe_line("https://ssda-openai-test.s3.amazonaws.com/239746-0037-02-08.jpg"))

"""urls = []
for x in range(10):
	urls.append(f"https://ssda-openai-test.s3.amazonaws.com/239746-0037-02-{'0' * (2 - len(str(x))) + str(x)}.jpg")
print(transcribe_entry(urls))"""