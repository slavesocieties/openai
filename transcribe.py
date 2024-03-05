from openai import OpenAI
from utility import *
from segmentation_driver import *
import json
import boto3
import os
import shutil

def transcribe_line(image_url, instructions, examples=None):
	client = OpenAI()

	conversation = []

	for instruction in instructions:
		conversation.append(
			{
					"role": "system",
					"content": instruction["text"]
			}
		)

	if examples is not None:
		for example in examples:
			conversation.append(
				{
			"role": "user",
			"content": [
				{"type": "text", "text": "Please transcribe this line."},
				{
				"type": "image_url",
				"image_url": {
					"url": example["url"],
					"detail": "high"
				}
				}
			]
			}
			)
			conversation.append(
				{
					"role": "assistant",
					"content": example["text"]
				}
			)

	conversation.append(
		{
		  "role": "user",
		  "content": [
			{"type": "text", "text": "Please transcribe this line."},
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

def transcribe_entry(image_urls, instructions, examples):
	entry_text = ""
	for url in image_urls:		
		"""try:
			entry_text += transcribe_line(url, instructions, examples) + "\n"
		except:
			print(url)			
			continue"""
		entry_text += transcribe_line(url, instructions, examples) + "\n"
	
	return entry_text

def transcribe_volume(volume_id, volume_metadata_path = "volumes.json", instructions_path = "instructions.json", source_bucket = "ssda-production-jpgs", image_bucket = "ssda-openai-test", transcription_bucket = "ssda-transcriptions", output_path = None):
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

	instructions = collect_instructions(instructions_path, volume_metadata, "transcription")
	examples = generate_htr_training_data(bucket_name="ssda-htr-training", metadata_path="volumes.json", keywords= {"identifier": volume_id}, match_mode="or", color=None, max_shots=10)
	
	for entry in entries:
		image_urls = []
		for line_id in range(entry["lines"]):
			padded_line = "0" * (2 - len(str(line_id + 1))) + str(line_id + 1)
			image_urls.append(f"https://{image_bucket}.s3.amazonaws.com/{entry['id']}-{padded_line}.jpg")
		entry["text"] = transcribe_entry(image_urls, instructions, examples)
		with open("temp.txt", "w", encoding="utf-8") as f:
			f.write(entry["text"])
		s3_client.upload_file("temp.txt", transcription_bucket, f"{entry['id']}.txt")
		entry["text"] = entry["text"].replace("\n", " ")
		
	os.unlink("temp.txt")

	print("Entries transcribed.")

	record = {}

	#TODO add support for additional record types
	if "type" in volume_metadata:
		record["type"] = volume_metadata["type"]	
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

#transcribe_volume(239746, volume_metadata_path = "demo.json", source_bucket = "ssda-openai-test")

"""volume_metadata = load_volume_metadata(239746, volume_metadata_path = "volumes.json")
instructions = collect_instructions("instructions.json", volume_metadata, "transcription")
examples = generate_htr_training_data(bucket_name="ssda-openai-test", metadata_path="volumes.json", keywords= {"identifier": 239746}, match_mode="or", color=None, max_shots=3)
print(transcribe_line("https://ssda-openai-test.s3.amazonaws.com/really_big.jpg", instructions))"""

"""urls = []
for x in range(1, 30):
	urls.append(f"https://ssda-openai-test.s3.amazonaws.com/239746-0001-01-{'0' * (2 - len(str(x))) + str(x)}.jpg")


volume_metadata = load_volume_metadata(239746, volume_metadata_path = "volumes.json")
instructions = collect_instructions("instructions.json", volume_metadata, "transcription")
examples = generate_htr_training_data(bucket_name="ssda-htr-training", metadata_path="volumes.json", keywords= {"identifier": 239746}, match_mode="or", color=None, max_shots=10)
print(transcribe_line("https://ssda-openai-test.s3.amazonaws.com/239746-0001-01-08.jpg", instructions, examples))"""
#print(transcribe_entry(urls, instructions, examples))
	