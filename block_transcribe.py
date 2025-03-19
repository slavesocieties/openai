from openai import OpenAI
from utility import *
from segmentation_driver import *
import json

def transcribe_block(id, instructions_path="instructions.json", examples_path="htr_training_data/239746_short_htr.json", bucket_name="ssda-openai-test", fine_tune=False):	
	volume_id = int(id.split("-")[0])
	block_id = id[id.find("-") + 1:]
	volume_metadata = load_volume_metadata(volume_id)
	instructions = collect_instructions(instructions_path, volume_metadata, "transcription")
	examples = generate_block_htr_training_data(examples_path)

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
                        {
							"type": "text",
							"text": f"Please transcribe the sacramental record {example['id']}, which appears in the attached images."
						},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": example["color"],
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": example["pooled"],
                                "detail": "high"
                            }
                        }
                    ]
			    }
			)
			output = {"id": example["id"], "lines": example["lines"]}
			conversation.append(
				{
					"role": "assistant",
					"content": json.dumps(output)
				}
			)

	conversation.append(
				{
                    "role": "user",
                    "content": [
                        {
							"type": "text",
							"text": f"Please transcribe the sacramental record {block_id}, which appears in the attached images."
						},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://{bucket_name}.s3.amazonaws.com/{id}-color.jpg",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://{bucket_name}.s3.amazonaws.com/{id}-pooled.jpg",
                                "detail": "high"
                            }
                        }
                    ]
			    }
			)
	if fine_tune:
		conversation = []

		for instruction in instructions:
			conversation.append(
				{
						"role": "system",
						"content": instruction["text"]
				}
			)
		conversation.append(
			{
                    "role": "user",
                    "content": [
                        {
							"type": "text",
							"text": f"Please transcribe the sacramental record {block_id}, which appears in the attached images."
						},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://{bucket_name}.s3.amazonaws.com/{id}-color.jpg",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://{bucket_name}.s3.amazonaws.com/{id}-pooled.jpg",
                                "detail": "high"
                            }
                        }
                    ]
			    }
		)
	
	try:
		if fine_tune:
			response = client.chat.completions.create(
				model= "ft:gpt-4o-2024-08-06:personal::B9yazDFQ",	
				messages = conversation
			)
			print(response.choices[0].message.content)
		else:
			response = client.chat.completions.create(
				model= "gpt-4o",	
				messages = conversation
			)
		print(f"{id} transcribed.")
	except:
		print(f"Failed to transcribe {id}.")
		return None

	return response.choices[0].message.content

def build_entry(gpt_output):
	try:
		gpt_output = json.loads(gpt_output)
	except:
		return False
	entry = {"id": gpt_output["id"]}
	text = ""
	for index, line in enumerate(gpt_output["lines"]):
		# TODO: investigate why this can occur
		if len(line) == 0:
			continue
		
		if line[0] == "-" and len(text) > 0:
			text = text[:len(text) - 1]
			text += line[1:]
		else:
			text += line
		if text[len(text) - 1] != "-" and index < (len(gpt_output["lines"]) - 1):
			text += " "
		elif text[len(text) - 1] == "-":
			text = text[:len(text) - 1]

	entry["raw"] = text

	return entry

def write_volume(volume_id, entries, output_path=None):
	volume_metadata = load_volume_metadata(volume_id)
	record = {"type": volume_metadata["type"], "id": volume_id}
	for key in ["country", "state", "city", "institution", "title"]:
		record[key] = volume_metadata["fields"][key]
	record["entries"] = []
	for entry in entries:
		record["entries"].append(entry)

	if output_path is None:
		output_path = f"json/{volume_id}.json"
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(record, f)	

"""output = transcribe_block("239746-0076-01")
entries = [build_entry(output)]
write_volume(239746, entries, output_path="testing/239746.json")"""
