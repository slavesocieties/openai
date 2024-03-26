from openai import OpenAI
from utility import *
from segmentation_driver import *
import json

def transcribe_block(id, instructions_path="instructions.json", examples_path="json/239746_htr_sample.json", bucket_name="ssda-openai-test"):	
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
	
	response = client.chat.completions.create(
		model="gpt-4-vision-preview",	
		messages = conversation
	)

	return response.choices[0].message.content

print(transcribe_block("239746-0076-01"))