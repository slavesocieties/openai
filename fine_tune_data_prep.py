import json
from utility import *

with open("htr_training_data/239746_htr.json", "r", encoding='utf-8') as f:
    data = json.load(f)

volume_id = 239746
volume_metadata = load_volume_metadata(volume_id)
instructions = collect_instructions("instructions.json", volume_metadata, "transcription")
bucket_name="ssda-openai-test"
examples = [] 

for entry in data["entries"]:    
    conversation = []

    for instruction in instructions:
        conversation.append(
			{
					"role": "system",
					"content": instruction["text"]
			}
		)

    example = {"id": entry["id"], "lines": entry["lines"]}
    example["color"] = f"https://{bucket_name}.s3.amazonaws.com/{volume_id}-{entry['id']}-color.jpg"
    example["pooled"] = f"https://{bucket_name}.s3.amazonaws.com/{volume_id}-{entry['id']}-pooled.jpg"
    output = {"id": example["id"], "lines": example["lines"]}

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
    
    conversation.append(
				{
					"role": "assistant",
					"content": json.dumps(output)
				}
			)
    
    examples.append({"messages": conversation})

with open("fine_tune.jsonl", "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")