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

from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open("fine_tune.jsonl", "rb"),
  purpose="fine-tune"
)

# file-UnYsSL8XTvRjFtWSciR7Eu

from openai import OpenAI

client = OpenAI()

"""response = client.files.create(
  file=open("fine_tune.jsonl", "rb"),
  purpose="fine-tune"
)"""

# print(client.files.list())

"""client.fine_tuning.jobs.create(
    training_file="file-UnYsSL8XTvRjFtWSciR7Eu",
    model="gpt-4o-2024-08-06"
)"""

# List 10 fine-tuning jobs
print(client.fine_tuning.jobs.list(limit=10))

# Retrieve the state of a fine-tune
# client.fine_tuning.jobs.retrieve("ftjob-abc123")

# Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# List up to 10 events from a fine-tuning job
# client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

# Delete a fine-tuned model
# client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")