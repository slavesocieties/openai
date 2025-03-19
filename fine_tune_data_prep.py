# fine_tune_updated.jsonl ft:gpt-4o-2024-08-06:personal::B4xSaBY4
# 3/11/25 fine_tune_239746_full.jsonl ft:gpt-4o-2024-08-06:personal::B9yazDFQ

'''import json
from utility import *

with open("htr_training_data/239746_full_htr.json", "r", encoding='utf-8') as f:
    data = json.load(f)

volume_id = 239746
volume_metadata = load_volume_metadata(volume_id)
instructions = collect_instructions("instructions.json", volume_metadata, "transcription")
bucket_name="ssda-fine-tuning"
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
					"content": json.dumps(output, ensure_ascii=False)
				}
			)
    
    examples.append({"messages": conversation})

with open("fine_tune_239746_full.jsonl", "w", encoding="utf-8") as f:
    for example in examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")'''

from openai import OpenAI
client = OpenAI()

'''response = client.files.create(
  file=open("fine_tune_239746_full.jsonl", "rb"),
  purpose="fine-tune"
)

print(client.files.list())'''

# fine_tune_239746_full.jsonl 3/6/25 file-UKXN7Y8NyHHHk1YBXhVzhQ
# fine_tune_239746_sample.jsonl 3/11/25 file-FMUtFWrUUNWttRbTPioK3E
# fine_tune_239746_full.jsonl 3/11/25 file-Th9EedvgSjCn5hPht5dLVW

'''client.fine_tuning.jobs.create(
    training_file="file-Th9EedvgSjCn5hPht5dLVW",
    model="gpt-4o-2024-08-06"
)'''

# List 10 fine-tuning jobs
# print(client.fine_tuning.jobs.list(limit=1))

# file-Th9EedvgSjCn5hPht5dLVW 3/11/25 ftjob-2JPYzqlfFPEJD3QWVmSUSy4u
# Retrieve the state of a fine-tune
print(client.fine_tuning.jobs.retrieve("ftjob-2JPYzqlfFPEJD3QWVmSUSy4u"))

# Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-XpJNsTlUNPirodFycbQPBF9I")

# List up to 10 events from a fine-tuning job
# client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

# Delete a fine-tuned model
# client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")