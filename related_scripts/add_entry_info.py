import json
import os

def update_training_data(training_data_path, transcriptions_folder="json", output_path="updated_training_data.json"):
    # Load the training data
    with open(training_data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    for example in training_data["examples"]:
        volume_id = example["id"]
        transcription_file = os.path.join(transcriptions_folder, f"{volume_id}.json")
        
        if not os.path.exists(transcription_file):
            print(f"Transcription file for volume ID {volume_id} not found.")
            continue
        
        # Load the corresponding transcription file
        with open(transcription_file, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)
        
        found = False
        for entry in transcription_data.get("entries", []):
            if entry["raw"] == example["raw"]:
                # Insert "entry" key right after "id" while preserving the rest of the order
                new_example = {}
                for key, value in example.items():
                    new_example[key] = value
                    if key == "id":  # Insert "entry" immediately after "id"
                        new_example["entry"] = entry["id"]
                example.clear()
                example.update(new_example)  # Modify in place
                
                found = True
                break
        
        if not found:
            print(f"No matching entry found for example with volume ID {volume_id}.")
    
    # Save the updated training data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4, ensure_ascii=False)

    print(f"Updated training data saved to {output_path}")

# Example usage
update_training_data("training_data.json", "json", "updated_training_data.json")
