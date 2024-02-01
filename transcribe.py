from openai import OpenAI
from utility import *
import json

#function that constructs training data
#create a simple list of metadata for all volumes that we have htr training data for, use keyword matching to select subset of examples to be used for training
#random sample can be constructed as before
#can include instructions in same file as those for nlp
#function that transcribes a single entry
#function that transcribes a full volume and creates record for nlp

#actually let's build this first without any training data or detailed instructions at all...

def transcribe_line(image_url):
    client = OpenAI()

    conversation = []

    conversation.append(
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Please use the OpenAI Vision System to manually transcribe this image. Your response should only include the transcribed text."},
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
        entry_text += transcribe_line(url) + "\n"
    
    return entry_text

def transcribe_volume(volume_id, volume_metadata_path = "volumes.json", image_bucket = "ssda-openai-test"):   
    volume_metadata = load_volume_metadata(volume_id, volume_metadata_path)

    if volume_metadata is None:
        print("No metadata found for that volume id.")
        return
    
    entries = []
    
    for image in range(volume_metadata["fields"]["images"]):
        #segment locally with modified algorithm?
        #append a record to entries containing entry id and number of lines
        entries.append("dict containing image id and number of lines")
    
    for entry in entries:
        image_urls = []
        for line_id in range(entry["lines"]):
            padded_line = "0" * (2 - len(str(line_id + 1))) + str(line_id + 1)
            image_urls.append(f"https://{image_bucket}.s3.amazonaws.com/{entry['id']}-{padded_line}.jpg")
            entry["text"] = transcribe_entry(image_urls)

  #save transcribed text with linebreaks somewhere
  #get rid of linebreaks and prep/save as volume record for nlp input



#print(transcribe_line("https://ssda-openai-test.s3.amazonaws.com/two.jpg"))