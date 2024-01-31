from openai import OpenAI

#function that constructs training data
#create a simple list of metadata for all volumes that we have htr training data for, use keyword matching to select subset of examples to be used for training
#random sample can be constructed as before
#can include instructions in same file as those for nlp
#function that transcribes a single line
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

print(transcribe_line("https://ssda-openai-test.s3.amazonaws.com/two.jpg"))