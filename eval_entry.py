from openai import OpenAI
import json

def check_entry(entry):
    client = OpenAI()
    conversation = []

    conversation.append(
        {
                "role": "system",
                "content": "You are assisting a historian who is transcribing early modern Spanish sacramental records. They will ask you whether a specified transcription contains a complete record or not."
        }
    )
	
    conversation.append(
        {
                "role": "system",
                "content": "Please respond strictly with one of these options: 'complete', 'beginning', or 'end'. Note that in this context a record is still considered 'complete' if a concluding signature and/or some part of the boilerplate language that typically concludes these sorts of records is missing."
        }
    )
	
    conversation.append(
        {
                "role": "system",
                "content": "Baptismal registers will always begin with the date and the affiliation of the priest who performed the baptism (if a baptismal register starts with a date, it can only be 'complete' or 'beginning'). They should also include, at a minimum, the name of the person being baptized."
        }
    )
	
    conversation.append(
        {
                "role": "system",
                "content": "Those pieces of information will frequently be followed by information about the parents of the person being baptized and the god-parents of the person being baptized, again in that order."
        }
    )
    
    conversation.append(
        {
                "role": "user",
                "content": f"This is a transcription of an early modern sacramental record: {entry['raw']} Is this a complete record? If not, is it the beginning or the end of a record?"
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o",    
        messages = conversation
    )

    return response.choices[0].message.content

'''with open("htr_training_data/entry_eval.json", "r", encoding="utf-8") as f:
	data = json.load(f)

for entry in data["entries"]:
	print(check_entry(entry))'''
