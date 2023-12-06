from openai import OpenAI
import os
from time import sleep

client = OpenAI()

"""file = client.files.create(
    file=open("example.json", "rb"),
    purpose='assistants'
)"""

print("File added.")

"""assistant = client.beta.assistants.create(
  instructions="You are a historian of the early modern Atlantic. Use the examples provided to you to extract information from transcriptions similar to those that appear as the value associated with the 'text' key in the examples.",
  model="gpt-3.5-turbo-1106",
  tools=[{"type": "retrieval"}],
  file_ids=["file-AlSFbtTbkDE0VeoVoT89aNo4"]
)"""

print("Assistant created.")

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I want to extract information about people, places, and events from the following transcription: 'En diezynuebe de Julio de mil ochocientoscuarenta; yo Don Antonio Loreto Sanchez Presbitero cura beneficiado por Su Majestad de la Iglesia Parroquial de la Purisima Concepción de esta villa de Cienfuegos en ella y su jurisdiccion vicario Eclesiastico por Santa Eclesiastica Illustrisima bauticé solemnemente y puse los santos oleos a una negra adulta de nación de Africa, de la propiedad de Don Juan Vives, en cuya negra ejerci las sacras ceremonias y preces y la puse por nombre Paula fue su madrina Ysabel aquien adverti el parentesco espiritual y demas obligaciones y lo firmé.' The ID of this transcription is 166470-0031-08. Please provided me with a JSON representation of extracted information about people, places, and events."
)

print("Thread created and message added.")

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id="asst_8F5GZ0ZAIjlU9ejqzP6iws8Y",
  instructions="Please address the user as Blackbeard. The user has a premium account."
)

print("Run has begun.")

run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
)

while run.status != 'completed':    
    if run.status in ['cancelled', 'failed', 'expired']:
        print(f"Run failed to complete with status {run.status}.")
        break
    print(f"The current status of the run is {run.status}.")
    sleep(10)
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

print("Run completed!")

messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

with open("output.txt", "w") as out:
    for message in messages:
        out.write(message)