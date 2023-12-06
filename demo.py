from openai import OpenAI
import os

client = OpenAI()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a historian of the early modern Atlantic with a large collection of transcriptions of related historical documents," \
                " mostly sacramental records from the Catholic Church. You want to learn more about the people, places, and events that appear in these documents."
        },
        {
            "role": "system",
            "content": "You will be provided with a series of transcriptions. Each of them will come with some descriptive metadata about the type of record that they are " \
                "and the historical context that they come from. For each transcription, you will be provided with the following information: the type of record that it is " \
                    "(usually a baptism, marriage, or burial register from the Catholic Church), when and where it was created, and a unique identifier that uniquely identifies the volume." \
                         " Next you will be provided with a sample transcription and detailed breakdown of desired output. Your job will be to produce a JSON file " \
                            "containing extracted information about people, places, and events for each transcription. " \
                                "You will be provided with additional information about the structure of that JSON file next."
        },
        {
            "role": "system",
            "content": "For each transcription, you will be provided with the following information: the type of record that it is " \
                "(usually a baptism, marriage, or burial register from the Catholic Church), when and where it was created, " \
                    "and a unique identifier that uniquely identifies the volume. Next you will be provided with a sample transcription and detailed breakdown of desired output."
        },
        {
            "role": "system",
            "content": 'Here is an example transcription: "En diezynuebe de Julio de mil ochocientoscuarenta; yo Don Antonio Loreto Sanchez Presbitero cura beneficiado por " \
                "Su Majestad de la Iglesia Parroquial de la Purisima Concepción de esta villa de Cienfuegos en ella y su jurisdiccion vicario Eclesiastico por " \
                    "Santa Eclesiastica Illustrisima bauticé solemnemente y puse los santos oleos a una negra adulta natural de Africa, de la propiedad de Don Juan Vives, " \
                        "en cuya negra ejerci las sacras ceremonias y preces y la puse por nombre Dominga fue su madrina Isabel aquien adverti el parentesco espiritual y demas " \
                            "obligaciones y lo firmé"'
        }
    ],
    model="gpt-3.5-turbo"
)

print(chat_completion)