def normalize_volume(volume_record_path, output_path = None):
    import json
    
    with open(volume_record_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data["type"] == "baptism":
        record_type = "baptismal register"

    if data["country"] in ["Cuba", "Colombia"]:
        language = "Spanish"

    for x, entry in enumerate(data["entries"]):
        norm = normalize_entry(entry, record_type, language)
        data["entries"][x]["normalized"] = norm

    if output_path == None:
        output_path = volume_record_path
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def normalize_entry(entry, record_type, language):
    from openai import OpenAI
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",    
        messages = [
            {
                "role": "system",
                "content": "You are assisting a historian of the early modern Atlantic with a large collection of transcriptions of Catholic sacramental records written in Spanish. " \
                "The historian will provide you with a transcription of a sacramental record written in early modern Spanish and ask you to normalize it by expanding abbreviations, " \
                "correcting idiosyncratic or archaic spellings, modernizing capitalization and punctuation, and correcting obvious transcription errors."
            },
            {
                "role": "system",
                "content": "Expand abbreviated names as well as words. Commonly abbreviated first names include Antonio or Antonia, Domingo or Dominga, Francisco or Francisca, " \
                "or Juan or Juana. Commonly abbreviated last names include Fernandez, Gonzalez, Hernandez, or Rodriguez. These are not intended to be complete lists. Use context " \
                f"to determine when a name has been abbreviated and your knowledge of {language} names to determine what the abbreviated name is."
            },
            {
                "role": "user",
                "content": "Please normalize this transcription of a Spanish baptismal register: `" + "Domingo veinte y dos de febrero yo Thomas de Orvera baptize, y pusse s.tos oleos " \
                "a Juana de nacion Mina esclava de Juan Joseph de Justis fueron sus P.P. Joseph Salcedo y Ana de Santiago su mugger, y lo firmé." + "`"
            },
            {
                "role": "assistant",
                "content": "Domingo veintidós de febrero yo Thomas de Orvera bauticé y puse santos óleos a Juana de nacion Mina, esclava de Juan Joseph de Justis. Fueron sus padrinos " \
                "Joseph Salcedo y Ana de Santiago su mujer, y lo firmé."
            },
            {
                "role": "user",
                "content": "Please normalize this transcription of a Spanish baptismal register: `" + "Juebes veinte y tres de feb.o de mil sietec.tos. y diez y nueve Yo Thomas de Orvera baptizé, " \
                "y pusse los santos15 oleos á Paula h. l. de Juan Joseph, y Maria Josepha esc.s del Capitan D. Luis Hurtado de Mendoza fue su Padrino Bartholome Rixo, y lo firmé." + "`"
            },
            {
                "role": "assistant",
                "content": "Jueves veintitrés de febrero de mil setecientos diecinueve yo Thomas de Orvera bauticé " \
                "y puse los santos óleos a Paula hija legítima de Juan Joseph y Maria Josepha, esclavos del Capitan Don Luis Hurtado de Mendoza. Fue su padrino Bartholome Rixo, y lo firmé."
            },
            {
                "role": "user",
                "content": f"Please normalize this transcription of a {language} {record_type}: `" + entry["raw"] + "`"
            }
        ]
    )

    return response.choices[0].message.content