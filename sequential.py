from openai import OpenAI
client = OpenAI()

example_01 = "En diezynuebe de Julio de mil ochocientoscuarenta; yo Don Antonio Loreto Sanchez Presbitero cura beneficiado por " \
                "Su Majestad de la Iglesia Parroquial de la Purisima Concepción de esta villa de Cienfuegos en ella y su jurisdiccion vicario Eclesiastico por " \
                    "Santa Eclesiastica Illustrisima bauticé solemnemente y puse los santos oleos a una negra Carabalí adulta natural de Africa, de la propiedad de Capitán Juan Vives, " \
                        "en cuya negra ejerci las sacras ceremonias y preces y la puse por nombre Dominga fue su madrina Isabel aquien adverti el parentesco espiritual y demas " \
                            "obligaciones y lo firmé"

output_01 = ""

with open("example_01.json","r",encoding="utf-8") as f:
    for line in f:
        output_01 += line

example_02 = "En diezyseis de Octubre de mil ochocientostreintaynuebe; yo Don Antonio Loreto Sanchez Presbitero cura beneficiado " \
                         "por Santa Majestad de la Iglesia Parroquial de la Purisima Concepción de esta villa de Cienfuegos en ella y su jurisdiccion " \
                         "vicario Eclesiastico Santa Eclesiastica Illustrisima bautice solemnemente y puse los santos oleos a una parvula que nació el trece de Junio " \
                         "del año de mil ochocientostreintaysiete hija legitima de los morenos Julian y Sabina, naturales de Guinea esclavos " \
                         "de Don Jose Corras: en cuya parvula ejervi las sacras ceremonias y preces y le puse por nombre Antonia " \
                         "fue su madrina Budesiceda aquien adverti el parentesco espiritual y demas obligaciones y lo firme"

output_02 = ""

with open("example_02.json","r",encoding="utf-8") as f:
    for line in f:
        output_02 += line

example_03 = "En diezyseis de Octubre de mil ochocientostreintaynuebe; yo Don Juan Gomes Presbitero cura beneficiado por Su Majestad de " \
                "la Iglesia Parroquial de la Purisima Concepción de esta villa de Cienfuegos en ella y su jurisdiccion vicario Eclesiastico por Santa Eclesiastica Illustrisima bautice " \
                "solemnemente y puse los santos oleos a una parvula que nacio el veinteycuatro de Septiembre del año pasado hija de " \
                "la negra Jucina de la propiedad de Teniente Jose Corras; en cuya parvula libre ejerci las " \
                "sacras preces y ceremonias y preces y le puse por nombre Merced fue su madrina la parda Doña Maria Ramona Prieto; " \
                "aquien adverti el parentesco espiritual y demas obligaciones y lo firme"

output_03 = ""

with open("example_03.json","r",encoding="utf-8") as f:
    for line in f:
        output_03 += line

user_input = "En diezysies de Octubre de mil ochocientostreintaynuebe; yo Don Antonio Loreto Sanchez Presbitero cura beneficiado por " \
             "Su Majestad de la Iglesia Parroquial de la Purisima Concepción de esta villa de Cienfuegos en ella y su jurisdiccion vicario " \
             "Eclesiastico por Santa Eclesiastica Illustrisima bautice solemnemente y puse los santos oleos a una parvula que nacio, el quince de Julio " \
             "del presente año hija legitima de los morenos Fabian y de Isabel naturales de Africa de la propiedad de Don Jose " \
             "Corras; en cuya parvula ejerci las sacras ceremonias y preces y le puse por nombre Maria del Carmen; " \
             "fue su madrina Francisca Maria Herrera; aquien adverti el parentesco espiritual y demas obligaciones y lo firme"

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={ "type": "json_object" },
    messages = [
        {
            "role": "system",
            "content": "You are assisting a historian of the early modern Atlantic with a large collection of transcriptions of Catholic sacramental records written in Spanish. " \
            "The historian will provide you with a transcription of a sacramental record and ask you to extract detailed information from that record. Output this information as JSON. "
        },
        {
            "role": "system",        
            "content": "These transcriptions contain information about people. You will create a record for each person, containing at minimum their name as it appears in the transcription. " \
            "Assign each person a unique identifier. These should take the form P##, where ## is a two-digit padded integer and P01 is assigned to the first person to appear in the transcription."            
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about honorific titles or military ranks earned by these people. When they do, please include this information in your output for the person in question. " \
            "Examples of honorific titles include (but are not limited to): Don, Doña, Doctor, Licenciado, or a variety of ecclesiastical titles such as Frai, Padre, or Hermano. List each title exactly as it appears and list multiple titles separately."
            "Examples of military ranks include (but are not limited to): Capitán, Sargento Mayor, or Alférez. List these ranks as their English equivalents (respectively for the listed examples: Captain, Sergeant Major, and Ensign)."
        },             
        {
            "role": "system",
            "content": "These records will sometimes contain information about the places of origin of these people. When they do, please include this information in your output for the person in question." \
            "In Spanish, the presence of this information will often be signalled by words such as `natural` for a single individual or `naturales` for multiple."
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about the ethnicities of these people. When they do, please include this information in your output for the person in question. " \
            "There are a large number of ethnonyms that might appear in these records, so their appearance should be determined contextually rather than relying on a controlled vocabulary."
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about the ages of these people. When they do, please include this information in your output for the person in question. " \
            "In Spanish, this information might appear in the forms of words including (but not limited to): parvulo/a (translate to `infant`), niño/a (translate to `child`), or adulto/a (translate to `adult`)."
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about the legitimacy of the birth of these people, particularly infants. " \
            "When they do, please include this information in your output for the person in question." \
            "In Spanish, this information will usually appear as either legítimo/a (for legitimate) or ilegítimo/a (for illegitimate)."
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about the occupations of these people. When they do, please include this information in your output for the person in question. " \
            "In Spanish, this information will most frequently appear as a variety of words including (but not limited to) religioso, eclesiástico, clérigo, or cura (translate to `cleric`)." \
            "Other possibilities include ingeniero/a (translate to `engineer`)."
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about the phenotypes of these people. When they do, please include this information in your output for the person in question. " \
            "In Spanish, this information might appear in the forms of words including (but not limited to): negro/a (list as `negro`), moreno/a (list as `moreno`), or pardo/a (list as `mixed-race`)."
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about the freedom status of these people. When they do, please include this information in your output for the person in question. " \
            "In Spanish, this information might appear explicitly as either esclavo/a (list as `enslaved`) or libre (list as `free`). " \
            "Freedom status can also be communicated implicitly, most notably when the enslaver of one or more individuals is listed."
        },
        {
            "role": "system",
            "content": "These records will sometimes contain information about relationships between people. " \
            "Each relationship that you identify should be between 2 people and appear twice in your output, once for each related person. " \
            "The relationship type that you list for each individual should indicate what the related person *is to them*."
        },               
        {            
            "role": "user",
            "content": "Please list each person who appears in this transcription of a Spanish baptismal register: `" + example_01 + "`"
        },
        {
            "role": "assistant",
            "content": output_01
        },
        {            
            "role": "user",
            "content": "Please list each person who appears in this transcription of a Spanish baptismal register: `" + example_02 + "`"
        },
        {
            "role": "assistant",
            "content": output_02
        },
        {            
            "role": "user",
            "content": "Please list each person who appears in this transcription of a Spanish baptismal register: `" + example_03 + "`"
        },
        {
            "role": "assistant",
            "content": output_03
        },
        {            
            "role": "user",
            "content": "Please list each person who appears in this transcription of a Spanish baptismal register: `" + user_input + "`"
        }
    ]
)

model_output = response.choices[0].message.content

with open("output.json", "w", encoding="utf-8") as f:
    for line in model_output:
        f.write(line)

print("Output saved.")
