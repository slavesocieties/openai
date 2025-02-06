import random
import json
import os.path

def read_names(spanish_names):
    with open(spanish_names, 'r') as sn:
        data = json.load(sn)
        names = data['names']
        return names

def random_element(names):
    name_index = random.randint(0, len(names)-1)
    return names[name_index]

def first_name(first_names):
    with open(first_names, 'r') as fn:
        data = json.load(fn)
        f_name = data['f_names']
    return f_name

def last_name(last_names):
    with open(last_names, 'r') as ln:
        data = json.load(ln)
        l_name = data['l_names']
    return l_name


# span_nl = read_names("spanish_names.json")
# print(random_element(span_nl))

# add code to randomize name selection

def generateName():
    fname = first_name("first_names.json")
    lname = last_name("last_names.json")
    gen_name = random.choice(fname) + ' ' + random.choice(lname)
    return gen_name

# create person record
# id, name, title, occupation, relationships etc.
# most likely: phenotype, title, maybe ethnicity/free/legitimate, everything else rare, relationships!!
'''def generatePersonRecord():
    indivRecord = {}
    indivRecord['name'] = generateName()
    return indivRecord

def addPersonRecord(peopleList):
    indivRecord = {}
    indivRecord['name'] = generateName()
    peopleList.update(indivRecord)
    return indivRecord

def testMyCode():
    myDictionary = {}
    addPersonRecord(myDictionary)
    print(myDictionary)'''

# general function to create field (whatever it is and select from the associated pool)
# basic control vocab for fields
# function to generate relationships
# update record creation function to build full record

'''def createPool():
    vocabPool = {}
    currentDir = os.getcwd()
    parentDir = os.path.dirname(currentDir)
    pathToFile = os.path.join(parentDir, "vocab.json")
    with open(pathToFile, 'r') as allVocab:
        data = json.load(allVocab)
        allVocab = data["controlled_vocabularies"]
        for vocabSet in allVocab:
            vocabPool[vocabSet["key"]] = vocabSet["vocab"]
    return vocabPool'''


# create single term
# personRecords is existing dictionary, add termToCreate
# can also modify: create personRecords within method and return single term dict
def createTerm(personRecords, termToCreate):
    randomTerm = "N/A"
    currentDir = os.getcwd()
    parentDir = os.path.dirname(currentDir)
    pathToFile = os.path.join(parentDir, "vocab.json")
    with open(pathToFile, 'r') as allVocab:
        data = json.load(allVocab)
        allVocab = data["controlled_vocabularies"]
    for vocabSet in allVocab:
        if vocabSet["key"] == termToCreate:
            vocabIndex = random.randint(0, len(vocabSet["vocab"]) - 1)
            randomTerm = vocabSet["vocab"][vocabIndex]
            break
    personRecords[termToCreate] = randomTerm


'''def createNewPerson(vocabPool):
    personRecords = {}
    personRecords["name"] = generateName()
    for vocabType, vocabList in vocabPool.items():
        vocabIndex = random.randint(0, len(vocabList) - 1)
        personRecords[vocabType] = vocabList[vocabIndex]
    return personRecords'''

# Test code
'''personRecords = {}
createTerm(personRecords, "titles")
print(personRecords)'''