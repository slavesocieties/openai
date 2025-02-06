import random
import json
import os.path


# create person record
# id, name, title, occupation, relationships etc.
# most likely: phenotype, title, maybe ethnicity/free/legitimate, everything else rare, relationships!!

# general function to create field (whatever it is and select from the associated pool)
# basic control vocab for fields
# function to generate relationships
# update record creation function to build full record

# create single term
# personRecords is existing dictionary, add termToCreate
# can also modify: create personRecords within method and return single term dict



# set numeric scale for frequency of terms
# missing origin (also missing in json)
necTerm = ["f_names", "l_names"]
allTerms = ["phenotype", "titles", "legitimate", "free", "ethnicity", "ranks", "age", "occupation"]
frequentTerms = ["phenotype", "titles"]
commonTerms = ["legitimate", "free", "ethnicity"]
rareTerms = ["ranks", "age", "occupation"]
relTerms = ["related_person", "relationship_type"]

# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
def genFrequency(this, term):
    frequencyIndex = random.randint(0, 10)
    if term in frequentTerms:
        if frequencyIndex < 6:
            createTerm(this, term)
    elif term in commonTerms:
        if frequencyIndex < 4:
            createTerm(this, term)
    else:
        if frequencyIndex < 2:
            createTerm(this, term)



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



def read_names(spanish_names):
    with open(spanish_names, 'r') as sn:
        data = json.load(sn)
        names = data['names']
        return names

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

def generateName():
    fname = first_name("first_names.json")
    lname = last_name("last_names.json")
    gen_name = random.choice(fname) + ' ' + random.choice(lname)
    return gen_name


# rudimentary generate person code
def genIndiv():
    thisPerson = {}

    for nameTerm in necTerm:
        createTerm(thisPerson, nameTerm)
    
    for term in allTerms:
        genFrequency(thisPerson, term)

    return thisPerson



# question: name affect relationship?
def genRelatedPerson():
    relatedFile = {}
    relatedFile["related_person"] = generateName()
    createTerm(relatedFile, "relationship_type")
    return relatedFile



# second person
# classification of relationship: same, diffname, diffppl
'''def otherRelated(similarity):
    if similarity == "exactSame":
        relatedPerson = existingPerson'''



# Test code
'''personRecords = {}
createTerm(personRecords, "titles")
print(personRecords)'''




# here lies scrap code RIP

'''
def random_element(names):
    name_index = random.randint(0, len(names)-1)
    return names[name_index]

# span_nl = read_names("spanish_names.json")
# print(random_element(span_nl))

# add code to randomize name selection

def generatePersonRecord():
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
    print(myDictionary)

def createPool():
    vocabPool = {}
    currentDir = os.getcwd()
    parentDir = os.path.dirname(currentDir)
    pathToFile = os.path.join(parentDir, "vocab.json")
    with open(pathToFile, 'r') as allVocab:
        data = json.load(allVocab)
        allVocab = data["controlled_vocabularies"]
        for vocabSet in allVocab:
            vocabPool[vocabSet["key"]] = vocabSet["vocab"]
    return vocabPool

def createNewPerson(vocabPool):
    personRecords = {}
    personRecords["name"] = generateName()
    for vocabType, vocabList in vocabPool.items():
        vocabIndex = random.randint(0, len(vocabList) - 1)
        personRecords[vocabType] = vocabList[vocabIndex]
    return personRecords
    '''