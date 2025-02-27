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

allRelTypes = ["slave", "enslaver", "godparent", "godchild", "spouse", "child", "parent"]
# death might not be the actual word for it - check
relsDict = {"baptism": ["enslaver", "godparent", "parent", "parent"],
            "marriage": ["slave", "enslaver", "spouse"],
            "death": ["slave", "enslaver", "child"]}


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

    thisScenario, allThisRels = genRelSet()
    if len(allThisRels) > 0:
        thisPerson["relationships"] = allThisRels

    return thisScenario, thisPerson


# alternative for generate rels: based on scenario

def genRelSet():
    scenario = random.choice(list(relsDict.keys()))
    possibleRels = relsDict[scenario][:]
    addRels = []

    howMany = random.randint(0, len(possibleRels))

    for i in range(howMany):
        if not possibleRels:
            return scenario, addRels

        relType = random.choice(possibleRels)

        if relType == "slave" and "enslaver" in possibleRels:
            possibleRels.remove("enslaver")
        elif relType == "enslaver" and "slave" in possibleRels:
            possibleRels.remove("slave")

        possibleRels.remove(relType)
        thisRel = {"related_person": generateName(), "relationship_type": relType}
        addRels.append(thisRel)
    
    return scenario, addRels

# second person
# classification of relationship: same, diffname, diffppl
# def otherRelated():


# Test code
scenario, personRecords = genIndiv()
print(personRecords)




# here lies scrap code RIP

'''def genRelatedPerson(possibleRels):
    relatedFile = {}
    relatedFile["related_person"] = generateName()

    vocabIndex = random.randint(0, len(possibleRels) - 1)
    randomTerm = possibleRels[vocabIndex]
    relatedFile["relationship_type"] = randomTerm

    return randomTerm, relatedFile


def genAllRels(possibleRels):
    copyRels = possibleRels[:]
    allRels = []

    for i in range(random.randint(0, 3)):
        relUsed, newFile = genRelatedPerson(copyRels)
        allRels.append(newFile)
        copyRels.remove(relUsed)

    return copyRels, allRels'''

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