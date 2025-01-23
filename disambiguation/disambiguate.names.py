import random
import json

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

def generate_name():
    fname = first_name("first_names.json")
    lname = last_name("last_names.json")
    gen_name = random.choice(fname) + ' ' + random.choice(lname)
    return gen_name

# create person record
# id, name, title, occupation, relationships etc.
# most likely: phenotype, title, maybe ethnicity/free/legitimate, everything else rare, relationships!!
def generatePersonRecord():
    indivRecord = {}
    indivRecord['name'] = generate_name()
    return indivRecord

def addPersonRecord(peopleList):
    indivRecord = {}
    indivRecord['name'] = generate_name()
    peopleList.update(indivRecord)
    return indivRecord

def testMyCode():
    myDictionary = {}
    addPersonRecord(myDictionary)
    print(myDictionary)

# general function to create field (whatever it is and select from the associated pool)
# basic control vocab for fields
# function to generate relationships
# update record creation function to build full record

testMyCode()