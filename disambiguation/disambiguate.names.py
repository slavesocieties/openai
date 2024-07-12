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

fname = first_name("first_names.json")
lname = last_name("last_names.json")
gen_name_1 = random.choice(fname)
gen_name_2 = random.choice(lname)
print(gen_name_1, gen_name_2)