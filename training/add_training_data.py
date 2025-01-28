import json

# f = open ("training/239746.json", "r")
# e = open("training_data.json")
# data = json.load(f)
# edata = json.load(e)

# e_raw = [e['raw'] for e in edata['examples']]

# new_data = [e for e in data['entries'] if "normalized" in e and e['raw'] not in e_raw]
# for e in new_data:
#     e['type'] = data['type']
#     e['language'] = data['language']
#     e['country'] = data['country']
#     e['state'] = data['state']
#     e['city'] = data['city']
#     e['institution'] = data['institution']
#     e.pop('id')
#     e['id'] = data['id']
#     e['raw'] = e.pop('raw')
#     e['normalized'] = e.pop('normalized')
#     e['data'] = e.pop('data')

# with open("training/239746-test.json", "w") as f:
#     json.dump(new_data, f)

nullables = ['rank', 'origin', 'ethnicity', 'age', 'legitimate', 'occupation', 'phenotype', 'free']
f = open("training_data.json", "r")
data = json.load(f)
for e in data['examples']:
    for p in e['data']['people']:
        assert('id' in p)
        assert('name' in p)
        for n in nullables:
             assert(n in p)
    for ev in e['data']['events']:
        assert('type' in ev)
        assert('principal' in ev)
        assert('date' in ev)
        if 'principals' in ev:
            ev.pop('principals')

with open("training/new_training_data.json", "w") as f:
    print(len(data['examples']))
    json.dump(data, f)