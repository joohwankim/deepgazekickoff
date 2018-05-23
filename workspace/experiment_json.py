import json, pdb

footages = list()
footages.append({
    'id':1,
    'method':'synthetic_blender',
    'setup':'lefteyeglasses',
    'subject':'female01',
    'task':'Plan002',
    'date':'20180520',
    'contents':['images','region_masks'],
    'labels':['gaze_xy','eyeball_xyz'],
    'setname':'0001_synthetic_blender_lefteyeglasses_female01_Plan002_20180520'
    })
footages.append({
    'id':2,
    'method':'synthetic_blender',
    'setup':'lefteyeglasses',
    'subject':'female01noglint',
    'task':'Plan002',
    'date':'20180520',
    'contents':['images','region_masks'],
    'labels':['gaze_xy','eyeball_xyz'],
    'setname':'0002_synthetic_blender_lefteyeglasses_female01noglint_Plan002_20180520'
    })
footages.append({
    'id':3,
    'method':'synthetic_blender',
    'setup':'lefteyeglasses',
    'subject':'male01noglint',
    'task':'Plan002',
    'date':'20180521',
    'contents':['images','region_masks'],
    'labels':['gaze_xy','eyeball_xyz'],
    'setname':'0003_synthetic_blender_lefteyeglasses_male01noglint_Plan002_20180521'
    })
footages.append({
    'id':4,
    'method':'synthetic_blender',
    'setup':'lefteyeglasses',
    'subject':'male02',
    'task':'Plan002',
    'date':'20180519',
    'contents':['images','region_masks'],
    'labels':['gaze_xy','eyeball_xyz'],
    'setname':'0004_synthetic_blender_lefteyeglasses_male02_Plan002_20180519'
    })
footages.append({
    'id':5,
    'method':'synthetic_blender',
    'setup':'lefteyeglasses',
    'subject':'male04noglint',
    'task':'Plan002',
    'date':'20180521',
    'contents':['images','region_masks'],
    'labels':['gaze_xy','eyeball_xyz'],
    'setname':'0005_synthetic_blender_lefteyeglasses_male04noglint_Plan002_20180521'
    })

with open('test.json','w') as fw:
    json.dump(footages, fw, indent = 4)

# with open('test.json','r') as fr:
#     json_str = fr.read()
#     test_data = json.loads(json_str)
#     print(test_data)

# pdb.set_trace()