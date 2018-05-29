import json, pdb, glob, os, sys, shutil
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot

class data_manager_backend():
    def update_json(self):
        with open(self.description_file_path,'w') as fw:
            json.dump(self.descriptions.to_dict('records'), fw, indent = 4)

    def get_desc_key_list(self):
        # generate key list of all the name of variables in the descriptions
        keys = list(self.descriptions)
        # Ordering of keys. We want to have the following keys to be at front.
        keys.sort()
        for k_i, key in enumerate(self.preset_keys):
            if key in keys:
                keys.insert(k_i, keys.pop(keys.index(key)))
        # Put setname at the end because it is long.
        keys.insert(len(keys), keys.pop(keys.index('setname')))
        return list(keys)

    def __init__(self, folder_path, preset_keys = list()):
        # Each column in the table shows value for a key in description dictionaries. The keys will be sorted alphabetically, but preset_keys if provided will override the ordering. Put the keys preferred to be in front here. Any keys in the description will be displayed even if absent in preset_keys, but in the back.
        self.preset_keys = preset_keys
        # import descriptions
        self.footage_path = folder_path
        self.candidate_folder_path = os.path.join(folder_path, 'candidates')
        self.active_folder_path = os.path.join(folder_path, 'active')
        self.inactive_folder_path = os.path.join(folder_path, 'inactive')
        self.description_file_path = os.path.join(folder_path, "descriptions.json")
        with open(self.description_file_path,'r') as fr:
            json_str = fr.read()
            self.descriptions = pd.DataFrame(json.loads(json_str))
        if self.is_any_set_missing(): # make sure all the sets are present in the footage folder.
            sys.exit("Some sets listed in the description file are missing.")

    def is_any_set_missing(self):
        active_sets = os.listdir(self.active_folder_path)
        inactive_sets = os.listdir(self.inactive_folder_path)
        missing_set_count = 0
        for idx, desc in self.descriptions.iterrows():
            if desc['status'] == 'active':
                if not desc['setname'] in active_sets:
                    missing_set_count += 1
            elif desc['status'] == 'inactive':
                if not desc['setname'] in inactive_sets:
                    missing_set_count += 1
        if missing_set_count > 0:
            return True
        else:
            return False

    def delete_sets(self, set_ids):
        # make sure inactive files folder exists and create one if not.
        if not os.path.exists(self.inactive_folder_path):
            os.mkdir(self.inactive_folder_path)
        for idx, desc in self.descriptions[self.descriptions['status'] == 'active'].iterrows():
            if desc['id'] in set_ids: # this set should be deleted (moved to the 'inactive files' folder).
                # determine the destination file name
                dst_setname = ''
                if os.path.exists(os.path.join(self.inactive_folder_path, desc['setname'])): # if folder already exists
                    try_cnt = 1
                    while(True):
                        if os.path.exists(os.path.join(self.inactive_folder_path, desc['setname']) + '_' + str(try_cnt)):
                            try_cnt += 1
                        else:
                            dst_setname = os.path.join(self.inactive_folder_path, desc['setname']) + '_' + str(try_cnt)
                            break
                else:
                    dst_setname = os.path.join(self.inactive_folder_path, desc['setname'])
                # move the set
                result = shutil.move(os.path.join(self.active_folder_path, desc['setname']), dst_setname)
                # set status of the description to inactive if 'move' was successful.
                if os.path.exists(result):
                    self.descriptions.at[idx,'status'] = 'inactive'
        # update json file
        self.update_json()

    def add_set(self, set_path):
        # create a deprived description of the new set.
        max_id = 0
        for desc in all_descriptions:
            if max_id < desc['id']:
                max_id = desc['id']
        new_desc = {
            'id':max_id + 1,
            'method':'',
            'setup':'',
            'subject':'',
            'task':'',
            'date':'',
            'contents':'',
            'labels':'',
            'setname':os.path.basename(set_path),
            'status':'active',
        }
        # add new_desc to both all_descriptions and active_descriptions

    def update_description(self, idx, key, value):
        self.descriptions.at[idx,key] = value
        self.update_json()

