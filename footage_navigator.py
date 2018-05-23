# This script provides a GUI that can do the following.
# 1. Show a table of items in 'footage' folder.
# 2. Sort the items with respect to the property chosen by the viewer.
# 3. Provide a clipboard copy of the paths to the selected items.

import json, pdb, glob, os, sys, shutil
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout,QHeaderView, QAbstractItemView, QMenuBar, QAction, QFileDialog, QInputDialog
from PyQt5 import QtGui
from PyQt5.QtGui import QClipboard
# from PyQt5.QtGui import QIcon, QColor, QMenuBar
from PyQt5.QtCore import pyqtSlot

class data_manager_backend():
    def update_json(self):
        with open(os.path.join(self.footage_path, self.description_filename),'w') as fw:
            json.dump(self.active_descriptions, fw, indent = 4)

    def get_desc_key_list(self):
        # generate key list of all the items in the descriptions
        keys = set()
        for desc in self.active_descriptions:
            for key in desc.keys():
                keys.add(key)
        keys = list(keys)
        # Ordering of keys. We want to have the following keys to be at front.
        keys.sort()
        predetermined_key_order = ['method','setup','subject','labels','contents']
        for k_i, key in enumerate(predetermined_key_order):
            if key in keys:
                keys.insert(k_i, keys.pop(keys.index(key)))
        # Put setname at the end because it is long.
        keys.insert(len(keys), keys.pop(keys.index('setname')))
        return list(keys)

    def sort_descriptions(self, sort_key): # sort active descriptions
        sorted_descriptions = sorted(self.active_descriptions, key = lambda x:x[sort_key])
        self.active_descriptions = sorted_descriptions

    def __init__(self, folder_path):
        # import descriptions
        self.footage_path = folder_path
        self.candidate_folder_path = os.path.join(folder_path, 'candidates')
        self.active_folder_path = os.path.join(folder_path, 'active')
        self.inactive_folder_path = os.path.join(folder_path, 'inactive')
        self.description_file_path = os.path.join(folder_path, "descriptions.json")
        with open(self.description_file_path,'r') as fr:
            json_str = fr.read()
            self.all_descriptions = json.loads(json_str)
            self.active_descriptions = list()
            self.inactive_descriptions = list()
            for desc in self.all_descriptions:
                if desc['status'] == 'active':
                    self.active_descriptions.append(desc)
                elif desc['status'] == 'inactive':
                    self.inactive_descriptions.append(desc)
        if self.is_any_set_missing(): # make sure all the sets are present in the footage folder.
            sys.exit("Some sets listed in the description file are missing.")

    def is_any_set_missing(self):
        active_sets = os.listdir(self.active_folder_path)
        inactive_sets = os.listdir(self.inactive_folder_path)
        missing_set_count = 0
        for desc in self.active_descriptions:
            if not desc['setname'] in active_sets:
                missing_set_count += 1
        for desc in self.inactive_descriptions:
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
        for desc in self.active_descriptions:
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
                # remove from descriptions if 'move' was successful.
                if os.path.exists(result):
                    for i, description in enumerate(self.active_descriptions):
                        if desc['setname'] == description['setname']:
                            del self.active_descriptions[i]
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


class DatasetManager(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Dataset Management'
        # initialize footage manager
        self.F = data_manager_backend("\\\\dcg-zfs-01.nvidia.com\\deep-gaze2.cosmos393/footage")
        self.w_left = 50
        self.w_top = 50
        self.w_width = 1200
        self.w_height = 800
        self.highlighted_set_ids = set()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.w_left, self.w_top, self.w_width, self.w_height)
 
        # Add action menus
        self.menu_bar = QMenuBar(self)
        fileMenu = self.menu_bar.addMenu('File')
        deleteAction = QAction('Delete',self)
        deleteAction.triggered.connect(self.on_delete)
        fileMenu.addAction(deleteAction)
        addAction = QAction('Add from candidates',self)
        addAction.triggered.connect(self.on_add)
        fileMenu.addAction(addAction)
 
        self.createTable()

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.menu_bar)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)

        # Show widget
        self.show()
 
    def createTable(self):
        desc_keys = self.F.get_desc_key_list()
       # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(self.F.active_descriptions))
        self.tableWidget.setColumnCount(len(desc_keys))
        # self.tableWidget.setItem(0,c_i,QTableWidgetItem(key))
        self.tableWidget.setHorizontalHeaderLabels(desc_keys)
        for c_i in range(len(desc_keys)):
            self.tableWidget.horizontalHeader().setSectionResizeMode(c_i, QHeaderView.ResizeToContents)
        self.update_cells()
        # setting up call back functions
        self.tableWidget.cellClicked.connect(self.on_cell_click)
        self.tableWidget.horizontalHeader().sectionClicked.connect(self.on_header_click)

    def update_cells(self):
        self.tableWidget.setRowCount(len(self.F.active_descriptions))
        desc_keys = self.F.get_desc_key_list()
        for r_i, desc in enumerate(self.F.active_descriptions):
            for c_i, key in enumerate(desc_keys):
                # fill in string contents
                if key in desc:
                    content_string = ''
                    if type(desc[key]).__name__ == 'list':
                        for i, item in enumerate(desc[key]):
                            content_string += item
                            if i < len(desc[key]) - 1:
                                content_string += ', '
                    elif type(desc[key]).__name__ != 'str':
                        content_string = str(desc[key])
                    else:
                        content_string = desc[key]
                    self.tableWidget.setItem(r_i,c_i,QTableWidgetItem(content_string))
                # color the background of each cell
                if desc['id'] in self.highlighted_set_ids:
                    self.tableWidget.item(r_i,c_i).setBackground(QtGui.QColor(250,150,150))
                else:
                    self.tableWidget.item(r_i,c_i).setBackground(QtGui.QColor(255,255,255))

    @pyqtSlot()
    def on_cell_click(self):
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            selected_set_id = self.F.active_descriptions[currentQTableWidgetItem.row()]['id']
            if selected_set_id in self.highlighted_set_ids:
                self.highlighted_set_ids.remove(selected_set_id)
            else:
                self.highlighted_set_ids.add(selected_set_id)
        desc_keys = self.F.get_desc_key_list()
        self.update_cells()

    @pyqtSlot()
    def on_header_click(self):
        c_i = self.tableWidget.selectedItems()[0].column()
        # sort active_descriptions with respect to the selected key
        self.F.active_descriptions = sorted(self.F.active_descriptions, key = lambda x:x[self.F.get_desc_key_list()[c_i]])
        # update the table so that active_descriptions are ordered with the new result
        self.update_cells()

    @pyqtSlot()
    def on_delete(self):
        # delete highlighted files
        self.F.delete_sets(self.highlighted_set_ids)
        # update the table
        self.update_cells()

    @pyqtSlot()
    def on_add(self):
        # open file dialog and receive file list
        selected_folder = QFileDialog.getExistingDirectory(self, "Select sets to be added.", self.F.candidate_folder_path) # one set at a time for now.
        # make the deprived version of description for the new set. For now the user is responsible to modify descriptions after the set is added.
        self.F.add_set(selected_folder)
        # update the table
        self.update_cells()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DatasetManager()
    sys.exit(app.exec_())