# This script provides a GUI that can do the following.
# 1. Show a table of items in 'footage' folder.
# 2. Sort the items with respect to the property chosen by the viewer.
# 3. Provide a clipboard copy of the paths to the selected items.

import json, pdb, glob, os, sys, shutil
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout,QHeaderView, QAbstractItemView, QMenuBar, QAction
from PyQt5 import QtGui
# from PyQt5.QtGui import QIcon, QColor, QMenuBar
from PyQt5.QtCore import pyqtSlot

class footage_manager_backend():
    def update_json(self):
        with open(os.path.join(self.footage_path, self.description_filename),'w') as fw:
            json.dump(self.descriptions, fw, indent = 4)

    def get_desc_key_list(self):
        # generate key list of all the items in the descriptions
        keys = set()
        for desc in self.descriptions:
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

    def sort_descriptions(self, sort_key):
        sorted_descriptions = sorted(self.descriptions, key = lambda x:x[sort_key])
        self.descriptions = sorted_descriptions

    def __init__(self):
        # import descriptions
        self.footage_path = "\\\\dcg-zfs-01.nvidia.com\\deep-gaze2.cosmos393/footage"
        self.inactive_files_path = "\\\\dcg-zfs-01.nvidia.com\\deep-gaze2.cosmos393/footage/inactive"
        self.description_filename = "footage_descriptions.json"
        with open(os.path.join(self.footage_path, self.description_filename),'r') as fr:
            json_str = fr.read()
            self.descriptions = json.loads(json_str)
        self.sort_descriptions('setname')
        if self.is_any_set_missing(): # make sure all the sets are present in the footage folder.
            sys.exit("Some sets listed in the description file are missing.")

    def is_any_set_missing(self):
        dirs = os.listdir(self.footage_path)
        missing_dir_count = 0
        for description in self.descriptions:
            if not description['setname'] in dirs:
                missing_dir_count += 1
        if missing_dir_count > 0:
            return True
        else:
            return False

    def delete_set(self, set_ids):
        # make sure inactive files folder exists and create one if not.
        if not os.path.exists(self.inactive_files_path):
            os.mkdir(self.inactive_files_path)
        for desc in self.descriptions:
            if desc['id'] in set_ids: # this set should be deleted (moved to the 'inactive files' folder).
                # determine the destination file name
                dst_setname = ''
                if os.path.exists(os.path.join(self.inactive_files_path, desc['setname'])): # if folder already exists
                    try_cnt = 1
                    while(True):
                        if os.path.exists(os.path.join(self.inactive_files_path, desc['setname']) + '_' + str(try_cnt)):
                            try_cnt += 1
                        else:
                            dst_setname = os.path.join(self.inactive_files_path, desc['setname']) + '_' + str(try_cnt)
                            break
                else:
                    dst_setname = os.path.join(self.inactive_files_path, desc['setname'])
                # move the set
                result = shutil.move(os.path.join(self.footage_path, desc['setname']), dst_setname)
                # remove from descriptions if 'move' was successful.
                if os.path.exists(result):
                    for i, description in enumerate(self.descriptions):
                        if desc['setname'] == description['setname']:
                            del self.descriptions[i]
        # update json file
        self.update_json()


        # for set_id in set_ids:
        #     # move the folder to the inactive files folder.
        #     # determine destination set name
        #     dst_setname = ''
        #     if os.path.exists(os.path.join(self.inactive_files_path, d_set['setname'])): # if folder already exists
        #         try_cnt = 1
        #         while(True):
        #             if os.path.exists(os.path.join(self.inactive_files_path, d_set['setname']) + '_' + str(try_cnt)):
        #                 try_cnt += 1
        #             else:
        #                 dst_setname = os.path.join(self.inactive_files_path, d_set['setname']) + '_' + str(try_cnt)
        #                 break
        #     else:
        #         dst_setname = os.path.join(self.inactive_files_path, d_set['setname'])
        #     # move the folder
        #     result = shutil.move(os.path.join(self.footage_path, d_set['setname']), dst_setname)
        #     # remove from descriptions if 'move' was successful.
        #     if os.path.exists(result):
        #         for i, description in enumerate(self.descriptions):
        #             if d_set['setname'] == description['setname']:
        #                 del self.descriptions[i]
        # # update json file
        # self.update_json()

class FootageManager(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Footage Management'
        self.F = footage_manager_backend()
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
        self.tableWidget.setRowCount(len(self.F.descriptions))
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
        self.tableWidget.setRowCount(len(self.F.descriptions))
        desc_keys = self.F.get_desc_key_list()
        for r_i, desc in enumerate(self.F.descriptions):
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
            selected_set_id = self.F.descriptions[currentQTableWidgetItem.row()]['id']
            if selected_set_id in self.highlighted_set_ids:
                self.highlighted_set_ids.remove(selected_set_id)
            else:
                self.highlighted_set_ids.add(selected_set_id)
        desc_keys = self.F.get_desc_key_list()
        self.update_cells()

    @pyqtSlot()
    def on_header_click(self):
        c_i = self.tableWidget.selectedItems()[0].column()
        # sort descriptions with respect to the selected key
        self.F.sort_descriptions(self.F.get_desc_key_list()[c_i])
        # update the table so that descriptions are ordered with the new result
        self.update_cells()

    @pyqtSlot()
    def on_delete(self):
        # delete highlighted files
        self.F.delete_set(self.highlighted_set_ids)
        # update the table
        self.update_cells()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FootageManager()
    sys.exit(app.exec_())