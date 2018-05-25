# This script provides a GUI that can do the following.
# 1. Show a table of items in 'footage' folder.
# 2. Sort the items with respect to the property chosen by the viewer.
# 3. Provide a clipboard copy of the paths to the selected items.

import json, pdb, sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
from MultiInputDialog import *
from DataNavigatorBackend import *

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
        self.sort_key = 'setname'
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
        addAction = QAction('Add footage',self)
        addAction.triggered.connect(self.on_add)
        fileMenu.addAction(addAction)
        createH5Action = QAction('Create H5',self)
        createH5Action.triggered.connect(self.on_create_h5)
        fileMenu.addAction(createH5Action)
 
        self.create_table()

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.menu_bar)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)

        # Show widget
        self.show()
 
    def create_table(self):
        desc_keys = self.F.get_desc_key_list()
       # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(self.F.descriptions[self.F.descriptions['status'] == 'active']))
        self.tableWidget.setColumnCount(len(desc_keys))
        # self.tableWidget.setItem(0,c_i,QTableWidgetItem(key))
        self.tableWidget.setHorizontalHeaderLabels(desc_keys)
        for c_i in range(len(desc_keys)):
            self.tableWidget.horizontalHeader().setSectionResizeMode(c_i, QHeaderView.ResizeToContents)
        self.update_cells()
        # setting up call back functions
        self.tableWidget.cellClicked.connect(self.on_cell_click)
        self.tableWidget.cellDoubleClicked.connect(self.on_cell_double_click)
        # self.tableWidget.itemChanged.connect(self.on_item_change)
        self.tableWidget.horizontalHeader().sectionClicked.connect(self.on_header_click)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def update_cells(self):
        # retrieve all the active descriptions from self.F, sort, and display.
        self.tableWidget.setRowCount(len(self.F.descriptions[self.F.descriptions['status'] == 'active']))
        desc_keys = self.F.get_desc_key_list()
        # NOTE: displayed_descriptions is for making connections between user-selected cells and self.F.descriptions.
        # Actual data management should directly happen on self.F.descriptions.
        self.displayed_descriptions = self.F.descriptions[self.F.descriptions['status'] == 'active'].sort_values(by = self.sort_key).reset_index()
        for r_i, desc in self.displayed_descriptions.iterrows():
            for c_i, key in enumerate(desc_keys):
                # fill in string contents
                if key in desc:
                    if type(desc[key]).__name__ == 'list':
                        content_string = ', '.join(desc[key])
                    elif type(desc[key]).__name__ != 'str':
                        content_string = str(desc[key])
                    else:
                        content_string = desc[key]
                    self.tableWidget.setItem(r_i,c_i,QTableWidgetItem(content_string))
                # color the background of each cell
                if desc['id'] in self.highlighted_set_ids:
                    self.tableWidget.item(r_i,c_i).setBackground(QColor(250,150,150))
                else:
                    self.tableWidget.item(r_i,c_i).setBackground(QColor(255,255,255))

    @pyqtSlot()
    def on_cell_click(self):
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            selected_set_id = self.displayed_descriptions.at[currentQTableWidgetItem.row(),'id']
            if selected_set_id in self.highlighted_set_ids:
                self.highlighted_set_ids.remove(selected_set_id)
            else:
                self.highlighted_set_ids.add(selected_set_id)
        self.update_cells()

    @pyqtSlot()
    def on_cell_double_click(self): # edit a cell
        displayed_i = self.tableWidget.selectedItems()[0].row()
        # find index in self.F.descriptions for the selected cell.
        idx = self.F.descriptions.index[self.F.descriptions['id'] == self.displayed_descriptions.at[displayed_i, 'id']].tolist()[0]
        key = self.F.get_desc_key_list()[self.tableWidget.selectedItems()[0].column()]
        if key in ['method','setup','subject','task','date']:
            text, ok_pressed = QInputDialog.getText(self, "Input text",key + ':', QLineEdit.Normal, self.F.descriptions.at[idx, key])
            if ok_pressed:
                self.F.update_description(idx,key,text)
        elif key == 'setname':
            text, ok_pressed = QInputDialog.getText(self, "Input text",key + ':', QLineEdit.Normal, self.F.descriptions.at[idx, key])
            if ok_pressed:
                # attempt to change the folder name.
                pdb.set_trace()
                result = shutil.move(os.path.join(self.F.active_folder_path, self.F.descriptions.at[idx, key]), os.path.join(self.F.active_folder_path, text))
                # if successful, change descriptions.
                if os.path.basename(result) == text:
                    self.F.update_description(idx,key,text)
                else:
                    QMessageBox.about(self,'Error message','File renaming was unsuccessful. Keeping previous value.')
        elif key in ['labels','contents']:
            content_string = ', '.join(self.F.descriptions.at[idx,key])
            text, ok_pressed = QInputDialog.getText(self, "Input text",key + ':', QLineEdit.Normal, content_string)
            if ok_pressed:
                content_list = text.replace(' ','').split(',')
                self.F.update_description(idx,key,content_list)
        self.update_cells()

    @pyqtSlot()
    def on_header_click(self):
        c_i = self.tableWidget.selectedItems()[0].column()
        self.sort_key = self.F.get_desc_key_list()[c_i]
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

    @pyqtSlot()
    def on_create_h5(self):
        # open file dialog and receive file list
        inputs, ok = MultiInputDialog.getInputs([['resolution H','resolution_V'],'subsampling','which_eye'])
        pdb.set_trace()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DatasetManager()
    sys.exit(app.exec_())