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

class DataManager(QWidget):
    def __init__(self, data_path, preset_keys):
        super().__init__()
        # self.title = 'Dataset Management'
        # initialize footage manager
        # data_path for footage: \\\\dcg-zfs-01.nvidia.com\\deep-gaze2.cosmos393/footage
        # preset_keys for footage: ['id','date','method','setup','subject','labels','contents']
        self.F = data_manager_backend(data_path, preset_keys)
        # self.w_left = 50
        # self.w_top = 50
        # self.w_width = 1200
        # self.w_height = 800
        self.highlighted_set_ids = set()
        self.sort_key = 'setname'
        self.initUI()
        # Show widget
        # self.show()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.create_buttons_box()
        self.layout.addLayout(self.buttons_box)
        self.create_table()
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)

    def create_buttons_box(self):
        # Add buttons
        self.buttons_box = QHBoxLayout()
        self.add_button = QPushButton('Add footage')
        self.add_button.clicked.connect(self.on_add)
        self.delete_button = QPushButton('Delete footage')
        self.delete_button.clicked.connect(self.on_delete)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.clicked.connect(self.on_refresh)
        self.buttons_box.addWidget(self.add_button)
        self.buttons_box.addWidget(self.delete_button)
        self.buttons_box.addWidget(self.refresh_button)

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

        # retrieve all the active descriptions from self.F, sort, and display.
        self.tableWidget.setRowCount(len(self.F.descriptions[self.F.descriptions['status'] == 'active']))
        for r_i in range(self.tableWidget.rowCount()):
            for c_i in range(self.tableWidget.columnCount()):
                self.tableWidget.setItem(r_i,c_i,QTableWidgetItem())

        # setting up call back functions
        self.tableWidget.cellClicked.connect(self.on_cell_click)
        self.tableWidget.cellDoubleClicked.connect(self.on_cell_double_click)
        # self.tableWidget.itemChanged.connect(self.on_item_change)
        self.tableWidget.horizontalHeader().sectionClicked.connect(self.on_header_click)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setFocusPolicy(Qt.NoFocus)

        self.update_cells()

    def update_cells(self):
        desc_keys = self.F.get_desc_key_list()
        # NOTE: displayed_descriptions is for making connections between user-selected cells and self.F.descriptions.
        # Actual data management should only happen through self.F (the backend manager)
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
                    self.tableWidget.item(r_i,c_i).setText(content_string)
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
        # If value is of string type...
        if key in ['method','setup','subject','task','date']:
            text, ok_pressed = QInputDialog.getText(self, "Input text",key + ':', QLineEdit.Normal, self.F.descriptions.at[idx, key])
            if ok_pressed:
                self.F.update_description(idx,key,text)
        # If change in value involves moving the folder...
        elif key in ['setname']:
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
        # If value is a list (of string)...
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
    def on_refresh(self):
        # refresh cell content
        self.update_cells()

class FootageManager(DataManager):
    def __init__(self, data_path, preset_keys):
        super().__init__(data_path, preset_keys)

    def create_buttons_box(self):
        super().create_buttons_box()
        self.create_h5_button = QPushButton('Create H5')
        self.create_h5_button.clicked.connect(self.on_create_h5)
        self.buttons_box.addWidget(self.create_h5_button)

    @pyqtSlot()
    def on_create_h5(self):
        # open file dialog and receive file list
        str_inputs, toggle_inputs, ok = MultiInputDialog.getInputs(line_edit_labels = [['resolution_H','resolution_V'],'subsampling','which_eye'], toggle_button_labels = 'Run on cluster (local if not highlighted)')
        # create an h5 set definition, update the h5 json list, run data_prep.py

class DatasetManager(DataManager):
    def __init__(self, data_path, preset_keys):
        super().__init__(data_path, preset_keys)

    def create_buttons_box(self):
        super().create_buttons_box()
        # in case an h5 file generation failed and needs to be regenerated.
        self.remake_h5_button = QPushButton('Restart H5 generation')
        self.remake_h5_button.clicked.connect(self.on_remake_h5)
        self.buttons_box.addWidget(self.remake_h5_button)

    @pyqtSlot()
    def on_remake_h5(self):
        # execute data_prep.py again
        pass # just for now; fill in execution of data_prep.py later.
 
class DataManagementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'CNN Gaze Data Management'
        self.left = 50
        self.top = 50
        self.width = 1200
        self.height = 800
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        self.tab_widget = DataManagementTabs(self)
        self.setCentralWidget(self.tab_widget)
 
        self.show()
 
class DataManagementTabs(QWidget):        
 
    def __init__(self, parent):   
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
 
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.footage_tab = FootageManager("\\\\dcg-zfs-01.nvidia.com\\deep-gaze2.cosmos393/footage", ['id','date','method','setup','subject','labels','contents'])
        self.dataset_tab = DatasetManager("\\\\dcg-zfs-01.nvidia.com\\deep-gaze2.cosmos393/footage", ['resolution','subsampling','which_eye','input footage'])
        self.tabs.resize(300,200) 
 
        # Add tabs
        self.tabs.addTab(self.footage_tab,"Footage")
        self.tabs.addTab(self.dataset_tab,"Dataset")
 
        # # Create first tab
        # self.tab1.layout = QVBoxLayout(self.tab1)
        # self.pushButton1 = QPushButton("PyQt5 button")
        # self.tab1.layout.addWidget(self.pushButton1)
        # self.tab1.setLayout(self.tab1.layout)
 
        # Add tabs to widget        
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
 
    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataManagementApp()
    sys.exit(app.exec_())
