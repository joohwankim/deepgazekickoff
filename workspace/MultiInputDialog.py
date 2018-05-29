import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class MultiInputDialog(QDialog):
    def __init__(self, line_edit_labels, toggle_button_labels = list(), parent = None):
        # The layout is determined by the structure of line_edit_labels.
        # The first dimension sets vertical location.
        # The second dimension (if provided) sets horizontal location.
        # Ordering of line_edit_labels is the same as in the ordering in the list (or list of list).
        super(MultiInputDialog, self).__init__(parent)

        layout = QVBoxLayout(self)
        self.edit_boxes = list() # 1D list that contains edit boxes in the order of line_edit_labels.

        for line_edit_labels_this_row in line_edit_labels:
            # layout that will store all the content for this row.
            hbox = QHBoxLayout()
            # if string, create a 1-element list using contaning string.
            if type(line_edit_labels_this_row).__name__ == 'str':
                line_edit_labels_this_row = [line_edit_labels_this_row]

            # iterate through all the items in the list to create a line_edit_label and an edit box
            for line_edit_label in line_edit_labels_this_row:
                l = QLabel(line_edit_label)
                l.setAlignment(Qt.AlignLeft)
                hbox.addWidget(l)
                self.edit_boxes.append(QLineEdit())
                hbox.addWidget(self.edit_boxes[-1])
            layout.addLayout(hbox)

        # Add toggle button if toggle_button_labels was provided
        if type(toggle_button_labels).__name__ != 'list': # make it a list if it is not
            toggle_button_labels = [toggle_button_labels]
        self.toggle_button_labels = toggle_button_labels

        self.toggle_buttons = list()
        for toggle_button_label in self.toggle_button_labels:
            self.toggle_buttons.append(QPushButton(toggle_button_label))
            self.toggle_buttons[-1].setCheckable(True)
            layout.addWidget(self.toggle_buttons[-1])

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # get current date and time from the dialog
    def dateTime(self):
        return self.datetime.dateTime()

    def getData(self):
        input_data = list()
        for i, line_edit_label in enumerate(self.edit_boxes):
            input_data.append(line_edit_label.text())
        if len(self.toggle_button_labels) == 0:
            return input_data
        else:
            toggle_inputs = list()
            for toggle_button in self.toggle_buttons:
                toggle_inputs.append(toggle_button.isChecked())
            return (input_data, toggle_inputs)

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getInputs(line_edit_labels, toggle_button_labels = list(), parent = None):
        dialog = MultiInputDialog(line_edit_labels, toggle_button_labels, parent)
        result = dialog.exec_()
        if len(toggle_button_labels) == 0:
            line_edit_inputs = dialog.getData()
            return (line_edit_inputs, result == QDialog.Accepted)
        else:
            line_edit_inputs, toggle_inputs = dialog.getData()
            return (line_edit_inputs, toggle_inputs, result == QDialog.Accepted)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    line_edit_inputs, ok = MultiInputDialog.getInputs(line_edit_labels = [['resolution H','resolution_V'],'subsampling','which_eye'])
    print(line_edit_inputs)
    line_edit_inputs, toggle_inputs, ok = MultiInputDialog.getInputs(line_edit_labels = [['resolution H','resolution_V'],'subsampling','which_eye'], toggle_button_labels = ['run on Saturn V','test'])
    print(line_edit_inputs)
    print(toggle_inputs)