import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class MultiInputDialog(QDialog):
    def __init__(self, labels, parent = None):
        # The layout is determined by the structure of labels.
        # The first dimension sets vertical location.
        # The second dimension (if provided) sets horizontal location.
        # Ordering of labels is the same as in the ordering in the list (or list of list).
        super(MultiInputDialog, self).__init__(parent)

        layout = QVBoxLayout(self)
        self.edit_boxes = list() # 1D list that contains edit boxes in the order of labels.

        for labels_this_row in labels:
            # layout that will store all the content for this row.
            hbox = QHBoxLayout()
            # if string, create a 1-element list using contaning string.
            if type(labels_this_row).__name__ == 'str':
                labels_this_row = [labels_this_row]

            # iterate through all the items in the list to create a label and an edit box
            for label in labels_this_row:
                l = QLabel(label)
                l.setAlignment(Qt.AlignLeft)
                hbox.addWidget(l)
                self.edit_boxes.append(QLineEdit())
                hbox.addWidget(self.edit_boxes[-1])
            layout.addLayout(hbox)

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
        for i, line_edit in enumerate(self.edit_boxes):
            input_data.append(line_edit.text())
        return input_data

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getInputs(labels, parent = None):
        dialog = MultiInputDialog(labels, parent)
        result = dialog.exec_()
        number = dialog.getData()
        return (number, result == QDialog.Accepted)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    input_data, ok = MultiInputDialog.getInputs([['resolution H','resolution_V'],'subsampling','which_eye'])
