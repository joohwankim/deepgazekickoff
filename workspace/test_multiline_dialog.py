import pdb, sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class MultiInputDialog(QDialog):
    def __init__(self, parent = None):
        super(MultiInputDialog, self).__init__(parent)

        layout = QVBoxLayout(self)

        # nice widget for editing the date
        self.datetime = QDateTimeEdit(self)
        self.datetime.setCalendarPopup(True)
        self.datetime.setDateTime(QDateTime.currentDateTime())
        layout.addWidget(self.datetime)

        # receive a number
        self.test_input = QLineEdit(self)

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

    def returnNumber(self):
        return float(self.test_input.text())

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getDateTime(parent = None):
        dialog = MultiInputDialog(parent)
        result = dialog.exec_()
        date = dialog.dateTime()
        number = dialog.returnNumber()
        return (date.date(), date.time(), number, result == QDialog.Accepted)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    date, time, number, ok = MultiInputDialog.getDateTime()
    pdb.set_trace()
