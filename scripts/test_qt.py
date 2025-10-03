"""
Simple PyQt5 test to see if Qt works at all.
"""
import sys
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()
window.setWindowTitle("Qt Test")
window.resize(400, 300)
label = QtWidgets.QLabel("If you see this, PyQt5 is working!", window)
label.setGeometry(50, 100, 300, 50)
window.show()
print("Qt window shown. Close the window to exit.")
sys.exit(app.exec_())
