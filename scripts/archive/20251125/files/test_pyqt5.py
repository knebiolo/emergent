"""
Minimal PyQt5 test - just open a window with a button
"""
import sys
from PyQt5 import QtWidgets

def main():
    print("Creating Qt Application...")
    app = QtWidgets.QApplication(sys.argv)
    
    print("Creating window...")
    window = QtWidgets.QWidget()
    window.setWindowTitle("PyQt5 Test")
    window.setGeometry(100, 100, 400, 200)
    
    layout = QtWidgets.QVBoxLayout()
    label = QtWidgets.QLabel("If you see this, PyQt5 works!")
    button = QtWidgets.QPushButton("Click Me")
    button.clicked.connect(lambda: print("Button clicked!"))
    
    layout.addWidget(label)
    layout.addWidget(button)
    window.setLayout(layout)
    
    print("Showing window...")
    window.show()
    
    print("Starting event loop...")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
