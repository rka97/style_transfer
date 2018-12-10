import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

from UI import UI
from main import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class App():
 
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        
        self.ui = UI()
        self.ui.transfer_btn.clicked.connect(self.transfer)
        self.ui.original_browse_btn.clicked.connect(self.setOriginalImage)
        self.ui.stlye_browse_btn.clicked.connect(self.setStyleImage)
        self.content_image = -1
        self.style_image = -1
        
    def run(self):
        sys.exit(self.app.exec_())

    def transfer(self):
        if self.content_image == -1 or self.style_image == -1:
            return
        
        output_image = mainGui(self.content_image, self.style_image)
        width = self.ui.output_image_view.width()
        height = self.ui.output_image_view.height()
        image = QtGui.QPixmap(output_image)
        image = image.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        
        self.ui.output_image_view.setPixmap(image)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.ui.window, "QFileDialog.getOpenFileName()", "","Images (*.png *.xpm *.jpg *.jepg)", options=options)
        if fileName:
            return fileName
        else:
            return -1

    def setOriginalImage(self):
        image_file = self.openFileNameDialog()
        if image_file == -1:
            return
        
        width = self.ui.original_image_view.width()
        height = self.ui.original_image_view.height()
        image = QtGui.QPixmap(image_file)
        image = image.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        self.ui.original_image_view.setPixmap(image)
        self.content_image = image_file

    def setStyleImage(self):
        image_file = self.openFileNameDialog()
        if image_file == -1:
            return
        
        width = self.ui.style_image_view.width()
        height = self.ui.style_image_view.height()
        image = QtGui.QPixmap(image_file)
        image = image.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        
        self.ui.style_image_view.setPixmap(image)
        self.style_image = image_file

if __name__ == '__main__':
    app = App()
    app.run()
