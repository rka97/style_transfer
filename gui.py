import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon, QImage
from PyQt5.QtCore import pyqtSlot

from UI import UI
from main import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class App():
 
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        
        self.ui = UI(title="Style Transfer")
        self.ui.transfer_btn.clicked.connect(self.transfer)
        self.ui.original_browse_btn.clicked.connect(self.setOriginalImage)
        self.ui.stlye_browse_btn.clicked.connect(self.setStyleImage)
        self.ui.export_btn.clicked.connect(self.export)
        self.content_image = -1
        self.style_image = -1
        self.output_image = False
        self.x = None

    def run(self):
        sys.exit(self.app.exec_())

    def transfer(self):
        if self.content_image == -1 or self.style_image == -1:
            return
        
        self.x = main_gui(self.content_image, self.style_image)
        self.output_image = True
        width = self.ui.output_image_view.width()
        height = self.ui.output_image_view.height()

        h, w, channel = self.x.shape
        bytes_per_line = 3 * w
        image = QImage(self.x.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QtGui.QPixmap(image)
        pix = pix.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        
        self.ui.output_image_view.setPixmap(pix)

    def export(self):
        filename = self.saveFileNameDialog()

        if not self.output_image or filename == -1:
            return
        
        io.imsave(filename, self.x)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.ui.window, "QFileDialog.getOpenFileName()", "","Images (*.png *.xpm *.jpg *.jepg)", options=options)
        if fileName:
            return fileName
        else:
            return -1

    def saveFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self.ui.window, "QFileDialog.getSaveFileName()", "","Images (*.png *.xpm *.jpg *.jepg)", options=options)
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
