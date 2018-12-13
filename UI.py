# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'StyleTransfer.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication

class UI(object):
    def __init__(self, title="title", width=1061, height=636):
        self.window = QMainWindow()
        self.title = title
        self.width = width
        self.height = height

        self.setupUi()
        self.window.show()

    def setupUi(self):
        self.window.setWindowTitle(self.title)
        self.window.resize(self.width, self.height)
        
        self.centralwidget = QtWidgets.QWidget(self.window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.transfer_btn = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.transfer_btn.sizePolicy().hasHeightForWidth())
        self.transfer_btn.setSizePolicy(sizePolicy)
        self.transfer_btn.setMinimumSize(QtCore.QSize(0, 50))
        self.transfer_btn.setMaximumSize(QtCore.QSize(16777215, 50))
        self.transfer_btn.setObjectName("transfer_btn")
        self.gridLayout_2.addWidget(self.transfer_btn, 1, 0, 1, 1)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(3)
        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)

        self.original_browse_btn = QtWidgets.QPushButton(self.centralwidget)
        self.original_browse_btn.setObjectName("original_browse_btn")
        self.gridLayout.addWidget(self.original_browse_btn, 3, 0, 1, 1)
        self.stlye_browse_btn = QtWidgets.QPushButton(self.centralwidget)
        self.stlye_browse_btn.setObjectName("stlye_browse_btn")
        self.gridLayout.addWidget(self.stlye_browse_btn, 3, 1, 1, 1)

        self.output_image_view = QtWidgets.QLabel(self.centralwidget)
        self.output_image_view.setText("")
        self.output_image_view.setObjectName("output_image_view")
        self.gridLayout.addWidget(self.output_image_view, 1, 2, 2, 1)

        self.style_image_view = QtWidgets.QLabel(self.centralwidget)
        self.style_image_view.setText("")
        self.style_image_view.setObjectName("style_image_view")
        self.gridLayout.addWidget(self.style_image_view, 1, 1, 2, 1)

        self.original_image_view = QtWidgets.QLabel(self.centralwidget)
        self.original_image_view.setText("")
        self.original_image_view.setObjectName("original_image_view")
        self.gridLayout.addWidget(self.original_image_view, 1, 0, 2, 1)

        self.export_btn = QtWidgets.QPushButton(self.centralwidget)
        self.export_btn.setObjectName("export_btn")
        self.gridLayout.addWidget(self.export_btn, 3, 2, 1, 1)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self.window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1061, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self.window)
        self.statusbar.setObjectName("statusbar")
        self.window.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(self.window)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExit = QtWidgets.QAction(self.window)
        self.actionExit.setObjectName("actionExit")
        self.actionDocumentation = QtWidgets.QAction(self.window)
        self.actionDocumentation.setObjectName("actionDocumentation")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionDocumentation)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.window)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.window.setWindowTitle(_translate("self.title", self.title))
        self.transfer_btn.setText(_translate("self.window", "Transfer"))
        self.label.setText(_translate("self.window", "Original"))
        self.label_2.setText(_translate("self.window", "Output"))
        self.label_3.setText(_translate("self.window", "Style"))
        self.original_browse_btn.setText(_translate("self.window", "Browse"))
        self.stlye_browse_btn.setText(_translate("self.window", "Browse"))
        self.export_btn.setText(_translate("self.window", "Export"))
        self.menuFile.setTitle(_translate("self.window", "File"))
        self.menuHelp.setTitle(_translate("self.window", "Help"))
        self.actionOpen.setText(_translate("self.window", "Open"))
        self.actionExit.setText(_translate("self.window", "Exit"))
        self.actionDocumentation.setText(_translate("self.window", "Documentation"))

