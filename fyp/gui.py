from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QWidget, QApplication, QMainWindow, QFileDialog, QLineEdit, QVBoxLayout, QPushButton, QTextEdit, QTableView, QTabWidget)
from PyQt5.QtCore import QAbstractTableModel, Qt, QCoreApplication


import pandas as pd
import numpy as np
import random
import operator
import math
from matplotlib import pyplot as plt

from FCM import *
from KM import *
from HC import *
from ENS import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        width, height = 641, 661

        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(width,height)
        MainWindow.setStyleSheet("background-color: rgb(162, 191, 222);")
        MainWindow.setWindowTitle("FIT3162")


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 651, 691))
        self.tabWidget.setStyleSheet("background-color: rgb(162, 191, 222);\n"
        "")
        self.tabWidget.setObjectName("tabWidget")
        
        
        
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        

        #LABELS
        
        self.l1 = QtWidgets.QLabel(self.tab)
        self.l1.setGeometry(QtCore.QRect(30, 15, 101, 21))
        self.l1.setStyleSheet("font: 11pt \"DejaVu Sans\";")
        self.l1.setObjectName("l1")

        self.l2 = QtWidgets.QLabel(self.tab)
        self.l2.setGeometry(QtCore.QRect(30, 80, 111, 21))
        self.l2.setStyleSheet("font: 11pt \"DejaVu Sans\";")
        self.l2.setObjectName("l2")
        
        
        #BROWSE
        self.b1 = QtWidgets.QPushButton(self.tab)
        self.b1.setGeometry(QtCore.QRect(430, 40, 83, 31))
        self.b1.setObjectName("b1")
        self.file_flag = False
        self.b1.clicked.connect(self.browse_csv)


        #LineEdit Filepath
        self.b2 = QtWidgets.QLineEdit(self.tab)
        self.b2.setGeometry(QtCore.QRect(30, 40, 381, 31))
        self.b2.setStyleSheet("background-color: rgb(203, 217, 232);")
        self.b2.setObjectName("b2")

        #Run
        self.b3 = QtWidgets.QPushButton(self.tab)
        self.b3.setGeometry(QtCore.QRect(420, 440, 83, 31))
        self.b3.setObjectName("b3")
        self.b3.clicked.connect(self.run)

        #FCM Radio Button
        self.b4 = QtWidgets.QRadioButton(self.tab)
        self.b4.setGeometry(QtCore.QRect(30, 370, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.b4.setFont(font)
        self.b4.setStyleSheet("QRadioButton::indicator {\n"
        "    width:                  15px;\n"
        "    height:                 15px;\n"
        "    border-radius:          7px;\n"
        "\n"
        "}\n"
        "QRadioButton::indicator:checked {\n"
        "    background-color:        rgb(0, 120, 249);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}\n"
        "QRadioButton::indicator:unchecked {\n"
        "    background-color:       rgb(203, 217, 232);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}")
        self.b4.setObjectName("b4")

        self.fcm_flag = False
        self.b4.clicked.connect(self.fcm_toggle)
        

        #KM Radio Button
        self.b5 = QtWidgets.QRadioButton(self.tab)
        self.b5.setGeometry(QtCore.QRect(120, 370, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.b5.setFont(font)
        self.b5.setStyleSheet("QRadioButton::indicator {\n"
        "    width:                  15px;\n"
        "    height:                 15px;\n"
        "    border-radius:          7px;\n"
        "\n"
        "}\n"
        "\n"
        "QRadioButton::indicator:checked {\n"
        "    background-color:        rgb(0, 120, 249);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}\n"
        "\n"
        "QRadioButton::indicator:unchecked {\n"
        "    background-color:       rgb(203, 217, 232);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}")
        self.b5.setObjectName("b5")

        self.km_flag = False
        self.b5.clicked.connect(self.km_toggle)
        

        #HC Radio Button
        self.b6 = QtWidgets.QRadioButton(self.tab)
        self.b6.setGeometry(QtCore.QRect(210, 370, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.b6.setFont(font)
        self.b6.setStyleSheet("QRadioButton::indicator {\n"
        "    width:                  15px;\n"
        "    height:                 15px;\n"
        "    border-radius:          7px;\n"
        "\n"
        "}\n"
        "\n"
        "QRadioButton::indicator:checked {\n"
        "    background-color:        rgb(0, 120, 249);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}\n"
        "\n"
        "QRadioButton::indicator:unchecked {\n"
        "    background-color:       rgb(203, 217, 232);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}")
        self.b6.setObjectName("b6")

        self.hc_flag = False
        self.b6.clicked.connect(self.hc_toggle)
        

        #Ensemble Radio Button
        self.b7 = QtWidgets.QRadioButton(self.tab)
        self.b7.setGeometry(QtCore.QRect(290, 370, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.b7.setFont(font)
        self.b7.setStyleSheet("QRadioButton::indicator {\n"
        "    width:                  15px;\n"
        "    height:                 15px;\n"
        "    border-radius:          7px;\n"
        "\n"
        "}\n"
        "\n"
        "QRadioButton::indicator:checked {\n"
        "    background-color:        rgb(0, 120, 249);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}\n"
        "\n"
        "QRadioButton::indicator:unchecked {\n"
        "    background-color:       rgb(203, 217, 232);\n"
        "    border:                 0.5px solid rgb(136, 138, 133);\n"
        "}")
        self.b7.setObjectName("b7")
        
        self.ens_flag = False
        self.b7.clicked.connect(self.ens_toggle)



        #Table
        self.table = QtWidgets.QTableView(self.tab)
        self.table.setGeometry(QtCore.QRect(30, 110, 571, 241))
        self.table.setStyleSheet("background-color: rgb(203, 217, 232);")
        self.table.setObjectName("table")
        
        #TextEditR
        self.texteditR = QtWidgets.QTextEdit(self.tab)
        self.texteditR.setGeometry(QtCore.QRect(30, 440, 371, 141))
        self.texteditR.setStyleSheet("background-color: rgb(203, 217, 232);")
        self.texteditR.setObjectName("texteditR")
        


        self.l3 = QtWidgets.QLabel(self.tab)
        self.l3.setGeometry(QtCore.QRect(30, 410, 111, 21))
        self.l3.setStyleSheet("font: 11pt \"DejaVu Sans\";")
        self.l3.setObjectName("l3")
        

        self.tabWidget.addTab(self.tab, "")
        

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        

        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 641, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        #MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.l1.setText(_translate("MainWindow", "CSV Filepath"))
        self.b1.setText(_translate("MainWindow", "Browse"))
        self.l2.setText(_translate("MainWindow", "Dataset Table"))
        self.b4.setText(_translate("MainWindow", "FCM"))
        self.b5.setText(_translate("MainWindow", "KM"))
        self.b6.setText(_translate("MainWindow", "HC"))
        self.b7.setText(_translate("MainWindow", "Ensemble"))
        self.l3.setText(_translate("MainWindow", "Message(s)"))
        self.b3.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Main"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Other"))


    ######BUTTON FUNCTIONS#######

    def browse_csv(self):
        self.filename = QFileDialog.getOpenFileName(MainWindow,'Open CSV','/home', 'CSV files (*.csv)')

        if self.filename[0]:

            self.file_flag = True
            self.b2.setText(self.filename[0])
            #print(self.filename)
            
            df_full = pd.read_csv(self.filename[0],header=None)
            columns = list(df_full.columns)
            features = columns[:len(columns)]
            df = df_full[features]
            
            m = dfTable(df)
            self.table.setModel(m)


    def reset_toggle_flags(self):
        self.fcm_flag = False
        self.km_flag = False
        self.hc_flag = False
        self.ens_flag = False
        #print(self.filename)
        #print(self.fcm_flag, self.km_flag, self.hc_flag, self.ens_flag)
        
    def fcm_toggle(self):
        self.reset_toggle_flags()
        self.fcm_flag = True

    def km_toggle(self):
        self.reset_toggle_flags()
        self.km_flag = True

    def hc_toggle(self):
        self.reset_toggle_flags()
        self.hc_flag = True

    def ens_toggle(self):
        self.reset_toggle_flags()
        self.ens_flag = True

    def run(self):
        if self.fcm_flag and self.file_flag:
            self.texteditR.append("Running...\n")
            QCoreApplication.processEvents()

            a, p, r = FCM(self.filename[0])
            
            output_str = ""
            output_str += "Accuracy: " + str(a[0]) + "\n"
            output_str += "Precision: " + str(p[0]) + "\n"
            output_str += "Recall: " + str(r[0]) + "\n"
            
            self.texteditR.append(output_str)
            self.texteditR.append("Done!")



class dfTable(QAbstractTableModel):
    def __init__(self, df):
        QAbstractTableModel.__init__(self)
        self._data = df

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
