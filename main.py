# -*- coding: utf-8 -*-
import sys
import math
import math_part
from interface import *
# Импортируем наш интерфейс из файла
from PyQt5.QtWidgets import QApplication, QMainWindow
from numpy import float64
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
class MyWin(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None, *args, **kwargs):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.MyFunction)
    def MyFunction(self):
        n = self.spinBox.value()
        
        math_part.mathpart.calculate_spline(self, n)
        

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass