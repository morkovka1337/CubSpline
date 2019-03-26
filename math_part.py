import math
import numpy as np
from numpy import float64
from matplotlib.figure import Figure
from interface import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt


class mathpart(Ui_MainWindow):
    def calculate_spline(self, n):
        def function_F(x):
            if self.comboBox.currentText() == "Тестовая":
                return x**3 + 3*x**2 if x <= 0 else -x**3 + 3*x**2
            elif self.comboBox.currentText() == "sqrt(1+x^4)":
                return math.sqrt(1+x**4)
            elif self.comboBox.currentText() == "sqrt(1+x^4) + cos(10x)":
                return math.sqrt(1+x**4) + math.cos(10*x)
            elif self.comboBox.currentText() == "sqrt(1+x^4) + cos(100x)":
                return math.sqrt(1+x**4) + math.cos(100*x)

        def derivative_F(x):
            if self.comboBox.currentText() == "Тестовая":
                return 3*x**2 + 6*x if x <= 0 else -3*x**2 + 6*x
            elif self.comboBox.currentText() == "sqrt(1+x^4)":
                return 2*x**3/math.sqrt(1+x**4)
            elif self.comboBox.currentText() == "sqrt(1+x^4) + cos(10x)":
                return 2*x**3/math.sqrt(1+x**4) - 10*math.sin(10*x)
            elif self.comboBox.currentText() == "sqrt(1+x^4) + cos(100x)":
                return 2*x**3/math.sqrt(1+x**4) - 100*math.sin(100*x)

        def second_der_F(x):
            if self.comboBox.currentText() == "Тестовая":
                return 6*x+6 if x <= 0 else -6*x + 6
            elif self.comboBox.currentText() == "sqrt(1+x^4)":
                return 2*x**2*(x**4+3)/((1+x**4)*math.sqrt(1+x**4))
            elif self.comboBox.currentText() == "sqrt(1+x^4) + cos(10x)":
                return 2*x**2*(x**4+3)/((1+x**4)*math.sqrt(1+x**4))  - 100*math.cos(10*x)
            elif self.comboBox.currentText() == "sqrt(1+x^4) + cos(100x)":
                return 2*x**2*(x**4+3)/((1+x**4)*math.sqrt(1+x**4))  - 10000*math.cos(100*x)
        left = -1 if self.comboBox.currentText() == "Тестовая" else 0
        right = 1
        h = (right-left)/n 
        c = np.zeros(n+1, float64)
        g = np.zeros(n+1, float64)
        C = np.zeros(n-1, float64)
        A = np.zeros(n-1, float64)
        B = np.zeros(n-1, float64)
        for i in range(1, n):  # последний индекс - n-1
            A[i-1] = 1
            C[i-1] = -4
            B[i-1] = 1
        if self.comboBox_2.currentText() == "Совпадение 2 произв.":
            g[0] = second_der_F(left)
        elif self.comboBox_2.currentText() == "ЕГУ":
            g[0] = 0
        f = np.zeros(n+1, float64)
        for i in range(0, n+1):
            f[i] = function_F(left + i*h)
        for i in range(1, n):
            g[i] = -6*(f[i-1] - 2*f[i] + f[i+1])/(h**2)
        if self.comboBox_2.currentText() == "Совпадение 2 произв.":
            g[0] = second_der_F(right)
        elif self.comboBox_2.currentText() == "ЕГУ":
            g[0] = 0

        c = TDMASolve(A, C, B, g)
        b = np.zeros(n, float64)
        d = np.zeros(n, float64)
        a = np.zeros(n, float64)
        
        for i in range(0, n):
            d[i] = (c[i+1] - c[i])/h
            a[i] = f[i+1]
            b[i] = (f[i+1] - f[i])/h + (2*c[i+1] + c[i]) * h / 6
        c = c[1:]
        self.tableWidget.setRowCount(n)
        for i in range(0, n):
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.tableWidget.setItem(
                i, 1, QtWidgets.QTableWidgetItem(str(left + i*h)))
            self.tableWidget.setItem(
                i, 2, QtWidgets.QTableWidgetItem(str(left + (i+1)*h)))
            self.tableWidget.setItem(
                i, 3, QtWidgets.QTableWidgetItem(str(a[i])))
            self.tableWidget.setItem(
                i, 4, QtWidgets.QTableWidgetItem(str(b[i])))
            self.tableWidget.setItem(
                i, 5, QtWidgets.QTableWidgetItem(str(c[i])))
            self.tableWidget.setItem(
                i, 6, QtWidgets.QTableWidgetItem(str(d[i])))

        plt.subplot(111)
        x = left
        y = np.zeros(4*n, float64)
        for i in range(0, n):
            x_i = left+(i+1)*h
            y[4*i] = (a[i] + b[i] * (x-x_i) + c[i]*((x-x_i)**2)/2 + d[i]*((x-x_i)**3)/6)
            x += h/4
            y[4*i+1] =  (a[i] + b[i] * (x-x_i) + c[i]*((x-x_i)**2)/2 + d[i]*((x-x_i)**3)/6)
            x += h/4
            y[4*i+2] =  (a[i] + b[i] * (x-x_i) + c[i]*((x-x_i)**2)/2 + d[i]*((x-x_i)**3)/6)
            x += h/4
            y[4*i+3] =  (a[i] + b[i] * (x-x_i) + c[i]*((x-x_i)**2)/2 + d[i]*((x-x_i)**3)/6)
            x += h/4
        plt.plot(y)

        x = left
        y_der = []
        for i in range(0, n):
            x_i = left+(i+1)*h
            y_der.append(b[i] + c[i]*(x-x_i) + d[i]/2*(x-x_i)**2)
            x += h/4
            y_der.append(b[i] + c[i]*(x-x_i) + d[i]/2*(x-x_i)**2)
            x += h/4
            y_der.append(b[i] + c[i]*(x-x_i) + d[i]/2*(x-x_i)**2)
            x += h/4
            y_der.append(b[i] + c[i]*(x-x_i) + d[i]/2*(x-x_i)**2)
            x += h/4
        plt.plot(y_der)
        x = left
        y_sec_der = []
        for i in range(0, n):
            x_i = left +(i+1)*h
            y_sec_der.append(c[i] + d[i]*(x-x_i))
            x += h/4
            y_sec_der.append(c[i] + d[i]*(x-x_i))
            x += h/4
            y_sec_der.append(c[i] + d[i]*(x-x_i))
            x += h/4
            y_sec_der.append(c[i] + d[i]*(x-x_i))
            x += h/4
        plt.plot(y_sec_der)

        N = 4*n
        h = (right-left)/(N)

        f = np.zeros(N, float64)
        for i in range(N):
            f[i] = function_F(left + i*h)
        f_der = np.zeros(N, float64)
        for i in range(N):
            f_der[i] = derivative_F(left + i*h)
        f_sec_der = np.zeros(N, float64)
        for i in range(N):
            f_sec_der[i] = second_der_F(left + i*h)
        plt.plot(f)
        plt.plot(f_der)
        plt.plot(f_sec_der)
        plt.legend(("Сплайн", "1 производная", "2 производная", "Функция", "1 производная", "2 производная"))
        plt.show()



        y_der = np.array(y_der)
        y_sec_der = np.array(y_sec_der)

        self.label.setText(QtCore.QCoreApplication.translate(
            "MainWindow",
            "Справка \nСетка сплайна: n = «" + str(n) + "» \n" +
            "Контрольная сетка: N = «" + str(N) + "» \n" +
            "Погрешность сплайна на контрольной сетке \n" +
            "max F(x) - S(x) = " + str(max(abs(y-f))) + " при x = " + str(-1 + np.argmax(abs(y-f))*h) + "\n" +
            "Погрешность производной на контрольной сетке \n" +
            "max F'(x) - S'(x) = " + str(max(abs(y_der-f_der))) + " при x = " + str(-1 + np.argmax(abs(y_der-f_der))*h) + "\n" +
            "Погрешность второй производной на контрольной сетке \n" +
            "max F''(x) - S''(x) = " + str(max(abs(y_sec_der-f_sec_der))) + " при x = " + str(-1 + np.argmax(abs(y_sec_der-f_sec_der))*h)))
        self.tableWidget_2.setRowCount(N)
        for i in range(0, N):
            self.tableWidget_2.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.tableWidget_2.setItem(
                i, 1, QtWidgets.QTableWidgetItem(str(left + i*h)))
            self.tableWidget_2.setItem(
                i, 2, QtWidgets.QTableWidgetItem(str(f[i])))
            self.tableWidget_2.setItem(
                i, 3, QtWidgets.QTableWidgetItem(str(y[i])))
            self.tableWidget_2.setItem(
                i, 4, QtWidgets.QTableWidgetItem(str(f[i] - y[i])))
            self.tableWidget_2.setItem(
                i, 5, QtWidgets.QTableWidgetItem(str(f_der[i])))
            self.tableWidget_2.setItem(
                i, 6, QtWidgets.QTableWidgetItem(str(y_der[i])))
            self.tableWidget_2.setItem(
                i, 7, QtWidgets.QTableWidgetItem(str(f_der[i] - y_der[i])))
        
        plt.subplot(111)
        plt.plot(abs(y-f))
        plt.plot(abs(y_der-f_der))
        plt.plot(abs(y_sec_der-f_sec_der))
        plt.legend(("погрешность сплайна",
                    "погрешность 1 производной", "погрешность 2 производной"))
        plt.show()

def TDMASolve(a, c, b, d):
    n = len(d)-1
    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n+1)
    alpha[0] = 0 # alpha[0] = kapa1, kapa1 в этой задаче всегда 0
    beta[0] = d[0] # beta[0] = mu[1]
    for i in range(0, n-1):
        alpha[i+1] = b[i]/(c[i]-alpha[i]*a[i])
        beta[i+1] = (d[i+1] + beta[i] * a[i])/(c[i] - alpha[i] * a[i])

    x[-1] = d[-1]
    for i in range (n-1, 0, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
    x[0] = alpha[0] * x[1] + beta[0]
    return x
