import matplotlib.pyplot as plt
import numpy as np


def dots(X, y):
    plt.scatter(X, y)


def line(w, b, style):
    x_line = np.linspace(0, 1, 10)  # 生成从0到1的10个点
    y_line = w * x_line + b  # 根据线性方程计算y值
    # 绘制直线
    plt.plot(x_line, y_line, style, label='Linear function: y = wx + b')


def show():
    plt.show()

