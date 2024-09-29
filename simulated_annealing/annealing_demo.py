import math
import random
import numpy as np
import matplotlib.pyplot as plt

T = 1.0  # 初始温度
delta = 0.99  # 变化率
eps = 1e-3  # 出口阈值
k = 1.0  # 计算是否接受更差解时的系数

kT = k * T  # k与T的乘积

W = 2  # 定义域半径（-W, W）

# 函数f(x)
f = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)

# 初始解为定义域内的随机数
x0 = random.uniform(-W, W)
# 计算部分
while T > eps:
    x1 = x0 + T * 2 * random.uniform(-W, W)
    while x1 > W or x1 < -W:
        # 确保x1落在定义域内
        x1 = x0 + T * 2 * random.uniform(-W, W)

    f0 = f(x0)
    f1 = f(x1)
    if f1 > f0:  # 新解更优，无条件接受
        x0 = x1
    elif math.exp((f1 - f0) / kT) > random.random():  # 概率接受更差解
        x0 = x1
    T *= delta
    # 绘制点
    plt.scatter(x0, f(x0), c='r', s=10)  # 过程点为红色

# 绘图部分
X = np.linspace(-W, W, 100)
plt.plot(X, f(X), label="func")
plt.scatter(x0, f(x0), c='g', s=10)  # 最终点为绿色
plt.show()
