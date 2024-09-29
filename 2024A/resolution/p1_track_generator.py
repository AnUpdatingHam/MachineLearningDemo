"""
p1_track_generator.py
生成一个螺旋轨迹，并根据特定的距离阈值调整轨迹上点的位置，以确保相邻点之间保持一定的距离。
"""

# 导入所需的库
import matplotlib.pyplot as plt
import config_1_2 as conf
from utils import *

# 定义角度范围，从0到16圈的2*pi倍
theta = np.linspace(0, 16 * 2 * np.pi, 300000000)

# 计算螺旋线上的x和y坐标
# conf.P是螺旋线的参数，代表螺距
r = theta * conf.P / (2 * np.pi)
x = r * np.cos(theta)
y = r * np.sin(theta)

# 打印配置文件中定义的迭代距离阈值
print(conf.ITER_DISTANCE2)

# 调整点的位置，确保相邻点之间的距离不小于配置文件中的阈值
new_x = [x[0]]
new_y = [y[0]]
for i in range(1, len(x)):
    if cal_distance2(x[i], new_x[-1], y[i], new_y[-1]) >= conf.ITER_DISTANCE2_THRESHOLD:
        new_x.append(x[i])
        new_y.append(y[i])

# 设置一个标志位，用于控制是否绘图
plt_flag = False
if plt_flag:
    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制等距螺线
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('等距螺线（螺距 = 55 cm）')
    plt.grid(True)

    # 以下两行被注释掉，如果需要可以取消注释来设置x和y轴的显示范围
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)

    # 绘制螺线，'r-'表示红色实线
    line, = ax.plot(x, y, 'r-')  
    # 初始化点的位置，'bo'表示蓝色圆点
    point, = ax.plot([], [], 'bo')  
    # 绘制满足条件的点
    plt.scatter(new_x, new_y, c="blue", s=1)
    # 设置坐标轴的长宽比相等，保持螺旋线的比例
    ax.set_aspect('equal', adjustable='box')

    # 显示图形
    plt.show()

# 打印相邻点之间的距离平方
for i in range(len(new_x) - 1):
    d_x = new_x[i + 1] - new_x[i]
    d_y = new_y[i + 1] - new_y[i]
    print("dis", d_x * d_x + d_y * d_y)

# 将满足条件的点的坐标保存到.npy文件中，以便下次直接读取
np.save('new_x.npy', new_x)
np.save('new_y.npy', new_y)