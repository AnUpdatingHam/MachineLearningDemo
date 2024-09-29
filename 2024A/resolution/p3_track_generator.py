"""
p3_track_generator.py
生成一个螺旋轨迹，并对其进行处理以确保轨迹上相邻点之间的距离满足特定条件。
"""

import matplotlib.pyplot as plt
import config_3 as conf
from utils import *


# 根据螺距和生成
def generate_track(pitch, plot_flag, sample_count):
    # 角度范围
    round_cnt = 4.5 / pitch + 2
    theta = np.linspace(0, round_cnt * 2 * np.pi, sample_count)

    # 计算 x 和 y 坐标
    r = theta * pitch / (2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dis2 = x ** 2 + y ** 2
    x = x[dis2 >= 4.5 ** 2]
    y = y[dis2 >= 4.5 ** 2]

    # 调整点的位置
    print(x.shape)
    new_x = [x[0]]
    new_y = [y[0]]

    for i in range(1, len(x)):
        if cal_distance2(x[i], new_x[-1], y[i], new_y[-1]) >= conf.ITER_DISTANCE2_THRESHOLD:
            new_x.append(x[i])
            new_y.append(y[i])

    if plot_flag:
        # 创建图形和轴
        fig, ax = plt.subplots()

        # 绘制等距螺线
        # plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'等距螺线（螺距 = {pitch * 100} cm）')
        plt.grid(True)

        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-10, 10)
        line, = ax.plot(x, y, 'r-', linewidth=0.5)  # 绘制螺线
        # point, = ax.plot([], [], 'bo')  # 初始化点的位置
        # plt.scatter(new_x, new_y, c="blue", s=1)
        ax.set_aspect('equal', adjustable='box')
        plt.scatter(x[0], y[0])
        plt.show()

    # 灵敏度检验
    # for i in range(len(new_x) - 1):
    #     d_x = new_x[i + 1] - new_x[i]
    #     d_y = new_y[i + 1] - new_y[i]
    #     print("dis", d_x * d_x + d_y * d_y)

    # 数据翻转
    iter_x = np.flip(new_x)
    iter_y = np.flip(new_y)
    # 数据绕(0, 0)旋转，使曲线过(0, 4.5)点
    r = math.sqrt(iter_x[-1]**2 + iter_y[-1]**2)
    sin_sita = iter_x[len(iter_x) - 1] / r
    cos_sita = iter_y[len(iter_x) - 1] / r

    iter_rotate_x = iter_x * cos_sita - iter_y * sin_sita
    iter_rotate_y = iter_x * sin_sita + iter_y * cos_sita

    np.save(f'../data/p3_track_x_{pitch}.npy', iter_rotate_x)
    np.save(f'../data/p3_track_y_{pitch}.npy', iter_rotate_y)

    return iter_rotate_x, iter_rotate_y



