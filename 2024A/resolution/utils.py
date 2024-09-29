import math
import numpy
import pandas as pd
import numpy as np

# 计算两点之间的距离的平方
def cal_distance2(x1, x2, y1, y2):
    return (x2 - x1)**2 + (y2 - y1)**2

# 计算两点之间的欧几里得距离
def cal_distance(x1, x2, y1, y2):
    return math.sqrt(cal_distance2(x1, x2, y1, y2))

# 计算两点中点到圆心的距离的平方
def medium_point_center_distance2(x1, x2, y1, y2):
    x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
    return x_mid**2 + y_mid**2

# 计算两点中点到圆心的欧几里得距离
def medium_point_center_distance(x1, x2, y1, y2):
    return math.sqrt(medium_point_center_distance2(x1, x2, y1, y2))

# 向量归一化
def vec_norm(vec):
    norm = np.linalg.norm(vec)  # 计算向量的L2范数
    v_normalized = vec / norm  # 单位长度标准化
    return v_normalized

# 点绕中心旋转
def rotate_by_center_and_angle(x, y, c, sin_theta, cos_theta):
    x_prime = x - c[0]  # 平移：将点相对于旋转中心移动
    y_prime = y - c[1]
    x_rotated = x_prime * cos_theta - y_prime * sin_theta  # 旋转
    y_rotated = x_prime * sin_theta + y_prime * cos_theta
    x_rotated = x_rotated + c[0]  # 反平移：将点移回原来的位置
    y_rotated = y_rotated + c[1]
    return x_rotated, y_rotated

# 生成以点p0、p1为直径的半圆上的点集
def generate_half_circle_dots(p0, p1, dir_vec, iter_dis):
    c = (p0 + p1) / 2  # 中心点
    r = cal_distance(p1[0], c[0], p1[1], c[1])  # 半径
    dot_cnt = math.floor(math.pi * r / iter_dis)  # 点的数量

    sin_theta = (p1[1] - c[1]) / r  # 旋转的正弦值
    cos_theta = (p1[0] - c[0]) / r  # 旋转的余弦值

    angles = [math.pi * i / dot_cnt for i in range(dot_cnt)]  # 生成角度
    x = np.array([c[0] + r * math.cos(angle) for angle in angles])  # 生成圆上的x坐标
    y = np.array([c[1] + r * math.sin(angle) for angle in angles])  # 生成圆上的y坐标

    x_rotated, y_rotated = rotate_by_center_and_angle(x, y, c, sin_theta, cos_theta)  # 旋转点
    judge_vec = np.array([x_rotated[1] - x_rotated[0], y_rotated[1] - y_rotated[0]])  # 方向向量
    if dir_vec.dot(judge_vec) < 0:  # 半圆方向与目标方向不同，切换
        x_rotated = 2 * c[0] - x_rotated
        y_rotated = 2 * c[1] - y_rotated

    return x_rotated, y_rotated