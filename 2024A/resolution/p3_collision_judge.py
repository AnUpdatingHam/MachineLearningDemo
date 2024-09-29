"""
p3_collision_judge.py
分析一组数据点，以判断在特定的运动轨迹中是否存在潜在的碰撞。
"""

from config_3 import *  # 导入配置参数
from utils import *     # 导入工具函数
from utils_3 import plot_by_ids  # 导入绘图函数
import numpy as np

# 定义一个函数，用于找到最有可能导致碰撞的数据点索引
def get_most_potential_id_for_collision(ids, it_x, it_y):
    ids = ids[ids > 0]  # 过滤掉非正数的ID
    if len(ids) < 1:
        return 0  # 如果没有有效的ID，返回0
    h0_id, h1_id = ids[0], ids[1]  # 获取前两个ID
    xh0, yh0 = it_x[h0_id], it_y[h0_id]  # 获取对应的坐标
    xh1, yh1 = it_x[h1_id], it_y[h1_id]
    s_vec = vec_norm(np.array([xh0 - xh1, yh0 - yh1]))  # 计算两个点的向量并归一化

    pre_dot = 1.0
    for i in range(1, len(ids) - 1):
        p_id = ids[i + 1]
        xp, yp = it_x[p_id], it_y[p_id]
        t_vec = vec_norm(np.array([xh0 - xp, yh0 - yp]))  # 计算两个点的向量并归一化
        cur_dot = np.dot(s_vec, t_vec)  # 计算两个向量的点积
        if 0 > cur_dot > pre_dot:
            return i  # 返回最有可能导致碰撞的点的索引
        pre_dot = cur_dot
    return 0  # 如果没有找到潜在的碰撞点，返回0

# 定义一个函数，用于判断是否存在碰撞
def judge_if_collision(ids, it_x, it_y):
    ids = ids[ids >= 0]  # 过滤掉负数的ID
    if len(ids) < 7:
        return False  # 如果ID数量不足7个，认为不会发生碰撞
    closest_id = get_most_potential_id_for_collision(ids, it_x, it_y)  # 获取最有可能导致碰撞的点的索引
    if closest_id == 0:
        return False  # 如果没有找到潜在的碰撞点，认为不会发生碰撞
    potential_queue = ids[closest_id - 1: closest_id + 2]  # 获取潜在碰撞点的ID集合

    h0_id, h1_id = ids[0], ids[1]
    xh0, yh0 = it_x[h0_id], it_y[h0_id]
    xh1, yh1 = it_x[h1_id], it_y[h1_id]
    head_hypotenuse_length = cal_distance(3.41 / 2, 0, medium_point_center_distance(xh0, xh1, yh0, yh1) + 0.15, 0)
    print("h0_id, h1_id:", h0_id, h1_id)
    print("closet_id", closest_id)
    print("head_hypotenuse_length:", head_hypotenuse_length)

    for i in range(len(potential_queue) - 1):
        p_id, q_id = potential_queue[i], potential_queue[i + 1]
        xp, yp = it_x[p_id], it_y[p_id]
        xq, yq = it_x[q_id], it_y[q_id]
        mx, my = (xp + xq) / 2, (yp + yq) / 2
        potential_length = medium_point_center_distance(xp, xq, yp, yq) - 0.15
        print("potential_length:", potential_length)
        if potential_length <= head_hypotenuse_length:
            return True  # 如果潜在的碰撞长度小于等于头部的斜边长度，则认为会发生碰撞
    return False

# 定义主函数
def main():
    if judge_if_collision(res_idx, iter_x, iter_y):
        print(f"螺距:{P}发生碰撞")
    else:
        print(f"螺距:{P}未发生碰撞")

    plot_by_ids(res_idx, True, iter_x, iter_y)

# 如果需要运行主函数，可以取消以下注释
# main()