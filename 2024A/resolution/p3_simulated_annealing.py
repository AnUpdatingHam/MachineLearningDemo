"""
p3_simulated_annealing.py
模拟退火算法求解最小螺距
"""

import numpy as np
import pandas as pd
from config_3 import *
from p3_track_generator import generate_track
from p3_iter_for_position import get_dragon_dots_position_ids
from p3_collision_judge import judge_if_collision
from utils_3 import *
import random


def full_judgement_process(p):
    it_x, it_y = generate_track(p, False, 5000000)
    # 创建龙把手索引数组，-1代表该把手未出现在螺线中
    res_idx = np.full((N + 1), -1, dtype=int)
    # 设置龙头坐标
    head_id = len(it_x) - 1
    get_dragon_dots_position_ids(head_id, res_idx, it_x, it_y)
    print("res[:10]:", res_idx[:10])
    if judge_if_collision(res_idx, it_x, it_y):
        print(f"螺距:{p}发生碰撞, True")
        return True, res_idx, it_x, it_y
    else:
        print(f"螺距:{p}未发生碰撞, False")
    return False, res_idx, it_x, it_y


# def circulate_for_p():
#     for p in np.arange(R_BOUND, L_BOUND, -0.005):
#         judge_flag, res_idx, it_x, it_y = full_judgement_process(p)
#         if judge_flag:
#             plot_by_ids(res_idx, True, it_x, it_y)

def simulated_annealing():
    T = 1.0  # 初始温度
    delta = 0.95  # 变化率
    eps = 1e-3  # 出口阈值
    C = 10000  # 调整常数
    k = 1.0  # 计算是否接受更差解时的系数

    kT = k * T  # k与T的乘积

    L_BOUND, R_BOUND = 0.30, 0.60  # 定义域半径
    W = R_BOUND - L_BOUND  # 定义域宽

    # 初始解为定义域内的随机数
    p0 = random.uniform(R_BOUND, L_BOUND)
    flag0, res_idx0, it_x0, it_y0 = full_judgement_process(p0)
    # debug
    cnt = 0
    # 计算部分
    while T > eps:
        cnt = cnt + 1
        print("迭代次数:", cnt)
        p1 = p0 + T * 2 * random.uniform(-W, W)
        while p1 > R_BOUND or p1 < L_BOUND:
            # 确保x1落在定义域内
            p1 = p0 + T * 2 * random.uniform(-W, W)
            # print("p0:", p0)
            # print("p1:", p1)

        flag1, res_idx1, it_x1, it_y1 = full_judgement_process(p1)

        print("-math.fabs(p0 - p1):", -C * math.fabs(p0 - p1))
        print("math.exp(-1000 * math.fabs(p0 - p1) / kT):", math.exp(-C * math.fabs(p0 - p1) / kT))
        if flag0 and not flag1:  # 旧解无法成立，新解成立，新解更优，无条件接受
            p0 = p1
        elif not flag0 and not flag1 and p0 > p1:  # 旧解、新解均成立，新解螺距小，新解更优，无条件接受
            p0 = p1
        elif flag0 and flag1:  # 旧解、新解均无法成立，接受螺距更大的，螺距越大，越靠近解成立的临界点
            p0 = max(p0, p1)
        elif math.exp(-C * math.fabs(p0 - p1) / kT) > random.random():  # 概率接受更差解
            p0 = p1
        # 衰减温度
        T *= delta
        flag0, res_idx0, it_x0, it_y0 = flag1, res_idx1, it_x1, it_y1
    print("总迭代次数:", cnt)
    print("模拟退火算法计算得到的最小不发生碰撞螺距为:", p0)

def main():
    simulated_annealing()

main()

