"""
p1_iter_for_position.py
根据龙头以及参数寻找后继把手位置。
代码中定义了几个函数，用于计算和可视化数据点的位置。
"""

import matplotlib.pyplot as plt
from config_1_2 import *  # 导入配置参数
from utils import *       # 导入工具函数
import math

# 定义一个二分搜索函数，用于找到与特定点距离在一定范围内的点的索引
def binary_search_for_index(left, right, ori_id, ids):
    if left == right:
        return left
    mid = math.floor((left + right) / 2)
    # 计算中点与目标点的距离
    if cal_distance2(iter_x[mid], iter_x[ids[ori_id]], iter_y[mid], iter_y[ids[ori_id]]) < DRAGON_SPACE2[ori_id] * 1.002:
        # 如果距离小于等于阈值，则在左半边继续搜索
        return binary_search_for_index(left, mid, ori_id, ids)
    # 否则在右半边继续搜索
    return binary_search_for_index(mid + 1, right, ori_id, ids)

# 定义一个函数，用于获取“龙”的点的位置ID
def get_dragon_dots_position_ids(head_id, ids):
    ids[0] = head_id  # 设置龙头的ID
    plt.scatter(iter_x[head_id], iter_y[head_id], c='red')  # 绘制龙头的位置

    for j in range(len(DRAGON_SPACE)):  # 遍历每个“龙”的部分
        # 寻找同一时刻下后继点的下标
        if ids[j] - BIN_L_BOUND_BIAS[j] < 0:
            break  # 如果超出范围，则停止搜索

        # 使用二分搜索找到后继点的索引
        id_res = binary_search_for_index(ids[j] - BIN_L_BOUND_BIAS[j], ids[j] - BIN_R_BOUND_BIAS[j], j, ids)
        ids[j + 1] = id_res
        plt.scatter(iter_x[ids[j + 1]], iter_y[ids[j + 1]], c="green", s=0.05)  # 绘制后继点的位置

        print("target:", DRAGON_SPACE[j])  # 打印目标距离
        print("dis:", cal_distance(iter_x[ids[j]], iter_x[ids[j+1]], iter_y[ids[j]], iter_y[ids[j+1]]))  # 打印实际距离

    return ids

# 定义一个函数，用于生成迭代索引
def generate_iter_idx(step_size):
    res_idx = np.full((4 * M + 800, N + 1), -1, dtype=int)  # 初始化结果索引数组

    for i in range(0, 4 * (M + 200)):  # 遍历步长
        head_id = math.floor(i * step_size)  # 计算龙头的索引
        if head_id >= len(iter_x):
            break  # 如果超出范围，则退出循环
        get_dragon_dots_position_ids(head_id, res_idx[i])  # 获取“龙”的点的位置ID
        print(i, res_idx[i][0])  # 打印当前步长和龙头的索引

    print("circulate exit")  # 循环结束
    plt.axis("equal")  # 设置坐标轴比例相同
    plt.plot(iter_x, iter_y)  # 绘制迭代点
    plt.show()  # 显示图形

    # 保存数组到.npy文件，下次使用直接读取即可
    np.save(f'{DATA_DIR}/res_idx_{step_size}.npy', res_idx)


# generate_iter_idx(50)  # 调用函数，设置步长为50

# main()  # 如果需要，可以调用主函数