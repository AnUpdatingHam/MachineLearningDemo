"""
p3_iter_for_position.py
根据龙头以及参数寻找后继把手位置。
代码中定义了几个函数，用于计算和可视化数据点的位置。
"""

import matplotlib.pyplot as plt
from config_3 import *  # 导入配置参数
from utils import *     # 导入工具函数
from utils_3 import plot_by_ids  # 导入绘图函数
import math

# 定义一个二分搜索函数，用于找到与特定点距离在一定范围内的点的索引
def binary_search_for_index(left, right, ori_id, ids, it_x, it_y):
    if left == right:
        return left
    mid = math.floor((left + right) / 2)
    # 如果中点与目标点的距离小于等于阈值，则在左半边继续搜索
    if cal_distance2(it_x[mid], it_x[ids[ori_id]], it_y[mid], it_y[ids[ori_id]]) < DRAGON_SPACE2[ori_id] * 1.002:
        return binary_search_for_index(left, mid, ori_id, ids, it_x, it_y)
    # 否则在右半边继续搜索
    return binary_search_for_index(mid + 1, right, ori_id, ids, it_x, it_y)

# 定义一个函数，用于获取“龙”的点的位置ID
def get_dragon_dots_position_ids(head_id, ids, it_x, it_y):
    ids[0] = head_id  # 设置龙头的ID
    # plt.scatter(it_x[head_id], it_y[head_id], c='red')  # 绘制龙头的位置（被注释掉）

    for j in range(len(DRAGON_SPACE)):  # 遍历每个“龙”的部分
        # 寻找同一时刻下后继点的下标
        if ids[j] - BIN_L_BOUND_BIAS[j] < 0:
            break  # 如果超出范围，则停止搜索

        # 使用二分搜索找到后继点的索引
        id_res = binary_search_for_index(ids[j] - BIN_L_BOUND_BIAS[j], ids[j] - BIN_R_BOUND_BIAS[j], j, ids, it_x, it_y)
        ids[j + 1] = id_res

    return ids

# 定义主函数
def main():
    print("iter_len:", len(iter_x))  # 打印迭代点的数量
    res_idx = np.full((N + 1), -1, dtype=int)  # 初始化结果索引数组
    # 设置龙头坐标
    head_id = len(iter_x) - 1
    get_dragon_dots_position_ids(head_id, res_idx, iter_x, iter_y)  # 获取“龙”的点的位置ID
    plt.axis("equal")  # 设置坐标轴比例相同
    plt.plot(iter_x, iter_y)  # 绘制迭代点
    plot_by_ids(res_idx, False, iter_x, iter_y)  # 绘制“龙”的点
    plt.show()  # 显示图形
    # 保存数组到.npy文件，下次使用直接读取即可（被注释掉）
    np.save(f'../data/p3_res_idx_{P}.npy', res_idx)


# main()