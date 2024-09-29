"""
p4_iter_for_position.py
根据龙头以及参数寻找后继把手位置.
"""


from config_4 import *  # 导入配置参数
from utils import *  # 导入工具函数
import math


# 二分搜索函数，用于找到与特定点距离在一定范围内的点的索引
def binary_search_for_index(left, right, ori_id, ids, it_x, it_y):
    if left == right:
        return left
    mid = math.floor((left + right) / 2)
    if cal_distance2(it_x[mid], it_x[ids[ori_id]], it_y[mid], it_y[ids[ori_id]]) < DRAGON_SPACE2[ori_id] * 1.002:
        return binary_search_for_index(left, mid, ori_id, ids, it_x, it_y)
    return binary_search_for_index(mid + 1, right, ori_id, ids, it_x, it_y)


# 获取“龙”的点的位置ID
def get_dragon_dots_position_ids(head_id, ids, it_x, it_y):
    ids[0] = head_id
    for j in range(len(DRAGON_SPACE)):
        if ids[j] - BIN_L_BOUND_BIAS[j] < 0:
            break
        id_res = binary_search_for_index(ids[j] - BIN_L_BOUND_BIAS[j], ids[j] - BIN_R_BOUND_BIAS[j], j, ids, it_x, it_y)
        ids[j + 1] = id_res
        print("target:", DRAGON_SPACE[j])
        print("dis:", cal_distance(it_x[ids[j]], it_x[ids[j + 1]], it_y[ids[j]], it_y[ids[j + 1]]))

    return ids


# 主函数
def main():
    res_idx = np.full((211, N + 1), -1, dtype=int)
    for i in range(-100, 110):
        cur_head_id = limit_enter_id + 200 * i
        get_dragon_dots_position_ids(cur_head_id, res_idx[i + 100], limit_iter_x, limit_iter_y)

    # 保存数组到.npy文件，下次使用直接读取即可
    np.save(f'{DATA_DIR}/p4_movement_res_idx.npy', res_idx)

# 如果需要运行主函数，可以取消以下注释
# main()