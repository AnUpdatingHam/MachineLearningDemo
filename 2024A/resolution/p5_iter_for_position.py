import matplotlib.pyplot as plt

from config_5 import *  # 导入配置参数
from utils import *     # 导入工具函数
from utils_3 import plot_by_ids  # 导入绘图函数
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
        lb = math.floor(ids[j] - BIN_L_BOUND_BIAS[j])
        rb = math.floor(ids[j] - BIN_R_BOUND_BIAS[j])
        id_res = binary_search_for_index(lb, rb, j, ids, it_x, it_y)
        ids[j + 1] = id_res
        print("target:", DRAGON_SPACE[j])
        print("dis:", cal_distance(it_x[ids[j]], it_x[ids[j+1]], it_y[ids[j]], it_y[ids[j+1]]))
    return ids

# 判断在给定速度下是否发生碰撞
def judge_velocity(head_velocity, iter_range, mov_ids):
    for i in range(1, iter_range - 1):
        p_ids_200l, p_ids_200r = mov_ids[i - 1], mov_ids[i + 1]
        dif4 = (p_ids_200r - p_ids_200l)
        t4 = 400 / STEP_SIZE / head_velocity
        v4 = dif4 / t4 * ITER_DISTANCE
        max_value, max_index = np.amax(v4), np.argmax(v4)
        max_id = mov_ids[i][max_index]
        if max_value > 2.0:
            plt.scatter(limit_iter_x[max_id], limit_iter_y[max_id], c='y')
            plot_by_ids(mov_ids[i], True, limit_iter_x, limit_iter_y)
            return False
    return True

# 二分搜索寻找合适的速度
def binary_search_for_velocity(left, right, iter_range):
    mid = (left + right) / 2
    if right - left < EPS:
        return mid
    res_idx = np.full((iter_range, N + 1), -1, dtype=int)
    for i in range(iter_range):
        cur_head_id = limit_enter_id + 200 * i
        get_dragon_dots_position_ids(cur_head_id, res_idx[i], limit_iter_x, limit_iter_y)
    if judge_velocity(mid, iter_range, res_idx):
        return binary_search_for_velocity(mid, right, iter_range)
    else:
        return binary_search_for_velocity(left, mid, iter_range)

# 主函数
def main():
    ITER_RANGE = 601
    ans = binary_search_for_velocity(0.01, 2.0, ITER_RANGE)
    print("ans:", ans)

# 调用主函数
main()
