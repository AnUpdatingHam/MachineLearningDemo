"""
p1_iter_for_position.py
计算和分析一组数据点（板凳龙把手的坐标）的动态变化。
代码中定义了几个函数，用于处理数据点、计算速度和差异，以及将结果保存到文件中。
"""

# 导入所需的库
from config_1_2 import *  # 导入配置参数
from utils import *       # 导入工具函数
from utils_1_2 import plot_by_ids  # 导入绘图函数
from p1_iter_for_position import get_dragon_dots_position_ids  # 导入获取特定位置ID的函数

# 初始化左右ID数组，填充-1
left_ids = np.full((444, N + 1), -1, dtype=int)
right_ids = np.full((444, N + 1), -1, dtype=int)

# 定义一个ID变量，可能用于标识特定的数据点
ID = 400

# 定义计算“龙”速度的函数
def cal_dragon_speed(head_id, l_ids, r_ids):
    # 检查head_id是否在有效范围内
    if head_id < VELOCITY_ANCHOR_GAP or head_id + VELOCITY_ANCHOR_GAP >= len(iter_x):
        return
    # 计算左右ID的起始和结束位置
    l_id = head_id - VELOCITY_ANCHOR_GAP
    r_id = head_id + VELOCITY_ANCHOR_GAP
    print(head_id, l_id, r_id)
    # 获取左右ID对应的位置
    get_dragon_dots_position_ids(l_id, l_ids)
    get_dragon_dots_position_ids(r_id, r_ids)

    # 移除-1的值
    l_ids = l_ids[l_ids != -1]
    r_ids = r_ids[r_ids != -1]

    # 绘制左右ID对应的点
    plot_by_ids(l_ids, True)
    plot_by_ids(r_ids, True)

# 定义计算差异数量的函数
def cal_dif_cnt():
    # 初始化差异数组，填充-1
    res_dif = np.full((443, N), -1, dtype=int)
    # 计算每两个连续点之间的差异
    for i in range(len(res_idx) - 1):
        for j in range(N):
            if res_idx[i][j] == -1:
                break
            res_dif[i][j] = res_idx[i+1][j] - res_idx[i][j]
    # 保存差异数组到CSV文件
    numpy.savetxt("res_dif.csv", res_dif)
    # 打印部分差异数据
    print(res_dif[410:])

# 调用计算差异数量的函数
cal_dif_cnt()
