"""
config_3.py
题3的配置类，
为数学建模或相关计算任务提供配置参数和数据。
文件中定义了一系列变量，这些变量通常用于控制算法的行为或直接参与计算。
"""

import numpy as np
import os

DATA_DIR = "../data"  # 定义数据目录路径

# 定义螺距（单位：cm）
P = 0.4375

# 定义保留点间的距离
ITER_DISTANCE = 0.005

# 计算确定目标等距离的平方和阈值
ITER_DISTANCE2 = ITER_DISTANCE**2
ITER_DISTANCE2_THRESHOLD = ITER_DISTANCE2 * 0.999

# 定义“龙”的空间距离数组，除第一个元素外，其余元素都是1.65
DRAGON_SPACE = np.full(223, 1.65)
DRAGON_SPACE[0] = 2.86

# 计算“龙”的空间距离的平方
DRAGON_SPACE2 = DRAGON_SPACE**2

# 计算二分搜索的左右边界偏差
BIN_R_BOUND_BIAS = DRAGON_SPACE / ITER_DISTANCE
BIN_L_BOUND_BIAS = np.ceil(BIN_R_BOUND_BIAS * 1.58)
BIN_R_BOUND_BIAS = np.floor(BIN_R_BOUND_BIAS * 0.8)

# 定义N和M的值
N = len(DRAGON_SPACE)
M = 301

# 定义步长
STEP_SIZE = 1.0 / ITER_DISTANCE
VELOCITY_ANCHOR_GAP = 800  # 定义速度锚点间隔

# 读取.npy文件
if os.path.exists(f"{DATA_DIR}/p3_track_x_{P}.npy"):
    iter_x = np.load(f"{DATA_DIR}/p3_track_x_{P}.npy")  # 读取x坐标数据
if os.path.exists(f"{DATA_DIR}/p3_track_y_{P}.npy"):
    iter_y = np.load(f"{DATA_DIR}/p3_track_y_{P}.npy")  # 读取y坐标数据
if os.path.exists(f"{DATA_DIR}/p3_res_idx_{P}.npy"):
    res_idx = np.load(f"{DATA_DIR}/p3_res_idx_{P}.npy").astype(np.int32)  # 读取索引数据
