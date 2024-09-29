"""
config_5.py
题5的配置类，
为数学建模或相关计算任务提供配置参数和数据。
文件中定义了一系列变量，这些变量通常用于控制算法的行为或直接参与计算。
"""

import numpy as np
import os

DATA_DIR = "../data"

# 螺距（单位：m）
P = 1.7

# 保留点间距离
ITER_DISTANCE = 0.005

# 二分边界
EPS = 1e-5

# 确定目标等距离的平方
ITER_DISTANCE2 = ITER_DISTANCE**2
ITER_DISTANCE2_THRESHOLD = ITER_DISTANCE2 * 0.9999

DRAGON_SPACE = np.full(223, 1.65)
DRAGON_SPACE[0] = 2.86

DRAGON_SPACE2 = DRAGON_SPACE**2

BIN_R_BOUND_BIAS = DRAGON_SPACE / ITER_DISTANCE
BIN_L_BOUND_BIAS = np.ceil(BIN_R_BOUND_BIAS * 1.58)
BIN_R_BOUND_BIAS = np.floor(BIN_R_BOUND_BIAS * 0.8)

N = len(DRAGON_SPACE)

STEP_SIZE = 1.0 / ITER_DISTANCE

# 读取.npy文件
if os.path.exists(f"{DATA_DIR}/p4_track_x.npy"):
    iter_x = np.load(f"{DATA_DIR}/p4_track_x.npy")

if os.path.exists(f"{DATA_DIR}/p4_track_y.npy"):
    iter_y = np.load(f"{DATA_DIR}/p4_track_y.npy")

if os.path.exists(f"{DATA_DIR}/p4_res_enter_idx.npy"):
    res_enter_idx = np.load(f"{DATA_DIR}/p4_res_enter_idx.npy").astype(np.int32)

if os.path.exists(f"{DATA_DIR}/p4_limit_x.npy"):
    limit_iter_x = np.load(f"{DATA_DIR}/p4_limit_x.npy")

if os.path.exists(f"{DATA_DIR}/p4_limit_y.npy"):
    limit_iter_y = np.load(f"{DATA_DIR}/p4_limit_y.npy")

if os.path.exists(f"{DATA_DIR}/p4_limit_enter_id.npy"):
    limit_enter_id = np.load(f"{DATA_DIR}/p4_limit_enter_id.npy")

if os.path.exists(f"{DATA_DIR}/p4_limit_leave_id.npy"):
    limit_leave_id = np.load(f"{DATA_DIR}/p4_limit_leave_id.npy")

if os.path.exists(f"{DATA_DIR}/p4_movement_res_idx.npy"):
    movement_res_idx = np.load(f"{DATA_DIR}/p4_movement_res_idx.npy")

