import matplotlib.pyplot as plt
from config_4 import *  # 导入配置参数
from utils import *     # 导入工具函数

# 生成螺旋轨迹的函数
def generate_spiral_track(pitch, plot_flag, sample_count):
    round_cnt = 4.5 / pitch + 12  # 计算旋转圈数
    theta = np.linspace(0, round_cnt * 2 * np.pi, sample_count)  # 角度范围
    r = theta * pitch / (2 * np.pi)  # 计算半径
    x = r * np.cos(theta)  # x坐标
    y = r * np.sin(theta)  # y坐标

    new_x = [x[0]]
    new_y = [y[0]]
    for i in range(1, len(x)):
        if cal_distance2(x[i], new_x[-1], y[i], new_y[-1]) >= ITER_DISTANCE2_THRESHOLD:
            new_x.append(x[i])
            new_y.append(y[i])

    iter_x = np.flip(new_x)  # 数据翻转
    iter_y = np.flip(new_y)
    r = math.sqrt(iter_x[0]**2 + iter_y[0]**2)  # 计算旋转半径
    sin_sita = iter_x[0] / r  # 计算旋转的正弦值
    cos_sita = iter_y[0] / r  # 计算旋转的余弦值

    iter_rotate_x = iter_x * cos_sita - iter_y * sin_sita  # 绕(0, 0)旋转
    iter_rotate_y = iter_x * sin_sita + iter_y * cos_sita

    if plot_flag:
        fig, ax = plt.subplots()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'等距螺线（螺距 = {pitch * 100} cm）')
        plt.grid(True)
        ax.plot(iter_rotate_x, iter_rotate_y, 'r-', linewidth=0.5)
        ax.plot(-iter_rotate_x, -iter_rotate_y, 'b-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        plt.scatter(x[0], y[0])
        plt.show()

    iter_rotate_x = np.concatenate([iter_rotate_x, -iter_rotate_x[::-1]])
    iter_rotate_y = np.concatenate([iter_rotate_y, -iter_rotate_y[::-1]])
    return iter_rotate_x, iter_rotate_y

# 生成双圆轨迹的函数
def generate_double_circles_track(p0_id, iter_dis, it_x, it_y, plot_flag):
    p0 = np.array([it_x[p0_id], it_y[p0_id]])
    p2 = -p0
    p1 = p2 * 2 / 3 + p0 / 3
    dir_vec = p0 - np.array([it_x[p0_id - 1], it_y[p0_id - 1]])
    x_half1, y_half1 = generate_half_circle_dots(p0, p1, dir_vec, iter_dis)
    x_half2, y_half2 = generate_half_circle_dots(p1, p2, -dir_vec, iter_dis)

    if plot_flag:
        plt.plot(x_half1, y_half1, 'g-')
        plt.plot(x_half2, y_half2, 'y-')

    x_half1 = np.concatenate([x_half1[::-1], x_half2])
    y_half1 = np.concatenate([y_half1[::-1], y_half2])

    return x_half1, y_half1

# 获取组合双圆轨迹的函数
def get_combined_double_circles_tracks(it_x, it_y, enter_id, plot_flag):
    if plot_flag:
        plt.figure(figsize=(6, 6))
    x_half, y_half = generate_double_circles_track(enter_id, ITER_DISTANCE, it_x, it_y, plot_flag)

    r2 = it_x[enter_id] ** 2 + it_y[enter_id] ** 2
    dis2 = it_x ** 2 + it_y ** 2
    it_x = it_x[dis2 >= r2]
    it_y = it_y[dis2 >= r2]

    midpoint = len(it_x) // 2
    enter_it_x, enter_it_y = it_x[:midpoint], it_y[:midpoint]
    depart_it_x, depart_it_y = it_x[midpoint:], it_y[midpoint:]

    if plot_flag:
        plt.scatter([it_x[enter_id]], [it_y[enter_id]], c='g', label='cut point')
        plt.plot(enter_it_x, enter_it_y, 'r')
        plt.plot(depart_it_x, depart_it_y, 'b')
        plt.show()

    it_x = np.concatenate([enter_it_x, x_half, depart_it_x])
    it_y = np.concatenate([enter_it_y, y_half, depart_it_y])
    return it_x, it_y, len(enter_it_x), len(enter_it_x) + len(x_half)

# 主函数1，用于生成螺旋轨迹并保存
def main1():
    iter_rotate_x, iter_rotate_y = generate_spiral_track(P, True, 200000000)
    np.save(f'../data/p4_track_x.npy', iter_rotate_x)
    np.save(f'../data/p4_track_y.npy', iter_rotate_y)

# 主函数2，用于生成双圆轨迹并保存
def main2():
    dis2 = iter_x**2 + iter_y**2
    dis2 -= ((2.86 * 1.5)**2)
    dis2 = np.fabs(dis2)
    enter_id = np.argmin(dis2)
    print("最小值的索引:", enter_id)
    print("最小值:", dis2[enter_id])
    new_iter_x, new_iter_y, enter_id, leave_id = get_combined_double_circles_tracks(iter_x, iter_y, enter_id, True)

    print("enter_id:", enter_id)
    print("leave_id:", leave_id)

    np.save(f"{DATA_DIR}/p4_limit_enter_id.npy", enter_id)
    np.save(f"{DATA_DIR}/p4_limit_leave_id.npy", leave_id)
    np.save(f"{DATA_DIR}/p4_limit_x.npy", new_iter_x)
    np.save(f"{DATA_DIR}/p4_limit_y.npy", new_iter_y)

    plt.figure(figsize=(6, 6))
    plt.plot(new_iter_x, new_iter_y)
    plt.show()

# 调用主函数2
# main2()