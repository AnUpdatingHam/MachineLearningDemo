from config_3 import *
import matplotlib.pyplot as plt

def plot_by_ids(ids, shown, it_x, it_y):
    ids = ids[ids != -1]
    it_res_x = it_x[ids]
    it_res_y = it_y[ids]
    # 绘制龙头
    # 绘制支撑点
    # 绘制板凳
    plt.plot(it_res_x, it_res_y, c='g', linewidth=5, alpha=0.5)
    plt.scatter(it_res_x[1:], it_res_y[1:], c='black', s=1)
    plt.scatter(it_x[ids[0]], it_y[ids[0]], c='r', marker='s')

    if shown:
        plt.axis("equal")
        plt.plot(it_x, it_y)
        plt.show()





# plt.axis("equal")
# plt.plot(iter_x, iter_y)
#
#
# radius = 8.8
# # 生成圆的参数方程
# theta = np.linspace(0, 2 * np.pi, 100)
# x = radius * np.cos(theta)
# y = radius * np.sin(theta)
#
# plt.plot(x, y, c='r')
#
#
# plt.show()
