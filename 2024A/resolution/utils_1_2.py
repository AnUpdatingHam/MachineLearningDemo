from config_1_2 import *  # 导入配置参数
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块

# 定义一个函数，用于根据给定的索引绘制数据点
def plot_by_ids(ids, shown):
    # 移除索引中的-1，-1通常表示无效或缺失的数据点
    ids = ids[ids != -1]
    # 根据索引获取x坐标
    iter_res_x = iter_x[ids]
    # 根据索引获取y坐标
    iter_res_y = iter_y[ids]

    # 绘制数据点的连线，颜色为绿色，线宽为6，透明度为0.3
    plt.plot(iter_res_x, iter_res_y, c='g', linewidth=6, alpha=0.3)
    # 绘制除第一个点外的每个数据点，颜色为黑色，点的大小为3
    plt.scatter(iter_res_x[1:], iter_res_y[1:], c='black', s=3)
    # 特别绘制第一个数据点，颜色为红色，使用方形标记（'s'）
    plt.scatter(iter_x[ids[0]], iter_y[ids[0]], c='r', marker='s')

    # 如果shown为True，则显示图形
    if shown:
        plt.axis("equal")  # 设置坐标轴比例相同，确保图形不会因为轴的不同比例而变形
        plt.plot(iter_x, iter_y, c='gray')  # 绘制整个数据集，颜色为灰色
        plt.show()  # 显示图形

# 这个函数可以在调用时传入一个索引数组和一个布尔值shown。
# 例如：plot_by_ids([0, 1, 2, 3], True)
# 这将绘制索引为0, 1, 2, 3的数据点，并显示整个数据集。

print(res_idx.shape)
plt.figure(figsize=(6, 6))
plt.plot(iter_x, iter_y)
plot_by_ids(res_idx[350], False)
plt.show()

