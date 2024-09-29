"""
p1_accuracy_check.py
第一问的精度检验脚本
计算一组点中相邻点之间的距离，然后分析这些点到特定轴的垂直距离的统计特性。
"""

# 导入配置和工具模块
from config_1_2 import *
from utils import *

# 从迭代数据中获取前1000个x和y坐标值
it_x = iter_x[:1000]
it_y = iter_y[:1000]

# 初始化结果列表，用于存储每对相邻点之间的距离
res = []

# 计算相邻点对之间的距离
# 循环遍历前999个点，因为我们需要与下一个点计算距离
for i in range(999):
    # 调用cal_distance函数计算两点之间的距离，并添加到结果列表中
    # 假设cal_distance函数接受四个参数：两个x坐标和两个y坐标
    res.append(cal_distance(it_x[i], it_x[i+1], it_y[i], it_y[i+1]))

# 将结果列表转换为NumPy数组，并计算每个距离与0.005的绝对差值
# 这可能表示计算每个点到某个参考线的垂直距离偏差
distances = np.abs(numpy.array(res) - 0.005)

# 计算均方差（MSE），它是衡量数据偏离平均值的常用指标
mse = np.mean(distances**2)
# 打印均方差结果
print("Mean Squared Error:", mse)

# 计算平均绝对偏差（MAD），它是测量数据平均偏离平均值的绝对值
mad = np.mean(np.abs(distances))
# 打印平均绝对偏差结果
print("Mean Absolute Deviation:", mad)

# 计算标准差，它是衡量数据分布的离散程度的一个指标
std_dev = np.sqrt(np.mean(distances**2))
# 打印标准差结果
print("Standard Deviation:", std_dev)

# 计算最大偏差，即所有距离中的最大值
max_deviation = np.max(distances)
# 打印最大偏差结果
print("Max Deviation:", max_deviation)