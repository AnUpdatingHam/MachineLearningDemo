import numpy as np
import matplotlib.pyplot as plt

# 螺距（单位：cm）
pitch = 55

# 转换为米
pitch = pitch / 100

# 每段长度（单位：cm）
cut_length = 10

# 转换为米
cut_length = cut_length / 100

# 角度范围
theta = np.linspace(0, 10 * 2 * np.pi, 100000)

# 计算 x 和 y 坐标
r = theta * pitch / (2 * np.pi)
x = r * np.cos(theta)
y = r * np.sin(theta)

# 计算相邻点之间的距离
distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

# 累计距离
cumulative_distances = np.cumsum(distances)

# 找到切割点的索引
cut_indices = np.where(cumulative_distances >= cut_length)[0]

# 添加初始点的索引
cut_indices = np.insert(cut_indices, 0, 0)

# 计算切割点的坐标
cut_x = x[cut_indices]
cut_y = y[cut_indices]

# 绘制等距螺线
plt.plot(x, y)

# 绘制切割点
plt.scatter(cut_x, cut_y, c='red', marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.title('等距螺线（螺距 = 55 cm）与切割点（每段 10 cm）')
plt.grid(True)
plt.show()

for i in range(len(x) - 1):
    d_x = cut_x[i + 1] - cut_x[i]
    d_y = cut_y[i + 1] - cut_y[i]
    print("dis", d_x * d_x + d_y * d_y)
