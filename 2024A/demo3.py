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
theta = np.linspace(0, 10 * 2 * np.pi, 10000)

# 计算 x 和 y 坐标
r = theta * pitch / (2 * np.pi)
x = r * np.cos(theta)
y = r * np.sin(theta)

# 计算切割点的数量
num_cuts = int(np.floor(r[-1] * 2 * np.pi / cut_length))

# 计算切割点的角度
cut_theta = np.linspace(0, theta[-1], num_cuts + 1)

# 计算切割点的坐标
cut_x = cut_theta * pitch / (2 * np.pi) * np.cos(cut_theta)
cut_y = cut_theta * pitch / (2 * np.pi) * np.sin(cut_theta)

# 绘制等距螺线
plt.plot(x, y)

# 绘制切割点
plt.scatter(cut_x, cut_y, c='red', marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.title('等距螺线（螺距 = 55 cm）与切割点（每段 10 cm）')
plt.grid(True)
plt.show()

