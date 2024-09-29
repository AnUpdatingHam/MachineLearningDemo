import numpy as np
import matplotlib.pyplot as plt

# 螺距（单位：cm）
pitch = 55

# 角度范围
theta = np.linspace(0, 10 * 2 * np.pi, 5000)

# 计算 x 和 y 坐标
r = theta * pitch / (2 * np.pi)
x = r * np.cos(theta)
y = r * np.sin(theta)

# 确定目标等距离的值
target_distance = 50

# 调整点的位置
new_x = [x[0]]
new_y = [y[0]]
for i in range(1, len(x)):
    theta = np.arctan2(y[i] - new_y[-1], x[i] - new_x[-1])
    new_x.append(new_x[-1] + target_distance * np.cos(theta))
    new_y.append(new_y[-1] + target_distance * np.sin(theta))

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制等距螺线
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('等距螺线（螺距 = 55 cm）')
plt.grid(True)

# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
line, = ax.plot(x, y, 'r-')  # 绘制螺线
point, = ax.plot([], [], 'bo')  # 初始化点的位置
plt.scatter(new_x, new_y, c="blue", s=0.5)
ax.set_aspect('equal', adjustable='box')


plt.show()

for i in range(len(x) - 1):
    d_x = new_x[i + 1] - new_x[i]
    d_y = new_y[i + 1] - new_y[i]
    print("dis", d_x * d_x + d_y * d_y)
