import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



a = 3  # 螺线起始半径
b = 0.1  # 控制螺线间距的参数
theta = np.linspace(0, 4 * np.pi, 100)  # 生成 100 个点的角度值

# 计算极坐标下的半径值
r = a + b * theta

# 将极坐标转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 计算原始点之间的距离
distances = [np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2) for i in range(len(x) - 1)]

# 确定目标等距离的值
target_distance = sum(distances) / len(distances)

# 调整点的位置
new_x = [x[0]]
new_y = [y[0]]
for i in range(1, len(x)):
    theta = np.arctan2(y[i] - new_y[-1], x[i] - new_x[-1])
    new_x.append(new_x[-1] + target_distance * np.cos(theta))
    new_y.append(new_y[-1] + target_distance * np.sin(theta))

# 创建图形和轴
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
line, = ax.plot(x, y, 'r-')  # 绘制螺线
point, = ax.plot([], [], 'bo')  # 初始化点的位置
ax.set_aspect('equal', adjustable='box')


# 初始化函数，设置点的初始位置
def init():
    point.set_data([], [])
    return point,

# 更新函数，移动点的位置
def update(frame):
    #point.set_xdata(x[:frame], y[:frame])
    point.set_xdata(new_x[:frame])
    point.set_ydata(new_y[:frame])
    return point,

# 创建动画
ani = FuncAnimation(fig=fig, func=update, frames=len(x), interval=30)
ani.save("test.gif", writer='pillow') # save a gif


for i in range(len(new_x) - 1):
    d_x = new_x[i + 1] - new_x[i]
    d_y = new_y[i + 1] - new_y[i]
    print("dis", d_x * d_x + d_y * d_y)

plt.show()


for i in range(len(x) - 1):
    d_x = new_x[i + 1] - new_x[i]
    d_y = new_y[i + 1] - new_y[i]
    print("dis", d_x * d_x + d_y * d_y)
