import matplotlib.pyplot as plt
import numpy as np

# 函数和导函数
func = lambda x : (x - 3.5) ** 2 - 4.5 * x + 10
grad = lambda x : 2 * (x - 3.5) - 4.5

fx = np.linspace(0, 11.5, 100)
fy = func(fx)
# 绘制函数轮廓
plt.plot(fx, fy)

# 学习率
eta = 5
# 目标精确度
precision = 0.0001
# 随机初始值
x = np.random.randint(0, 12, size=1)[0]
# 用于记录上一次的值
last_x = x + 0.1

update_cnt = 0
print("--------------------随机x:", x)

# 多次while循环，每次梯度下降，更新，记录上一次的值
while True:
    # 循环出口:变化率过小
    if np.abs(x - last_x) < precision:
        break
    update_cnt += 1
    last_x = x
    x -= eta * grad(x)
    print("---------------------更新后的x:", x)

    # 绘制更新后的连线
    plt.plot([last_x, x], [func(last_x), func(x)], 'ro-')  # 'ro-' 表示红色

print("更新次数:", update_cnt)

# 添加标题和轴标签
plt.title('Plot of Functions f(x) and g(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()




