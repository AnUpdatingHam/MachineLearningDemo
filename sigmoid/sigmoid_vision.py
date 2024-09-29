from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Z-Score归一化
from sklearn.preprocessing import scale, StandardScaler

X, y = datasets.load_breast_cancer(return_X_y=True)

print(X.shape, y.shape)

X = X[:, :2]  # 切片前两个特征，便于画图

X = scale(X)

model = LogisticRegression()
model.fit(X,y)

w1 = model.coef_[0, 0]
w2 = model.coef_[0, 1]
b = model.intercept_
print("方程系数:", w1, w2)
print("方程截距:", b)


def sigmoid(x, w1, w2, b):
    z = w1 * x[0] + w2 * x[1] + b
    return 1 / (1 + np.exp(-z))


def loss_function(X, y, w1, w2, b):
    loss = 0
    for x_i, y_i in zip(X, y):
        p = sigmoid(x_i, w1, w2, b)  # 计算概率
        p = np.clip(p, 0.00000001, 0.99999999)  # 裁剪p保证其不出现0
        loss += - (y_i * np.log(p) + (1 - y_i) * np.log(1 - p))
    return loss


# w1/w2的取值空间
w1_space = np.linspace(w1 - 2, w1 + 2, 100)
w2_space = np.linspace(w2 - 2, w2 + 2, 100)

loss1_ = np.array([loss_function(X, y, i, w2, b) for i in w1_space])
loss2_ = np.array([loss_function(X, y, w1, i, b) for i in w2_space])

figure1 = plt.figure(figsize=(12, 9))
plt.subplot(2, 2, 1)
plt.title("W1 Loss Function")
plt.plot(w1_space, loss1_, color='green', label='')

plt.subplot(2, 2, 2)
plt.title("W2 Loss Function")
plt.plot(w2_space, loss2_, color='red', label='')

# 等高线（w1/w2 同时绘制）
plt.subplot(2, 2, 3)
w1_grid, w2_grid = np.meshgrid(w1_space, w2_space)
loss_grid = loss_function(X, y, w1_grid, w2_grid, b)
plt.contour(w1_grid, w2_grid, loss_grid, color='purple')

# 登高面
plt.subplot(2, 2, 4)
plt.contourf(w1_grid, w2_grid, loss_grid, color='purple')


# 3D可视化
figure2 = plt.figure(figsize=(12, 9))
ax = figure2.add_axes(Axes3D(figure2))
ax.plot_surface(w1_grid, w2_grid, loss_grid, cmap='viridis')

plt.xlabel('w1')
plt.ylabel('w2')

ax.view_init(30, -30)

plt.show()





