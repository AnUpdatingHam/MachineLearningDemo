import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 样本数、特征数
n, m = 100, 1

# 函数和导函数
func = lambda x : (x - 3.5) ** 2 - 4.5 * x + 10
grad = lambda x : 2 * (x - 3.5) - 4.5

X = np.linspace(0, 11.5, n)

# 训练目标y需要在f(x)的基础上添加噪声
y = func(X) + np.random.randn(n)

# 对X进行多项式升维
poly = PolynomialFeatures(degree=2, interaction_only=False)
X = X.reshape(-1, 1)
X_poly = poly.fit_transform(X)

# 数据切片，前80%作为训练数据
train_cnt = int(0.8 * n)
X_poly_train, y_train = X_poly[:train_cnt], y[:train_cnt]
X_poly_test, y_test = X_poly[train_cnt:], y[train_cnt:]

# 建模
model = LinearRegression(fit_intercept=True)
model.fit(X_poly_train, y_train)

# 不使用科学计数法打印
np.set_printoptions(suppress=True)

print("W:", model.coef_)
print("b:", model.intercept_)
print("多项式升维度模型得分 (越接近1，拟合度越高) :", model.score(X_poly_test, y_test))

# 可视化部分
plt.scatter(X, y)
# 根据自定义参数创建的函数
plt.plot(X, func(X), 'green', label="Created Function Curve")
# 拟合得到的图像
y_pred = model.predict(X_poly)
plt.plot(X_poly[:, 1], y_pred, 'red', label="Model Fitted Curve")

# 添加标题和轴标签
plt.title('Polynomial Regression Fitting Case')
plt.xlabel('x')
plt.ylabel('y')
# 添加图例
plt.legend()
plt.show()




