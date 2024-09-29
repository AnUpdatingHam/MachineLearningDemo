import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1, 2], [2, 5]])
y = np.array([1, 2])

# 不计算截距
model = LinearRegression(fit_intercept=False)

try:
    model.fit(x, y)

    # 打印系数和截距
    print("系数:", model.coef_)
    print("截距:", model.intercept_)

    # 打印模型的得分
    print("模型得分（R^2）:", model.score(x, y))

    # 打印预测结果
    print("预测结果:", model.predict(x))

except np.linalg.LinAlgError as e:
    print("矩阵是奇异的，无法求解:", e)


