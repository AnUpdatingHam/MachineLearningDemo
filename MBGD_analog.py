import matplotlib.pyplot as plt
import numpy as np
import plot_constant as plc

'''
小批量梯度下降（Mini-Batch Gradient Descent, MBGD）
'''

# 样本数，特征个数（引入偏置之前）
N, M = 128, 1

# 1.创建数据集X，y
X = np.random.rand(N, M)
w, b = np.random.randint(1, 10, size=2)
# 为y添加噪声
y = w * X + b + np.random.randn(N, M)

# 2. 绘制预设的回归线和点
plc.dots(X, y)
plc.line(w, b, 'g-')

# 3.在X中引入偏置项
X = np.concatenate((X, np.ones((N, 1))), axis=1)

# 4.学习率调整函数
t0, t1 = 5, 500
def learning_rate_schedule(t):
    return t0 / (t + t1)


# 5.创建超参数轮次、样本数量、小批量数量
epochs = 100
batch_size = 16
num_batches = int(N / batch_size)

# 6.初始化W0...Wn，标准正态分布
θ = np.random.randn(M + 1, 1)

# 多次循环实现梯度下降，最终结果收敛
for epoch in range(epochs):
    # 每个轮次开始分批迭代前打乱索引顺序
    index = np.arange(N)
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    for i in range(num_batches):
        # 一次取batch_size个样本
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        y_batch = y[i * batch_size:(i + 1) * batch_size]
        grad = X_batch.T.dot(X_batch.dot(θ) - y_batch)
        learning_rate = learning_rate_schedule(epoch * N + i)
        θ = θ - learning_rate * grad

print("Real w, b:", w, b)
print("Calculated w, b:", θ)

plc.line(θ[0], θ[1], 'r-')

plc.show()
