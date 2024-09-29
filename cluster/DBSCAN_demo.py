import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn import datasets

# 创建两个同心圆环数据
X, y = datasets.make_circles(n_samples=1000, noise=0.05, factor=0.5)
# 创建右上角的点簇
X1, y1 = datasets.make_blobs(n_samples=500, n_features=2, centers=[(1.5, 1.5)], cluster_std=0.2)
# 合并数据
X = np.concatenate((X, X1), axis=0)
y = np.concatenate((y, y1 + 2), axis=0)

# 使用Kmeans聚类
kMeans_instance = KMeans(n_clusters=3)
kMeans_instance.fit(X)
y_pred1 = kMeans_instance.predict(X)

# 使用DBSCAN聚类
DBSCAN_instance = DBSCAN(eps=0.2, min_samples=3)
DBSCAN_instance.fit(X)
y_pred2 = DBSCAN_instance.labels_

# 绘图部分
# 创建图像显示的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(X[:, 0], X[:, 1], c=y_pred1)
ax1.set_title('KMeans Result')
ax1.axis('off')  # 不显示坐标轴

ax2.scatter(X[:, 0], X[:, 1], c=y_pred2)
ax2.set_title(f'DBSCAN Result')
ax2.axis('off')  # 不显示坐标轴

plt.show()
