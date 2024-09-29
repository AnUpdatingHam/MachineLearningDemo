import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d.axes3d import Axes3D
# 自底向上聚类
from sklearn.cluster import AgglomerativeClustering, KMeans
# 分层聚类连接约束
from sklearn.neighbors import kneighbors_graph

X, y = make_swiss_roll(n_samples=1500, noise=0.05)

# KMeans聚类
kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
y_pred1 = kmeans.labels_

# 分层聚类
# linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
# 创建连接性约束
conn = kneighbors_graph(X, n_neighbors=10)  # 采用邻居约束

agg = AgglomerativeClustering(n_clusters=6, linkage='ward', connectivity=conn)
agg.fit(X)
y_pred2 = agg.labels_

# 绘图部分
plt.figure(figsize=(12, 9))
a1 = plt.subplot(121, projection='3d')
a1.set_title('K-Means Clustering')
a1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred1)
a1.view_init(10, -80)

a2 = plt.subplot(122, projection='3d')
a2.set_title('Agg Clustering')
a2.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred2)
a2.view_init(10, -80)
plt.show()
