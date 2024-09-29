import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

IMG_PATH = './avatar.png'
COLOR_CNT = 4

img = plt.imread(IMG_PATH)

h, w = img.shape[:2]
X = img.reshape(-1, 3)  # 图片转二维训练数据

kmeans = KMeans(n_clusters=COLOR_CNT)  # 几种主要颜色
kmeans.fit(X)

main_colors = kmeans.cluster_centers_  # 提取出的主要颜色

y_pred = kmeans.predict(X)
img_result = main_colors[y_pred]
img_result = img_result.reshape(h, w, 3)

# 绘图部分
# 创建图像显示的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 显示原图
ax1.imshow(img)
ax1.set_title('Original Image')
ax1.axis('off')  # 不显示坐标轴

# 显示生成图
ax2.imshow(img_result)
ax2.set_title(f'Generated Image with {COLOR_CNT} Main Colors')
ax2.axis('off')  # 不显示坐标轴

# 显示图像
plt.show()
