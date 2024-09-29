import numpy as np
from utils import *
from utils_1_2 import plot_by_ids

res = np.load('res.npy')

for i in range(len(res) - 1):
    x1 = res[i][0][0]
    y1 = res[i][0][1]
    x2 = res[i + 1][0][0]
    y2 = res[i + 1][0][1]
    print("dx, dy: ", x1 - x2, y1 - y2, "| dis:", cal_distance(x1, x2, y1, y2))


# # 读取.npy文件
# res = np.load('res.npy')
# print(res.shape)
# res_x = res[:, :, 0]
# res_y = res[:, :, 1]
#
# print(res_x.shape)
# print(res_y.shape)
# # 将 NumPy 数组转换为 Pandas DataFrame
# df_x = pd.DataFrame(res_x)
# df_y = pd.DataFrame(res_y)
# # 将 DataFrame 保存为 CSV 文件
# df_x.to_csv('res_x.csv', index=True)
# df_y.to_csv('res_y.csv', index=True)

