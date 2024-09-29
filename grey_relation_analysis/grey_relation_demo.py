import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 创建数据
# data = pd.DataFrame({
#     '实际温度': [1.1, 2.2, 3.3, 4.4, 5.5],
#     '预测模型1': [1.2, 2.3, 3.0, 4.2, 5.1],
#     '预测模型2': [0.8, 1.8, 3.4, 3.8, 5.3]
# })

# 原始数据
data = pd.DataFrame({
    '产品销量': [100, 120, 150, 130, 140],
    '广告投入': [10, 12, 15, 11, 13],
    '促销力度': [8, 10, 12, 9, 11],
    '产品价格': [50, 60, 70, 55, 65],
    '市场需求': [80, 90, 110, 95, 105],
    '竞争对手数量': [5, 6, 8, 7, 9],
    '产品质量评分': [8, 9, 7, 8, 9],
    '销售渠道数量': [3, 4, 5, 4, 6],
    '售后服务评分': [7, 8, 6, 7, 8],
    '品牌知名度': [7, 8, 9, 8, 10]
})

# 增加 5 行数据
new_rows = pd.DataFrame({
    '产品销量': [80, 180, 110, 160, 120],
    '广告投入': [8, 18, 12, 16, 13],
    '促销力度': [6, 12, 10, 14, 11],
    '产品价格': [45, 80, 65, 75, 60],
    '市场需求': [70, 100, 90, 110, 95],
    '竞争对手数量': [4, 10, 7, 9, 8],
    '产品质量评分': [7, 10, 8, 9, 8],
    '销售渠道数量': [2, 6, 4, 5, 7],
    '售后服务评分': [6, 9, 7, 8, 7],
    '品牌知名度': [6, 10, 8, 9, 9]
})

data = pd.concat([data, new_rows], ignore_index=True)

print(data)
print(data.T)

# 使用MinMaxScaler进行最小-最大标准化
scaler_minmax = MinMaxScaler()
normalized_data_minmax = pd.DataFrame(scaler_minmax.fit_transform(data.T),
                                      columns=data.T.columns)

# 使用StandardScaler进行Z-score标准化
scaler_zscore = StandardScaler()
normalized_data_zscore = pd.DataFrame(scaler_zscore.fit_transform(data.T),
                                      columns=data.T.columns)


print(normalized_data_minmax)
print(normalized_data_zscore)

# 计算灰色关联系数的函数
def calculate_gray_relation_coefficient(reference, compare, rho=0.5):
    delta = np.abs(reference - compare)
    delta_min = delta.min()
    delta_max = delta.max()
    return (delta_min + rho * delta_max) / (delta + rho * delta_max)

# 计算灰色关联度的函数
def calculate_gray_relation_degree(normalized_data, reference_row=0):
    relation_degrees = {}
    for index, row in normalized_data.iterrows():
        if index != reference_row:
            coeff = calculate_gray_relation_coefficient(normalized_data.loc[reference_row],
                                                       normalized_data.loc[index],
                                                       rho=0.5)
            relation_degrees[index] = coeff.mean()
    return relation_degrees

# 计算并打印最小-最大标准化后的灰色关联度
gray_relation_degrees_minmax = calculate_gray_relation_degree(normalized_data_minmax)
print("最小-最大标准化后的灰色关联度:")
print(gray_relation_degrees_minmax)

# 计算并打印Z-score标准化后的灰色关联度
gray_relation_degrees_zscore = calculate_gray_relation_degree(normalized_data_zscore)
print("\nZ-score标准化后的灰色关联度:")
print(gray_relation_degrees_zscore)
