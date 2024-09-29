import numpy as np
from scipy.stats import spearmanr

# 第一列代表GDP，第二列代表人均寿命
gdp = np.array([62794, 39286, 47603, 42943, 41464, 34483, 46233, 31362, 11289, 8920,
                9771, 2010, 57305, 30371, 9946, 3894, 52367, 23219, 9370, 82950])
life_expectancy = np.array([78.5, 84.1, 80.8, 80.9, 82.3, 82.8, 81.9, 82.0, 72.4, 75.1,
                            76.4, 68.8, 82.6, 83.1, 75.0, 71.5, 81.6, 74.8, 77.4, 83.3])

# 假设 gdp 和 life_expectancy 是已经定义好的两个数值数组
corr, p_value = spearmanr(gdp, life_expectancy)

print(f"斯皮尔曼相关系数: {corr:.2f}")
print(f"p 值: {p_value:.4f}")