import numpy as np
from scipy.stats import pearsonr

# 第一列代表GDP，第二列代表人均寿命
gdp = np.array([62794, 39286, 47603, 42943, 41464, 34483, 46233, 31362, 11289, 8920,
                9771, 2010, 57305, 30371, 9946, 3894, 52367, 23219, 9370, 82950])
life_expectancy = np.array([78.5, 84.1, 80.8, 80.9, 82.3, 82.8, 81.9, 82.0, 72.4, 75.1,
                            76.4, 68.8, 82.6, 83.1, 75.0, 71.5, 81.6, 74.8, 77.4, 83.3])

corr, p_value = pearsonr(gdp, life_expectancy)

print(f"皮尔逊相关系数: {corr:.2f}")
print(f"p 值: {p_value:.4f}")
