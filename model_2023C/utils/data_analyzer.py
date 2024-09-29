import pandas as pd
import numpy as np


class NumericalFeature:
    def __init__(self, skewness, kurt, min_value, max_value, mean_value,
                 median_value, mode_value):
        self.skewness = skewness
        self.kurt = kurt
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.median_value = median_value
        self.mode_value = mode_value

    def __repr__(self):
        return (f"NumericalFeature(\n偏度={self.skewness},\n 峰度={self.kurt},\n "
                f"最小值={self.min_value},\n 最大值={self.max_value},\n "
                f"平均值={self.mean_value},\n 中位数={self.median_value},\n "
                # f"众数={self.mode_value})"
                )


def get_numerical_features(df, column_name):
    skewness = df[column_name].skew()  # 计算偏度
    kurt = df[column_name].kurt()  # 计算峰度
    min_value = df[column_name].min()  # 计算最小值
    max_value = df[column_name].max()  # 计算最大值
    mean_value = df[column_name].mean()  # 计算平均数
    median_value = df[column_name].median()  # 计算中位数
    mode_value = df[column_name].mode()  # 计算众数
    return NumericalFeature(skewness, kurt, min_value, max_value, mean_value, median_value, mode_value)





