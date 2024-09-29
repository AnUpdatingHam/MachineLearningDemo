## 导入会使用到的相关库
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import SimpleExpSmoothing,Holt,ExponentialSmoothing,AR,ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# import pmdarima as pm
from sklearn.metrics import mean_absolute_error

# import pyflux as pf
# from fbprophet import Prophet

## 忽略提醒
import warnings
warnings.filterwarnings("ignore")

## 读取时间序列数据,该数据包含:X1为飞机乘客数据,X2为一组随机数据
df = pd.read_csv("辣椒类_sales_amount_by_date.csv")
## 查看数据的变化趋势

def plot_data():
    # 设置中文字体
    matplotlib.rcParams['font.family'] = 'Microsoft Yahei'  # 字体，改为微软雅黑，默认 sans-serif
    matplotlib.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常

    df.plot(kind="line", figsize=(10, 6))

    plt.grid()
    plt.title("时序数据")
    plt.show()


# 白噪声检验Ljung-Box检验
# 该检验用来检查序列是否为随机序列，如果是随机序列，那它们的值之间没有任何关系
# 使用LB检验来检验序列是否为白噪声，原假设为在延迟期数内序列之间相互独立。
def Ljung_Box():
    lags = [4, 8, 16, 32]
    LB = sm.stats.diagnostic.acorr_ljungbox(df["销售金额(元)"], lags=lags, return_df=True)
    print("序列：销售金额(元)的检验结果:\n", LB)
    LB = sm.stats.diagnostic.acorr_ljungbox(df["销量(千克)"], lags=lags, return_df=True)
    print("序列：销量(千克)的检验结果:\n", LB)

    # 在延迟阶数为[4, 6, 16, 32]
    # 的情况下，序列销售金额(元)/销量(千克)的LB检验P值均小于0.05，即该数据不是随机的。有规律可循，有分析价值。
    # 若序列的LB检验P值均大于0.05，该数据为白噪声，没有分析价值


# 时间序列是否平稳，对选择预测的数学模型非常关键
# 如果数据是平稳的，可以使用自回归平均移动模型（ARMA）
# 如果数据是不平稳的，可以使用差分移动自回归平均移动模型（ARIMA）
def stable_analyze():
    ## 序列的单位根检验，即检验序列的平稳性
    dftest = adfuller(df["销售金额(元)"], autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['adf', 'p-value', 'usedlag', 'Number of Observations Used'])
    print("销售金额(元)单位根检验结果:\n", dfoutput)

    dftest = adfuller(df["销量(千克)"], autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['adf', 'p-value', 'usedlag', 'Number of Observations Used'])
    print("销量(千克)单位根检验结果:\n", dfoutput)

    ## 对X1进行一阶差分后的序列进行检验
    X1diff = df["销售金额(元)"].diff().dropna()
    dftest = adfuller(X1diff, autolag='BIC')
    print(dftest)
    dfoutput1 = pd.Series(dftest[0:4], index=['adf', 'p-value', 'usedlag', 'Number of Observations Used'])
    print("销售金额(元)一阶差分单位根检验结果:\n", dfoutput1)

    ## 对X2进行一阶差分后的序列进行检验
    X2diff = df["销量(千克)"].diff().dropna()
    dftest = adfuller(X2diff, autolag='BIC')
    print(dftest)
    dfoutput2 = pd.Series(dftest[0:4], index=['adf', 'p-value', 'usedlag', 'Number of Observations Used'])
    print("销量(千克)一阶差分单位根检验结果:\n", dfoutput2)

    ## 对X2进行一阶差分后的序列进行检验
    X3diff = df["加权平均单价(元)"].diff().dropna()
    dftest = adfuller(X3diff, autolag='BIC')
    print(dftest)
    dfoutput3 = pd.Series(dftest[0:4], index=['adf', 'p-value', 'usedlag', 'Number of Observations Used'])
    print("加权平均单价(元)一阶差分单位根检验结果:\n", dfoutput3)

    # 如果p-value小于显著性水平（通常是0.05），则拒绝原假设，认为序列是平稳的；如果p-value大于显著性水平，则不能拒绝原假设，认为序列是非平稳的。

stable_analyze()


