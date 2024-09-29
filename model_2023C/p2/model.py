from model_2023C.p1.data_initializer import *

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def process_data():
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    sale_amount_df = load_classify_sales_amount_by_date()

    print(sale_amount_df)
    for key, value in classify_name_dict.items():
        print(value)
        sale_amount_df_one_classify = sale_amount_df[sale_amount_df['分类编码'] == key]
        sale_amount_df_one_classify = sale_amount_df_one_classify.drop('分类编码', axis=1)
        sale_amount_df_one_classify.to_csv(f"../indirect_dataset/{value}_sales_amount_by_date.csv", index=False)

        X = sale_amount_df_one_classify['加权平均单价(元)'].to_numpy()
        y = sale_amount_df_one_classify['销量(千克)'].to_numpy()

        corr, p_value = pearsonr(X, y)
        print(f"皮尔逊相关系数: {corr:.2f}")
        print(f"p 值: {p_value:.4f}")

        X = X.reshape(-1, 1)
        print(X[:, 0].shape)

        # 对数据进行Z-Score归一化
        # ss = MinMaxScaler()
        # ss.fit(X)
        # X = ss.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 建模
        model = SVR(kernel='linear', C=0.1)
        model.fit(X_train, y_train)
        print("W:", model.coef_[0])
        print("b:", model.intercept_)

        # 绘图
        plt.scatter(X, y)
        x_plot = np.linspace(0, max(X[:, 0]), 100)  # 生成从0到10的100个点
        y_plot = model.coef_ * x_plot + model.intercept_
        plt.plot(x_plot, y_plot[0], color='red')
        plt.show()

        # 评价
        print("SVR_Linear拟合后的模型得分 (越接近1，拟合度越高) :", model.score(X_test, y_test))

        # 使用模型进行预测
        y_pred = model.predict(X_test)

        # 计算 MSE
        mse = mean_squared_error(y_test, y_pred)
        print("测试集的均方误差(MSE):", mse)

        # 计算 RMSE
        rmse = np.sqrt(mse)
        print("测试集的均方根误差(RMSE):", rmse)

        # 计算 R²
        r2 = r2_score(y_test, y_pred)
        print("测试集的 R²:", r2)


    # # 创建透视表
    # pivot_df = sale_amount_df.pivot_table(index='销售日期', columns='分类编码', values='销量(千克)', aggfunc='sum')
    #
    # # 重置索引，使销售日期成为列
    # pivot_df = pivot_df.reset_index()
    #
    # # 填充可能存在的 NaN 值（例如，某些日期或分类编码没有销量记录）
    # pivot_df = pivot_df.fillna(0)
    #
    # # 现在 pivot_df 就是你想要的格式，可以用于后续的训练
    # print(pivot_df)


def process_wholesale_price():
    sale_amount_df = load_sale_detail()
    print(sale_amount_df)
    wholesale_price_df = load_wholesale_price()

    print(wholesale_price_df)



def train_data(pivot_df):
    for column in pivot_df.columns:
        if column == '销售日期':
            continue
        print(pivot_df[column])


process_data()


