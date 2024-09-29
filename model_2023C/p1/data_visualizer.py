import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from model_2023C.utils.data_analyzer import *
from data_initializer import *


def plot_classify_sales_amount_by_date():
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    classify_sales_amount_df = load_classify_sales_volume_by_date()
    classify_sales_amount_df['分类名称'] = classify_sales_amount_df['分类编码'].map(classify_name_dict)

    # 设置绘图风格
    sns.set(style="ticks")

    # 设置中文字体
    matplotlib.rcParams['font.family'] = 'Microsoft Yahei'  # 字体，改为微软雅黑，默认 sans-serif
    matplotlib.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常

    # 绘制趋势图
    plt.figure(figsize=(12, 6))  # 设置图形大小
    sns.lineplot(data=classify_sales_amount_df, x='销售日期', y='销量(千克)', hue='分类名称')

     # 添加图形标题和坐标轴标签
    plt.title('销量随日期变化的趋势图')
    plt.xlabel('date')
    plt.ylabel('销量(千克)')

    # 设置x轴的主要刻度定位器为每个月的第一天
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=3))

    # 设置x轴的日期格式为 'YYYY-MM'
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=45)

    # 显示图例
    plt.legend(title='分类名称')

    # 显示图形
    plt.show()


def plot_classify_sales_amount_by_quarter():
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    classify_sales_amount_df = load_classify_sales_volume_by_quarter()
    classify_sales_amount_df['分类名称'] = classify_sales_amount_df['分类编码'].map(classify_name_dict)

    # 设置绘图风格
    sns.set(style="ticks")

    # 设置中文字体
    matplotlib.rcParams['font.family'] = 'Microsoft Yahei'  # 字体，改为微软雅黑，默认 sans-serif
    matplotlib.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常

    # 绘制趋势图
    plt.figure(figsize=(12, 6))  # 设置图形大小
    sns.lineplot(data=classify_sales_amount_df, x='季度', y='销量(千克)', hue='分类名称',
                 lowess=True)

     # 添加图形标题和坐标轴标签
    plt.title('销量随季度变化的趋势图')
    plt.xlabel('quarter')
    plt.ylabel('销量(千克)')

    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=45)

    # 显示图例
    plt.legend(title='分类名称')

    # 显示图形
    plt.show()


def show_classify_numerical_features():
    df = load_classify_sales_volume_by_date()
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    specify_df = {}
    for code, name in classify_name_dict.items():
        df_specify = df[df['分类编码'] == code]
        nf = get_numerical_features(df_specify, '销量(千克)')
        specify_df[name] = nf


plot_classify_sales_amount_by_date()


