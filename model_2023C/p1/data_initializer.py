import pandas as pd


def load_sale_detail():
    return pd.read_csv("../dataset/d2_sale_detail.csv")  # 假设字段是用制表符分隔的


def load_classify_sales_volume_by_date():
    df = pd.read_csv("../indirect_dataset/classify_sales_volume_by_date.csv")
    # 将销售日期转换为 datetime 类型（如果尚未转换）
    df['销售日期'] = pd.to_datetime(df['销售日期'])
    return df


def load_classify_sales_volume_by_quarter():
    return pd.read_csv("../indirect_dataset/classify_sales_volume_by_quarter.csv")


def load_classify_sales_amount_by_date():
    return pd.read_csv("../indirect_dataset/classify_sales_amount_by_date.csv")


def load_specify_sales_amount_by_date():
    return pd.read_csv("../indirect_dataset/specify_sales_amount_by_date.csv")


def load_specify_june_24_to_30_data():
    return pd.read_csv("../indirect_dataset/specify_june_24_to_30_data.csv")


def load_classify_june_24_to_30_data():
    return pd.read_csv("../indirect_dataset/classify_june_24_to_30_data.csv")

def load_wholesale_price():
    df = pd.read_csv("../dataset/d3_wholesale_price.csv")
    df['日期'] = pd.to_datetime(df['日期'])
    return df

def load_specify_sum_and_medium_amount():
    return pd.read_csv("../dataset/d3_wholesale_price.csv")

def load_type_info():
    # 加载.csv文件
    type_info = pd.read_csv("../dataset/d1_type_info.csv")
    # 分类字典
    type_classify_dict = {}
    item_name_dict = {}
    classify_name_dict = {}
    for index, row in type_info.iterrows():
        type_classify_dict[row['单品编码']] = row['分类编码']
        item_name_dict[row['单品编码']] = row['单品名称']
        classify_name_dict[row['分类编码']] = row['分类名称']
    return type_classify_dict, item_name_dict, classify_name_dict


# 输出根据蔬菜种类分类的日销售额csv文件
def transform_and_save_classify_sales_volume_by_date():
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    sales_data_df = load_sale_detail()

    # 替换单品编码为分类编码
    sales_data_df['分类编码'] = sales_data_df['单品编码'].map(type_classify_dict)

    # 按销售日期和分类编码分组，并对销量进行求和
    aggregated_data = sales_data_df.groupby(['销售日期', '分类编码']).agg({'销量(千克)': 'sum'}).reset_index()

    # 保存转换后的数据到新的CSV文件
    aggregated_data.to_csv("classify_sales_volume_by_date.csv", index=False)


def transform_and_save_classify_sales_volume_by_quarter():
    sales_data_df = load_classify_sales_volume_by_date()
    # 确保销售日期是日期类型
    sales_data_df['销售日期'] = pd.to_datetime(sales_data_df['销售日期'])

    # 创建一个新列来表示季度
    sales_data_df['季度'] = sales_data_df['销售日期'].dt.to_period('Q')

    # 按季度和分类编码分组，并对销量进行求和
    aggregated_data = sales_data_df.groupby(['季度', '分类编码']).agg({'销量(千克)': 'sum'}).reset_index()

    # 保存转换后的数据到新的CSV文件
    aggregated_data.to_csv("classify_sales_volume_by_quarter.csv", index=False)


# 输出根据蔬菜种类分类的日销售额和日销量csv文件
def transform_and_save_classify_sales_amount_by_date():
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    sales_data_df = load_sale_detail()

    # 替换单品编码为分类编码
    sales_data_df['分类编码'] = sales_data_df['单品编码'].map(type_classify_dict)
    sales_data_df['销售金额(元)'] = sales_data_df['销量(千克)'] * sales_data_df['销售单价(元/千克)']

    # 按销售日期和分类编码分组，并对销量进行求和
    aggregated_data = (sales_data_df.groupby(['销售日期', '分类编码']).
                       agg({
                            '销售金额(元)': 'sum',
                            '销量(千克)': 'sum'}).reset_index())
    aggregated_data['加权平均单价(元)'] = aggregated_data['销售金额(元)'] / aggregated_data['销量(千克)']

    aggregated_data['分类名称'] = aggregated_data['分类编码'].map(classify_name_dict)
    # 保存转换后的数据到新的CSV文件
    aggregated_data.to_csv("../indirect_dataset/classify_sales_amount_by_date.csv", index=False)


def transform_and_save_specify_sales_amount_by_date():
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    sales_data_df = load_sale_detail()

    sales_data_df['销售金额(元)'] = sales_data_df['销量(千克)'] * sales_data_df['销售单价(元/千克)']

    # 按销售日期和分类编码分组，并对销量进行求和
    aggregated_data = (sales_data_df.groupby(['销售日期', '单品编码']).
                       agg({
        '销售金额(元)': 'sum',
        '销量(千克)': 'sum'}).reset_index())

    aggregated_data['单品名称'] = aggregated_data['单品编码'].map(item_name_dict)
    aggregated_data['加权平均单价(元)'] = aggregated_data['销售金额(元)'] / aggregated_data['销量(千克)']
    # 保存转换后的数据到新的CSV文件
    aggregated_data.to_csv("../indirect_dataset/specify_sales_amount_by_date.csv", index=False)


def transform_and_save_classify_june_24_to_30_data():
    raw_data_df = load_classify_sales_amount_by_date()
    # 将销售日期列转换为日期格式
    raw_data_df['销售日期'] = pd.to_datetime(raw_data_df['销售日期'])
    start_date = pd.Timestamp('2023-06-24')
    # 筛选特定日期范围内的数据
    filtered_df = raw_data_df[(raw_data_df['销售日期'] >= start_date)]
    # 显示结果
    filtered_df.to_csv("../indirect_dataset/classify_june_24_to_30_data.csv", index=False)


def transform_and_save_specify_june_24_to_30_data():
    raw_data_df = load_specify_sales_amount_by_date()
    # 将销售日期列转换为日期格式
    raw_data_df['销售日期'] = pd.to_datetime(raw_data_df['销售日期'])
    start_date = pd.Timestamp('2023-06-24')
    # 筛选特定日期范围内的数据
    filtered_df = raw_data_df[(raw_data_df['销售日期'] >= start_date)]
    # 显示结果
    filtered_df.to_csv("../indirect_dataset/specify_june_24_to_30_data.csv", index=False)


def get_specify_sum_and_medium_amount():
    df = load_specify_sales_amount_by_date()

    # 提取销售日期列，并找出所有唯一的日期
    unique_dates = df['销售日期'].unique()
    # 计算唯一日期的数量
    number_of_days = len(unique_dates)

    median_sales = df.groupby('单品名称')['销量(千克)'].median().reset_index()
    median_sales.rename(columns={'销量(千克)': '日销量中位数(千克)'}, inplace=True)
    print(median_sales)

    df = df.groupby('单品名称').agg({
        '销售金额(元)': 'sum',
        '销量(千克)': 'sum'
    }).reset_index()

    df['加权平均单价'] = df['销售金额(元)'] / df['销量(千克)']
    df['日销量平均数(千克)'] = df['销量(千克)'] / number_of_days

    df_with_median = pd.merge(df, median_sales, on='单品名称', how='left')

    df.to_csv("../indirect_dataset/specify_sum_and_medium_amount")


get_specify_sum_and_medium_amount()


