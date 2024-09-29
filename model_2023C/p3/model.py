from model_2023C.p1.data_initializer import *

def regress_specify_24_to_30_data():
    type_classify_dict, item_name_dict, classify_name_dict = load_type_info()
    specify_sales_df = load_specify_june_24_to_30_data()
    for code, name in item_name_dict.items():
        df = specify_sales_df[specify_sales_df['单品编码'] == code]
        if df.empty:
            continue
        print(df)



regress_specify_24_to_30_data()

