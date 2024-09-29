import pandas as pd

file_name = '../dataset/d4_loss_rate.xlsx'
csv_file = '../dataset/d4_loss_rate.csv'

# 读取Excel文件
df = pd.read_excel(file_name)

# 将DataFrame保存为CSV文件
df.to_csv(csv_file, index=False)
print("成功")
