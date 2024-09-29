import pandas as pd


planting_condition2023 = pd.read_csv('../dataset/2023planting_condition.csv')
statistics2023 = pd.read_csv('../dataset/2023statistics.csv')
available_farmland = pd.read_csv('../dataset/available_farmland.csv')
crop_type = pd.read_csv('../dataset/crop_type.csv')

available_farmland = available_farmland.drop("说明 ", axis=1)
crop_type = crop_type.drop("说明", axis=1)

print(planting_condition2023)
print(statistics2023)
print(available_farmland)
print(crop_type.head())







