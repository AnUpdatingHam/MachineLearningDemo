import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

diabetes = datasets.load_diabetes()

'''
age: 年龄（以年为单位）
sex: 性别
bmi: 体重指数（Body Mass Index）
bp: 平均血压（Blood Pressure）
s1: 总血清胆固醇（Total Serum Cholesterol，tc）
s2: 低密度脂蛋白（Low-Density Lipoproteins，ldl）
s3: 高密度脂蛋白（High-Density Lipoproteins，hdl）
s4: 总胆固醇与高密度脂蛋白的比值（Total Cholesterol / HDL，tch）
s5: 血清甘油三酯水平的可能对数值（Log of Serum Triglycerides Level，ltg）
s6: 血糖水平（Blood Sugar Level，glu）
'''

column_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

X = diabetes.data    # data
y = diabetes.target  # target

# 行列数
n, m = X.shape

# 对数据进行Min-Max归一化
mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)

# 打乱数据，筛选80%作为训练数据
indexes = np.arange(n)
np.random.shuffle(indexes)

train_cnt = int(0.8 * n)
X_train, y_train = X[indexes[:train_cnt]], y[indexes[:train_cnt]]
X_test, y_test = X[indexes[train_cnt:]], y[indexes[train_cnt:]]

# 建模
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
print("W:", model.coef_)
print("b:", model.intercept_)
print("Min-Max归一化后的模型得分 (越接近1，拟合度越高) :", model.score(X_test, y_test))
