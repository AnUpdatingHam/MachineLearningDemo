# 1.导入需要的库
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = load_breast_cancer()
# 乳腺癌数据集有569条记录，30个特征
# 单看维度虽然不算太高，但是样本量非常少。过拟合的情况可能存在

# 3.进行一次简单的建模，看看模型本身在数据集上的效果
rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
print(score_pre)
# 可以看到，随机森林在乳腺癌数据上的表现本就还不错，在现实数据集上，基本上不可能什么都不调就看到95%以上的准确率

# 随机森林调参第一步：调n_estimators
# 在这里选择学习曲线，可以看见趋势
# 看见n_estimators在什么取值开始变得平稳，是否一直推动模型整体准确率的上升等信息
# 第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何引起模型整体准确率的变化
# scores = []
# for i in range(0, 200, 10):
#     rfc = RandomForestClassifier(n_estimators=i+1,# n_estimators不能为0
#                                  n_jobs=-1,
#                                  random_state=90)
#     score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
#     scores.append(score)
# print(max(scores), (scores.index(max(scores)) * 10) + 1)
# plt.figure(figsize=(20, 5))
# plt.plot(range(1, 201, 10), scores)
# plt.xlabel("Number of estimators")
# plt.ylabel("cross_val_score")
# plt.show()

# 5.在确定好的范围内，进一步细化学习曲线
scores = []
for i in range(65, 75):# 从上面得到最大值的索引为71，所以范围取65~75
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scores.append(score)
print(max(scores), ([*range(65, 75)][scores.index(max(scores))]))
plt.figure(figsize=(20, 5))
plt.plot(range(65, 75), scores)
plt.show()
