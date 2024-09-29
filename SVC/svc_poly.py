import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = datasets.load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

svc = SVC(kernel='poly')

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

score = accuracy_score(y_test, y_pred)
print("使用poly核函数，得分是：", score)

