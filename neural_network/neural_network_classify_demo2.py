from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 加载数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建神经网络模型
mlp = MLPClassifier(hidden_layer_sizes=(100,),  # 一个隐藏层，100个神经元
                     activation='relu',         # 使用ReLU激活函数
                     solver='adam',            # 使用adam优化器
                     max_iter=300,             # 最大迭代次数
                     random_state=42,           # 随机种子
                     learning_rate_init=0.01)  # 初始学习率

# 训练模型
mlp.fit(X_train, y_train)

# 预测测试集
y_pred = mlp.predict(X_test)

# 评估模型
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))
print("准确率:", accuracy_score(y_test, y_pred))

# 打印训练好的模型参数
# print("模型权重:\n", mlp.coefs_)
