# 首先我们要导入科学计算库，用于一些科学计算
import numpy as np # 为numpy起一个别名，调用使用起来会很方便
# 现在导入神经网络中的一个多分类模型，用于训练多分类数据
from sklearn.neural_network import MLPClassifier
# 现在导入sklearn中的用于评测预测结果指标的库，如混淆矩阵和分类报告
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载iris数据集
iris = load_iris()
X_data = iris.data
y_data = iris.target

# 定义数据预处理函数
def preprocess(X, y):
	# 对数据的处理我通常会都放入这个函数中，下面将列出部分处理步骤，根据实际情况进行处理
	# 数据提取
	# 特征缩放
	X_min = np.min(X)
	X_max = np.max(X)
	X = (X - X_min) / (X_max - X_min)
	# 前面加一列全1的数据，便于计算截距
	X = np.c_[np.ones(len(X)), X]
	y = np.c_[y]
	# 数据洗牌
	np.random.seed(1)
	m = len(X)
	o = np.random.permutation(m)
	X = X[o]
	y = y[o]
	# 数据切割
	d = int(0.7 * m)
	X_train, X_test = np.split(X,[d])
	y_train, y_test = np.split(y,[d])
	# 数据处理基本完毕，返回处理好的数据
	return X_train,X_test,y_train,y_test
# 调用数据处理函数，获得处理好的数据
X_train, X_test, y_train, y_test = preprocess(X_data,y_data)

"""
	主要参数：
	hidden_layer_sizes: 隐藏层单元数(tuple)，如(100,100,100,50)
	activation : 激活函数，{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, 缺省 ‘relu‘; [f(x) = x, 1/(1+exp(-x)), tanh(x), max(0, x)]
	solver : 解决器, {‘lbfgs’, ‘sgd’, ‘adam’}, 缺省 ‘adam’; [牛顿法,随机梯度下降, 自适应momemtum] 
	alpha : L2正则化参数，float, 可选，缺省0.0001
	batch_size : 批次，可选, 不适用于’lbfgs’, 缺省 ‘auto‘, 此时，batch_size=min(200, n_samples)`
	learning_rate : 学习率, {‘constant’, ‘invscaling’, ‘adaptive’}, 缺省 ‘constant’, 只适用于梯度下降sgd
	learning_rate_init : 学习率初始值, 可选, 缺省 0.001, 仅适用于sgd或adam
	power_t : 下降指数, 可选, 缺省 0.5,  适用于’invscaling’,learning_rate_init/pow(t,power_t), 仅适用于sgd
	max_iter : 最大迭代数, 可选, 缺省200, 迭代器收敛迭代次数，对于sgd/adam, 代表的是epochs数目，不是下降步数
	shuffle : 每次迭代, 是否洗牌, 可选, 缺省True,仅适用于sgd或adam
	random_state: 缺省None; 若int, 随机数产生器seed, 若RandomStates实例, 随机数产生器, 若None, np.random
	tol : 容忍度, 可选, 缺省le-4, 连续两次迭代loss达不到此值，除非设置成’adaptive’,否则，就停止迭代，
	beta_1 : adam指数衰减参数1，可选, 缺省0.9
	beta_2 : adam指数衰减参数2，可选, 缺省0.999
	epsilon : adam数值稳定值，可选，缺省1e-8
"""
# 首先，创建一个多分类模型对象 类似于Java的类调用
# 括号中填写多个参数，如果不写，则使用默认值，我们一般要构建隐层结构，调试正则化参数，设置最大迭代次数
mlp = MLPClassifier(hidden_layer_sizes=(400,100),alpha=0.01,max_iter=300)
# 调用fit函数就可以进行模型训练，一般的调用模型函数的训练方法都是fit()
mlp.fit(X_train,y_train.ravel()) # 这里y值需要注意，还原成一维数组
# 模型就这样训练好了，而后我们可以调用多种函数来获取训练好的参数
# 比如获取准确率
print('训练集的准确率是：', mlp.score(X_train,y_train))
# 比如输出当前的代价值
print('训练集的代价值是：', mlp.loss_)
# 比如输出每个theta的权重
print('训练集的权重值是：', mlp.coefs_)

# 混淆矩阵和分类报告是评价预测值和真实值的一种指标
# 混淆矩阵可以直观的看出分类中正确的个数和分错的个数，以及将正确的样本错误地分到了哪个类别
matrix_train = confusion_matrix(y_train, mlp.predict(X_train))
print('训练集的混淆矩阵是：', matrix_train)
# 分类报告中有多个指标用于评价预测的好坏。
'''
	TP: 预测为1(Positive)，实际也为1(Truth-预测对了)
	TN: 预测为0(Negative)，实际也为0(Truth-预测对了)
	FP: 预测为1(Positive)，实际为0(False-预测错了)
	FN: 预测为0(Negative)，实际为1(False-预测错了)
'''
report_train = classification_report(y_train, mlp.predict(X_train))
print('训练集的分类报告是：\n', report_train)
