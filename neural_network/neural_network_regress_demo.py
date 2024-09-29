from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def grid_search_for_best_param():
    # 创建一个管道，包括特征缩放和神经网络模型
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 特征缩放
        ('mlp', MLPRegressor(random_state=42)),  # 神经网络模型
    ])

    # 定义要搜索的超参数网格
    param_grid = {
        'mlp__hidden_layer_sizes': [(40,), (50,), (60,)],  # 隐藏层大小
        'mlp__solver': ['adam', 'sgd', 'lbfgs'],  # 优化器
        'mlp__learning_rate_init': [0.005, 0.01, 0.015],  # 初始学习率
        'mlp__max_iter': [200, 300, 400],  # 最大迭代次数
        'mlp__activation': ['relu', 'tanh'],  # 激活函数
    }

    # 创建网格搜索对象
    grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')

    # 执行网格搜索
    grid_search.fit(X_train, y_train)

    # 输出最佳参数和最佳模型的负均方误差
    print("最佳参数:", grid_search.best_params_)
    print("最佳模型的负均方误差:", grid_search.best_score_)

    return grid_search.best_estimator_


# 使用最佳模型进行预测
best_model = grid_search_for_best_param()
y_pred = best_model.predict(X_test)

# 计算测试集的均方误差
mse = mean_squared_error(y_test, y_pred)
print("测试集的均方误差(MSE):", mse)

# 计算决定系数 R^2
r2 = r2_score(y_test, y_pred)
print("测试集的决定系数 R^2:", r2)

# 计算均方根误差
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("测试集的均方根误差(RMSE):", rmse)
