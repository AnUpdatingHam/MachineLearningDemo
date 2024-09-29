# 导入所需库
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用随机森林进行回归
rf = RandomForestRegressor(random_state=42)

# 设置随机森林的参数网格
param_grid = {
    'n_estimators': [67, 68, 69],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [5, 10, 20],  # 节点分裂所需的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点的最小样本数
    'bootstrap': [True, False]  # 是否使用bootstrap样本
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=0, return_train_score=True)

# 运行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳模型的性能
print("Best parameters found: ", grid_search.best_params_)
print("Best Mean Squared Error: ", -grid_search.best_score_)

# 使用最佳参数的模型进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 输出性能指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")

# 绘制预测值与实际值的散点图
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Random Forest Regression on Diabetes Dataset)")
plt.show()

# 绘制特征重要性
feature_importance = best_rf.feature_importances_
features = diabetes.feature_names
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Regression (Diabetes Dataset)")
plt.show()
