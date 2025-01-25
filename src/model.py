# model.py
# 模型构建与训练：创建并训练线性回归模型

from sklearn.linear_model import LinearRegression
def train_model(X_train, y_train):
    # 创建线性回归模型
    model = LinearRegression()
    # 训练模型
    model.fit(X_train, y_train)
    return model
