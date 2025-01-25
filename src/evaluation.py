# evaluation.py
# 模型评估与优化：计算MSE和R^2等指标

from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    # 模型预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
