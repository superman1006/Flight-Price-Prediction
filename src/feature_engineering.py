# feature_engineering.py
# 特征工程模块：包括特征选择和数据转换等

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def select_features(X, y, k=10):
    # 使用SelectKBest进行特征选择
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)

    # 获取选择的特征
    selected_features = X.columns[selector.get_support()]
    return selected_features
