# feature_engineering.py
# Feature engineering module: includes feature selection and data transformation

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def select_features(X, y, k=10):
    # Use SelectKBest for feature selection
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)

    # Get selected features
    selected_features = X.columns[selector.get_support()]
    return selected_features
