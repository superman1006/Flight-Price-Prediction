# model.py
# Model building and training: Create and train machine learning models

import numpy as np
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train a model for flight price prediction
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target (price)
    X_val : DataFrame, optional
        Validation features
    y_val : Series, optional
        Validation target (price)
        
    Returns:
    --------
    model : trained model object
    """
    
    # Use LightGBM for better performance with categorical features
    # If validation set is provided, use it for early stopping
    if X_val is not None and y_val is not None:
        try:
            # Convert data to LightGBM format
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            # Set model parameters
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
            }
            
            # Try different API versions
            try:
                # Method 1: Use train with early_stopping_round (singular)
                model = lgb.train(
                    params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_train, lgb_eval],
                    early_stopping_round=50,
                    verbose_eval=False
                )
            except TypeError:
                try:
                    # Method 2: Use LGBMRegressor instead
                    model = lgb.LGBMRegressor(
                        boosting_type='gbdt',
                        objective='regression',
                        metric='rmse',
                        num_leaves=31,
                        learning_rate=0.05,
                        feature_fraction=0.9,
                        bagging_fraction=0.8,
                        bagging_freq=5,
                        verbose=0,
                        n_estimators=1000
                    )
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except:
                    # Fallback to Linear Regression
                    print("LightGBM failed, falling back to Linear Regression")
                    model = LinearRegression()
                    model.fit(X_train, y_train)
        except:
            # Fallback to Linear Regression if LightGBM setup fails
            print("LightGBM setup failed, falling back to Linear Regression")
            model = LinearRegression()
            model.fit(X_train, y_train)
    else:
        # Fallback to simpler model if no validation set is provided
        model = LinearRegression()
        model.fit(X_train, y_train)
    
    return model
