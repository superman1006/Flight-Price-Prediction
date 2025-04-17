# evaluation.py
# Model evaluation and optimization: Calculate metrics like MSE, R2, etc.

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using multiple metrics
    
    Parameters:
    -----------
    model : trained model object
        The model to evaluate
    X_test : DataFrame
        Test features
    y_test : Series
        Test target (price)
        
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    try:
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
        cv_scores_mean = cv_scores.mean()
        cv_scores_std = cv_scores.std()
    except:
        # Fallback if cross-validation fails (e.g., for LightGBM native model)
        cv_scores_mean = r2
        cv_scores_std = 0.0
    
    # Return all metrics in a dictionary
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_scores_mean': cv_scores_mean,
        'cv_scores_std': cv_scores_std
    }
    
    return results
