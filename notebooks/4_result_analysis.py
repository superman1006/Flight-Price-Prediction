'''
Author: superman1006 1402788264@qq.com
Date: 2025-03-31 23:20:18
LastEditors: superman1006 1402788264@qq.com
LastEditTime: 2025-04-15 23:08:47
FilePath: Flight-Price-Prediction\notebooks\4_result_analysis.py
Description: Model result analysis and visualization script
'''

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add project root to Python path to ensure src modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure matplotlib and seaborn settings
plt.style.use('ggplot')  # Use ggplot style instead of seaborn
sns.set()  # Apply seaborn defaults

if __name__ == "__main__":
    """
    Result Analysis Main Program
    Main functionalities:
    1. Load trained model and test data
    2. Generate predictions
    3. Create visualization plots:
       - Prediction vs Actual comparison scatter plot
       - Feature importance bar plot
       - Prediction error distribution histogram
       - Residuals analysis plot
    4. Save analysis results and visualization plots
    """
    
    # Set paths
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    figures_dir = os.path.join(project_root, 'data', 'figure')
    
    # Ensure output directories exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    model_file = os.path.join(models_dir, 'lightgbm_model.pkl')
    data_file = os.path.join(processed_data_dir, 'selected_features_data.csv')
    
    if not os.path.exists(model_file) or not os.path.exists(data_file):
        print("Error: Cannot find required model file or data file")
        print("Please run 3_model_training.py first")
        sys.exit(1)
    
    # Load model and data
    model = joblib.load(model_file)
    data = pd.read_csv(data_file)
    
    # Prepare features and target variable
    X = data.drop('price', axis=1)
    y_true = data['price']
    
    # Check if model has feature_importances_ attribute (LightGBM or tree-based models)
    has_feature_importance = hasattr(model, 'feature_importances_')
    
    # Make predictions
    try:
        y_pred = model.predict(X)
    except:
        # Handle possible format issues for different model types
        print("Warning: Using alternative prediction method")
        # Try to handle LightGBM booster vs sklearn model differences
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        else:
            if hasattr(model, 'booster_'):
                y_pred = model.booster_.predict(X)
            else:
                print("Error: Cannot make predictions with this model type")
                sys.exit(1)
    
    # 1. Create prediction vs actual comparison plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual Price Comparison')
    plt.savefig(os.path.join(figures_dir, 'prediction_comparison.png'))
    plt.close()
    
    # 2. Create feature importance plot if available
    if has_feature_importance:
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Ranking')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'feature_importance.png'))
        plt.close()
    else:
        print("Note: Feature importance plot not created (model does not support feature importance)")
    
    # 3. Create prediction error distribution plot
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.savefig(os.path.join(figures_dir, 'error_distribution.png'))
    plt.close()
    
    # 4. Create residuals plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals Analysis')
    plt.savefig(os.path.join(figures_dir, 'residuals.png'))
    plt.close()
    
    # Save analysis results
    analysis_file = os.path.join(results_dir, 'analysis_results.txt')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("Model Analysis Results Summary:\n\n")
        f.write("1. Prediction Performance Metrics:\n")
        f.write(f"   - Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.2f}\n")
        f.write(f"   - Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}\n")
        f.write(f"   - Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.2f}\n")
        f.write(f"   - R-squared (R^2): {r2_score(y_true, y_pred):.4f}\n\n")
        
        # Write feature importance if available
        if has_feature_importance:
            f.write("2. Feature Importance Ranking:\n")
            for idx, row in feature_importance.iterrows():
                f.write(f"   - {row['Feature']}: {row['Importance']:.4f}\n")
    
    print("\nAnalysis complete!")
    print(f"Analysis results saved to: {analysis_file}")
    print(f"Visualization plots saved to: {figures_dir}")
    print("\nGenerated plots include:")
    print("1. prediction_comparison.png - Predicted vs Actual Price Comparison")
    if has_feature_importance:
        print("2. feature_importance.png - Feature Importance Plot")
    print("3. error_distribution.png - Prediction Error Distribution")
    print("4. residuals.png - Residuals Analysis Plot")