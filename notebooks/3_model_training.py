'''
Author: superman1006 1402788264@qq.com
Date: 2025-03-31 23:20:18
LastEditors: superman1006 1402788264@qq.com
LastEditTime: 2025-04-15 16:09:25
FilePath: Flight-Price-Prediction\notebooks\3_model_training.py
Description: Model training script for flight price prediction using LightGBM
'''

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import joblib

# Add project root to Python path to ensure src modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import model training related modules from src directory
from src.model import train_model, evaluate_model

if __name__ == "__main__":
    """
    Model Training Main Program
    Main functionalities:
    1. Load feature-selected dataset
    2. Split dataset:
       - Training set (70%)
       - Validation set (15%)
       - Test set (15%)
    3. Model training and optimization:
       - Use LightGBM model
       - Perform cross-validation
       - Hyperparameter optimization
    4. Model evaluation:
       - Calculate MSE, MAE, R² metrics
       - Generate prediction vs actual comparison plots
    5. Save model and evaluation results
    """
    
    # Set paths for data and model storage
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    
    # Ensure output directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load feature-selected data
    input_file = os.path.join(processed_data_dir, 'selected_features_data.csv')
    if not os.path.exists(input_file):
        print("Error: Cannot find feature-selected data file")
        print("Please run 2_feature_selection.py first")
        sys.exit(1)
    
    # Load data and prepare for training
    print("Loading data...")
    data = pd.read_csv(input_file)
    X = data.drop('price', axis=1)  # Feature matrix
    y = data['price']  # Target variable
    
    # Split dataset
    print("\nSplitting training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    eval_results = evaluate_model(model, X_test, y_test)
    
    # Save model
    model_file = os.path.join(models_dir, 'lightgbm_model.pkl')
    joblib.dump(model, model_file)
    print(f"\nModel saved to: {model_file}")
    
    # Save evaluation results
    results_file = os.path.join(results_dir, 'model_evaluation.txt')
    with open(results_file, 'w') as f:
        f.write("Model Evaluation Results:\n")
        f.write(f"MSE: {eval_results['mse']:.2f}\n")
        f.write(f"RMSE: {eval_results['rmse']:.2f}\n")
        f.write(f"MAE: {eval_results['mae']:.2f}\n")
        f.write(f"R²: {eval_results['r2']:.4f}\n")
        f.write(f"\nCross-validation R² scores: {eval_results['cv_scores_mean']:.4f} (±{eval_results['cv_scores_std']:.4f})")
    
    print(f"Evaluation results saved to: {results_file}")
    
    # Print main evaluation metrics
    print("\nModel Performance Summary:")
    print(f"Mean Squared Error (MSE): {eval_results['mse']:.2f}")
    print(f"Root Mean Squared Error (RMSE): {eval_results['rmse']:.2f}")
    print(f"Mean Absolute Error (MAE): {eval_results['mae']:.2f}")
    print(f"R²: {eval_results['r2']:.4f}")
    print(f"Cross-validation R² scores: {eval_results['cv_scores_mean']:.4f} (±{eval_results['cv_scores_std']:.4f})")