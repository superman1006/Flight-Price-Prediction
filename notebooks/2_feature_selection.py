'''
Author: superman1006 1402788264@qq.com
Date: 2025-03-31 23:20:18
LastEditors: superman1006 1402788264@qq.com
LastEditTime: 2025-04-15 23:03:25
FilePath: Flight-Price-Prediction\notebooks\2_feature_selection.py
Description: Feature selection script for flight price prediction
'''

import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# Add project root to Python path to ensure src modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import feature engineering module from src directory
from src.feature_engineering import select_features

if __name__ == "__main__":
    """
    Feature Selection Main Program
    Main functionalities:
    1. Load preprocessed data from processed directory
    2. Perform feature selection:
       - Calculate feature correlations with target variable
       - Use SelectKBest and f_regression for feature scoring
       - Select the most important feature subset
    3. Save selected feature data to processed directory
    4. Output feature importance rankings
    """
    
    # Set data file paths
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    input_file = os.path.join(processed_data_dir, 'clean_processed_data.csv')
    
    if os.path.exists(input_file):
        # Load preprocessed data
        print("Loading preprocessed data...")
        df = pd.read_csv(input_file)
        
        # Perform feature selection
        print("\nStarting feature selection...")
        selected_features, feature_importance = select_features(df)
        
        # Save feature importance rankings
        importance_file = os.path.join(processed_data_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_file, index=True)
        print(f"\nFeature importance rankings saved to: {importance_file}")
        
        # Create new dataset with selected features
        selected_data = df[selected_features + ['price']]  # Include target variable 'price'
        
        # Save selected feature data
        output_file = os.path.join(processed_data_dir, 'selected_features_data.csv')
        selected_data.to_csv(output_file, index=False)
        print(f"Selected feature data saved to: {output_file}")
        
        # Print feature selection summary
        print("\nFeature Selection Summary:")
        print(f"Original feature count: {df.shape[1] - 1}")  # Subtract target variable
        print(f"Selected feature count: {len(selected_features)}")
        print("\nSelected Features List:")
        for i, feature in enumerate(selected_features, 1):
            importance = feature_importance.loc[feature, 'importance']
            print(f"{i}. {feature}: {importance:.4f}")
    else:
        print(f"Error: Cannot find preprocessed data file {input_file}")
        print("Please run 1_data_preprocessing.py first")
