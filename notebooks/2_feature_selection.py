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
    
    # Set paths for data directory access
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    input_file = os.path.join(processed_data_dir, 'clean_processed_data.csv')
    
    if os.path.exists(input_file):
        # Load the preprocessed data from clean_processed_data.csv
        # This print statement indicates the data loading process has started
        print("Loading preprocessed data...")
        df = pd.read_csv(input_file)
        
        # Separate features (X) and target variable (y)
        # X contains all columns except 'price'
        # y contains only the 'price' column (what we want to predict)
        X = df.drop('price', axis=1)  # Feature matrix
        y = df['price']  # Target variable
        
        # Start the feature selection process using the imported function
        # This print statement indicates the feature selection process has started
        print("\nStarting feature selection...")
        # Call select_features to get the most important features using F-regression
        selected_features = select_features(X, y)
        
        # Calculate feature importance scores using F-regression
        # We use the same number of features as returned by select_features
        selector = SelectKBest(f_regression, k=len(selected_features))
        selector.fit(X, y)
        
        # Create a DataFrame containing feature importance scores
        # - Each feature name is an index
        # - 'importance' column contains the F-scores
        # - Sorted in descending order of importance
        feature_importance = pd.DataFrame(
            {'importance': selector.scores_},
            index=X.columns
        ).sort_values('importance', ascending=False)
        
        # Save the feature importance rankings to a CSV file
        # This makes it easier to analyze feature importance later
        importance_file = os.path.join(processed_data_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_file, index=True)
        # This print statement confirms the feature importance file has been saved
        # and shows the file path
        print(f"\nFeature importance rankings saved to: {importance_file}")
        
        # Create a new dataset that only includes the selected features and the price
        # This reduces dimensionality and focuses on the most important features
        selected_data = df[selected_features.tolist() + ['price']]  # Include target variable 'price'
        
        # Save the reduced feature dataset to a CSV file for model training
        output_file = os.path.join(processed_data_dir, 'selected_features_data.csv')
        selected_data.to_csv(output_file, index=False)
        # This print statement confirms the selected features dataset has been saved
        # and shows the file path
        print(f"Selected feature data saved to: {output_file}")
        
        # Print a summary of the feature selection process
        # This gives an overview of the dimensionality reduction achieved
        print("\nFeature Selection Summary:")
        # Display the original number of features (excluding the target variable)
        print(f"Original feature count: {df.shape[1] - 1}")
        # Display the number of selected features after filtering
        print(f"Selected feature count: {len(selected_features)}")
        
        # Print a list of all selected features with their importance scores
        # This helps identify which features are most predictive of flight prices
        print("\nSelected Features List:")
        for i, feature in enumerate(selected_features, 1):
            importance = feature_importance.loc[feature, 'importance']
            # Each line shows: rank. feature_name: importance_score
            print(f"{i}. {feature}: {importance:.4f}")
    else:
        # Error message if the input file does not exist
        # This indicates that data preprocessing step must be run first
        print(f"Error: Cannot find preprocessed data file {input_file}")
        print("Please run 1_data_preprocessing.py first")
