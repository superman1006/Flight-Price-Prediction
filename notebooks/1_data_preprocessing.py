'''
Author: superman1006 1402788264@qq.com
Date: 2025-03-31 23:20:18
LastEditors: superman1006 1402788264@qq.com
LastEditTime: 2025-04-15 23:00:47
FilePath: Flight-Price-Prediction\notebooks\1_data_preprocessing.py
Description: Data preprocessing script for flight price prediction
'''
import sys
import os
# Add project root to Python path to ensure src modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import data preprocessing module from src directory
from src.data_preprocessing import preprocess_data


if __name__ == "__main__":
    """
    Data Preprocessing Main Program
    Main functionalities:
    1. Read the original dataset 'Clean_Dataset.csv'
    2. Preprocess the data, including:
       - Handle missing values
       - Standardize numerical features
       - Encode categorical features
       - Feature selection
    3. Save processed data to processed directory
    4. Output basic statistical information after processing
    """
    
    # Set data file paths
    # raw_data_dir: directory for raw data storage
    # processed_data_dir: directory for processed data storage
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    
    # Ensure output directory exists, create if not
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Process Clean_Dataset
    clean_file = os.path.join(raw_data_dir, 'Clean_Dataset.csv')
    if os.path.exists(clean_file):
        # Call preprocessing function to process data
        df_clean = preprocess_data(clean_file)
        
        if df_clean is not None:
            # Save processed data to CSV file
            output_file = os.path.join(processed_data_dir, 'clean_processed_data.csv')
            df_clean.to_csv(output_file, index=False)
            print(f"\nProcessed data saved to: {output_file}")
            
            # Output price statistics for validation
            print("\nPrice range after processing:")
            print(f"Minimum price: {df_clean['price'].min():.2f}")  # Display lowest price
            print(f"Maximum price: {df_clean['price'].max():.2f}")  # Display highest price
            print(f"Average price: {df_clean['price'].mean():.2f}")  # Display average price
            print(f"Price std dev: {df_clean['price'].std():.2f}")  # Display price volatility