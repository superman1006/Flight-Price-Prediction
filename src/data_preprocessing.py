import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import src.utils as utils

def preprocess_data(file_path):
    try:
        # These three datasets have different features, so they need to be processed differently
        if "Clean_Dataset" in file_path:
            df = pd.read_csv(file_path, usecols=['airline', 'flight', 'source_city', 'departure_time',
                                                 'stops', 'arrival_time', 'destination_city', 'class',
                                                 'duration', 'days_left', 'price'],
                             converters={
                                 'duration': utils.convert_duration,
                                 'days_left': utils.convert_days_left,
                                 'price': utils.convert_price,
                                 'flight': utils.convert_flight
                             }
                             )

        elif "economy" in file_path or "business" in file_path:
            df = pd.read_csv(file_path,
                             usecols=['date', 'airline', 'ch_code', 'num_code', 'dep_time', 'from', 'time_taken',
                                      'stop', 'arr_time', 'to', 'price'],
                             converters={
                                 'num_code': utils.convert_num_code,
                                 'dep_time': utils.convert_dep_time,
                                 'time_taken': utils.convert_time_taken,
                                 'arr_time': utils.convert_arr_time,
                                 'price': utils.convert_price,
                                 'date': utils.convert_date,
                                 'stop': utils.convert_stop
                             }
                             )
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

    # I. Feature Classification
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"numerical_cols:{numerical_cols}")
    print(f"categorical_cols:{categorical_cols}")

    # II. Missing Value Handling
    print('-' * 100)
    print(f"Data columns:\n{df.columns}")
    print('-' * 100)
    print(f"Missing values situation:\n{df.isnull().sum()}")

    # 2.1 Numerical Missing Values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # 2.2 Categorical Missing Values
    print('-' * 100)
    for col in categorical_cols:
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)

    # III. Feature Transformation
    # 3.1 Numerical Feature Standardization
    scaler = StandardScaler()

    # Exclude price from numerical features
    numerical_cols_without_price = [col for col in numerical_cols if col != 'price']

    # Only standardize non-price features
    if len(numerical_cols_without_price) > 0:
        df[numerical_cols_without_price] = scaler.fit_transform(df[numerical_cols_without_price])

    # 3.2 Convert categorical features to dummy variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Print price range
    print("\nPrice Range:")
    print(f"Minimum price: {df['price'].min():.2f}")
    print(f"Maximum price: {df['price'].max():.2f}")
    print(f"Average price: {df['price'].mean():.2f}")
    print(f"Price std dev: {df['price'].std():.2f}")


    return df