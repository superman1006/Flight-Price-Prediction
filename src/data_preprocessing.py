import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import src.utils as utils

def preprocess_data(file_path):
    try:
        # 这三个数据集的特征不一样，所以需要分类处理
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

    # 一.特征分类
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"numerical_cols:{numerical_cols}")
    print(f"categorical_cols:{categorical_cols}")

    # 二.缺失值处理
    print('-' * 100)
    print(f"数据的列:\n{df.columns}")
    print('-' * 100)
    print(f"缺失值情况:\n{df.isnull().sum()}")

    # 2.1 数值型缺失
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # 2.2 类别型缺失
    print('-' * 100)
    for col in categorical_cols:
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)

    # 三.特征转换
    # 3.1 数值特征 标准化
    scaler = StandardScaler()

    # 将price从数值特征中排除
    numerical_cols_without_price = [col for col in numerical_cols if col != 'price']

    # 只对非价格特征进行标准化
    if len(numerical_cols_without_price) > 0:
        df[numerical_cols_without_price] = scaler.fit_transform(df[numerical_cols_without_price])

    # 3.2 类别型特征转换为虚拟变量
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # 打印价格范围
    print("\n价格范围:")
    print(f"最小价格: {df['price'].min():.2f}")
    print(f"最大价格: {df['price'].max():.2f}")
    print(f"平均价格: {df['price'].mean():.2f}")
    print(f"价格标准差: {df['price'].std():.2f}")


    return df