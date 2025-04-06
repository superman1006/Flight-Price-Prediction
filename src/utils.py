# utils.py
# 辅助工具函数
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib



def save_model(model, filename):
    # 使用joblib将模型保存到指定文件
    joblib.dump(model, filename)
    print(f"模型已保存到 {filename}")

def load_model(filename):
    # 从指定文件加载模型
    import joblib
    model = joblib.load(filename)
    print(f"模型从 {filename} 加载成功")
    return model

def convert_price(x):
    # 将价格字符串中的逗号去掉并转换为浮点数
    if isinstance(x, str):
        price = x.replace(',', '')
        return float(price)
    return float(x)

def convert_date(x):
    # 将日期转换为时间戳
    date = pd.to_datetime(x, dayfirst=True)
    return date.timestamp()

def convert_duration(x):
    # 将时长转换为小时数
    if isinstance(x, str):
        # 处理可能的字符串格式
        return float(x)
    return float(x)

def convert_days_left(x):
    # 将剩余天数转换为整数
    if isinstance(x, str):
        return int(x)
    return int(x)

def convert_flight(x):
    # 将航班号转换为数值型
    return hash(x)

def convert_num_code(x):
    # 将航班号转换为数值型
    if isinstance(x, str):
        return int(x.replace('AI', ''))
    return int(x)

def convert_dep_time(x):
    # 将出发时间转换为分钟数
    if isinstance(x, str):
        # 处理时间格式
        return 0  # 默认值，根据实际数据格式修改
    return 0

def convert_time_taken(x):
    # 将飞行时间转换为分钟数
    if isinstance(x, str):
        # 处理时间格式
        return 0  # 默认值，根据实际数据格式修改
    return 0

def convert_arr_time(x):
    # 将到达时间转换为分钟数
    if isinstance(x, str):
        # 处理时间格式
        return 0  # 默认值，根据实际数据格式修改
    return 0

def convert_stop(x):
    # 将停靠次数转换为数值型
    if isinstance(x, str):
        if x.lower() == 'non-stop':
            return 0
        return int(x.split()[0])
    return int(x)
