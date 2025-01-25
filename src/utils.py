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

def convert_num_code(x):
    # 将num_code转换为整数
    return int(x)

def convert_dep_time(x):
    # 将出发时间转换为分钟
    hours, minutes = x.split(':')
    hours = int(hours)
    minutes = int(minutes)
    return hours * 60 + minutes

def convert_time_taken(time_str):
    # 将飞行时间转换为分钟
    hours, minutes = time_str.split('h')
    hours = float(hours.strip())
    minutes = minutes.strip()
    if len(minutes) == 1:
        minutes = 0.0
    else:
        minutes = float(minutes.replace('m', '').strip())
    return hours * 60 + minutes

def convert_arr_time(x):
    # 将到达时间转换为分钟
    hours, minutes = x.split(':')
    hours = int(hours)
    minutes = int(minutes)
    return hours * 60 + minutes

def convert_price(x):
    # 将价格字符串中的逗号去掉并转换为浮点数
    price = x.replace(',', '')
    return float(price)

def convert_date(x):
    # 将日期转换为时间戳
    date = pd.to_datetime(x, dayfirst=True)
    return date.timestamp()

def convert_stop(x):
    # 去除stop字符串中的空格和换行符
    stop = x.replace('\n', '').replace('\t', '').replace(' ', '')
    # 将stop特征转换为数值型特征
    if 'non-stop' in stop:
        return 0
    elif '1-stop' in stop:
        return 1
    elif '2+-stop' in stop:
        return 2
    else:
        return 3  # 其他情况

def convert_duration(x):
    # 将持续时间转换为浮点数
    return float(x)

def convert_days_left(x):
    # 将剩余天数转换为整数
    return int(x)

def convert_flight(x):
    # 将航班号转换为数值型
    return hash(x)
