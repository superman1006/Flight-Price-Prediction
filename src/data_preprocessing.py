import pandas as pd
from sklearn.preprocessing import StandardScaler
import src.utils as utils

def preprocess_data(file_path):
    try:
        # 这三个数据集的特征不一样，所以需要分类处理
        if "Clean_Dataset" in file_path:
            # pd.read_csv()是用pandas读取csv文件，读取完后df是一个DataFrame对象
            # usecols参数是为了选取需要的列，减少内存占用
            # converters参数是为了对读取的数据进行转换，这里是把price列的数据中的逗号去掉，然后转换为float类型,还有把dep_time转换为分钟
            # lambada x: y 匿名函数，输入x，返回y，这里是把price列的数据中的逗号去掉，然后转换为float类型
            df = pd.read_csv(file_path, usecols=['airline', 'flight', 'source_city', 'departure_time',
                                                 'stops', 'arrival_time', 'destination_city', 'class',
                                                 'duration', 'days_left', 'price'],
                             converters={
                                 'duration':utils.convert_duration,
                                 'days_left':utils.convert_days_left,
                                 'price': utils.convert_price,
                                 'flight': utils.convert_flight
                                }
                             )

        elif "economy" in file_path or "business" in file_path:
            df = pd.read_csv(file_path, usecols=['date', 'airline', 'ch_code', 'num_code', 'dep_time', 'from','time_taken', 'stop', 'arr_time', 'to', 'price'],
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
    # 作用:分类出    numerical_cols数值型(像price和num_code这种数字)   和   categorical_cols类别型(像date和airline这种)   的特征
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"numerical_cols:{numerical_cols}")
    # business_df和economy_df的输出结果:['date', 'num_code', 'dep_time', 'time_taken', 'arr_time', 'price']
    # clean_df的输出结果:['duration', 'days_left', 'price']

    print(f"categorical_cols:{categorical_cols}")
    # business_df和economy_df的输出结果:['airline', 'ch_code', 'from', 'stop', 'to']
    # clean_df的输出结果:['airline', 'flight', 'source_city', 'departure_time', 'stops','arrival_time', 'destination_city', 'class']



    # 二.缺失值处理
    # 查看每一个特征的缺失值情况，缺失值查看的是每列的缺失值数量，isnull()是判断是否为空值，sum()是求和
    print('-' * 100)
    print(f"数据的列:\n{df.columns}")
    print('-' * 100)
    print(f"缺失值情况:\n{df.isnull().sum()}")


    # 2.1 数值型缺失
    # 对于 数值型特征 中的空数据，使用每列的均值填充, .fillna()是填充函数，作用就是把数据中的空值填充为均值
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())


    # 2.2 类别型缺失
    # 对于 类别型特征 中的空数据，使用每列的众数填充
    print('-' * 100)
    for col in categorical_cols:
        # df[col].mode() 返回一个Series类型的对象,包含四个参数: 众数值，众数出现的次数，众数的索引，众数的数据类型
        # .iloc[0] 获取这个 Series 中的第一个值，也就是众数值
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)
        # print(f'{col}的众数:{mode_value.strip()}')


    # 三.特征转换(类别型特征转换为虚拟变量,数值型特征标准化)

    # 3.1 数值特征 标准化
    # 创建标准化器对象
    scaler = StandardScaler()
    # fit_transform将特征缩放到均值为0，标准差为1的范围。标准化后的特征将替换原始的数值特征
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    # print("标准化前的数值特征:")
    # print(df[numerical_cols].head())
    # print("标准化后的数值特征:")
    # print(pd.DataFrame(scaler.transform(df[numerical_cols]), columns=numerical_cols).head())


    # 3.2 把 类别型 --> 数值型.  类别型特征 to 有三个值: Hyderabad, Kolkata, Mumbai, 转换为虚拟变量后，就变成了三个数值型特征:
    #     to                  to_Hyderabad    to_Kolkata    to_Mumbai
    #    Mumbai                   False         False         True
    #    Kolkata       --->       False         True          False
    #    Hyderabad                True          False         False

    # 将类别型特征转换为数值型特征时,使用独热编码（one-hot encoding）会增加数据量,但它避免了数值编码带来的潜在问题。
    # 直接将每个类别转换为数字（如0, 1, 2, 3等）会引入顺序关系!!!!!!!,而这种顺序关系在实际数据中可能并不存在,从而影响模型的性能
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df
