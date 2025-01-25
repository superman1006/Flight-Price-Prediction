from src.data_preprocessing import preprocess_data  # 导入预处理函数
# 读取数据并进行预处理
print("business_df")
business_df = preprocess_data('../data/raw/business.csv')
print("economy_df")
economy_df = preprocess_data('../data/raw/economy.csv')
print("clean_df")
clean_df = preprocess_data('../data/raw/Clean_Dataset.csv')



# # 查看特征分类
print(f"business_df列：{business_df.columns}")
print(f"economy_df列：{economy_df.columns}")
print(f"clean_df列：{clean_df.columns}")


# # 保存预处理后的数据
business_df.to_csv('../data/processed/business_processed_data.csv', index=False)
economy_df.to_csv('../data/processed/economy_processed_data.csv', index=False)
clean_df.to_csv('../data/processed/clean_processed_data.csv', index=False)
print("数据保存成功")
