import sys
import os
# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_preprocessing import preprocess_data


if __name__ == "__main__":
    # 设置数据文件路径
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    
    # 确保输出目录存在
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # 处理Clean_Dataset
    clean_file = os.path.join(raw_data_dir, 'Clean_Dataset.csv')
    if os.path.exists(clean_file):
        df_clean = preprocess_data(clean_file)
        if df_clean is not None:
            # 保存处理后的数据
            output_file = os.path.join(processed_data_dir, 'clean_processed_data.csv')
            df_clean.to_csv(output_file, index=False)
            print(f"\n处理后的数据已保存到: {output_file}")
            
            # 再次打印价格范围以确认
            print("\n保存后的价格范围:")
            print(f"最小价格: {df_clean['price'].min():.2f}")
            print(f"最大价格: {df_clean['price'].max():.2f}")
            print(f"平均价格: {df_clean['price'].mean():.2f}")
            print(f"价格标准差: {df_clean['price'].std():.2f}")