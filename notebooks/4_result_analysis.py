import sys
import os
# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm

# 设置数据文件路径
processed_data_dir = os.path.join(project_root, 'data', 'processed')
figure_dir = os.path.join(project_root, 'data', 'figure')
notebooks_dir = os.path.join(project_root, 'notebooks')

# 确保输出目录存在
os.makedirs(figure_dir, exist_ok=True)

# Define the file path
file_path = os.path.join(processed_data_dir, 'clean_processed_data.csv')

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load data and trained model
df = pd.read_csv(file_path)
model = LinearRegression()

# Assume the training process is completed and load the model
X = df.drop(columns=['price'])
y = df['price']
model.fit(X, y)

# Get model coefficients
coefficients = model.coef_
features = X.columns

# Visualize model coefficients
coef_df = pd.DataFrame(coefficients, index=features, columns=['Coefficient'])
coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)

# Set font properties for Chinese characters
font_path = 'C:/Windows/Fonts/simhei.ttf'  # Path to a font that supports Chinese characters
font_prop = fm.FontProperties(fname=font_path)

plt.figure(figsize=(15, 12))
sns.barplot(x=coef_df.index, y=coef_df['Coefficient'])
plt.title("各特征对票价的影响", fontproperties=font_prop)
plt.xticks(rotation=45, ha='right', fontproperties=font_prop)  # Rotate and align x-axis labels
plt.xlabel("特征", fontproperties=font_prop)
plt.ylabel("系数", fontproperties=font_prop)

# Save the plot as an image file in the figure directory
plt.savefig(os.path.join(figure_dir, 'feature_importance_plot.png'))

# Show the plot
plt.show()

# Summary of results
print("模型系数分析：")
print(coef_df)

# Save the trained model
joblib.dump(model, os.path.join(notebooks_dir, 'linear_regression_model.pkl'))