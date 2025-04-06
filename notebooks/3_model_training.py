import sys
import os
# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
  import lightgbm as lgb
except ImportError:
  print("LightGBM 未安装。请运行以下命令安装：")
  print("pip install lightgbm")
  raise
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置数据文件路径
processed_data_dir = os.path.join(project_root, 'data', 'processed')
figure_dir = os.path.join(project_root, 'data', 'figure')
notebooks_dir = os.path.join(project_root, 'notebooks')

# 确保输出目录存在
os.makedirs(figure_dir, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(processed_data_dir, 'clean_processed_data.csv'))

# 验证价格范围
print("\n加载数据后的价格范围:")
print(f"最小价格: {df['price'].min():.2f}")
print(f"最大价格: {df['price'].max():.2f}")
print(f"平均价格: {df['price'].mean():.2f}")
print(f"价格标准差: {df['price'].std():.2f}")

# Print the columns to verify
print("\n特征列:")
print(df.columns)

# Prepare training data
X = df.drop(columns=['price'])  # 移除price
y = df['price']

# Define preprocessing for numeric features
numeric_features = ['duration', 'days_left']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_features = [
    'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo',
    'airline_SpiceJet', 'airline_Vistara', 'source_city_Bangalore', 'source_city_Chennai',
    'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
    'departure_time_Afternoon', 'departure_time_Early_Morning', 'departure_time_Evening',
    'departure_time_Late_Night', 'departure_time_Morning', 'departure_time_Night',
    'stops_one', 'stops_two_or_more', 'stops_zero', 'arrival_time_Afternoon',
    'arrival_time_Early_Morning', 'arrival_time_Evening', 'arrival_time_Late_Night',
    'arrival_time_Morning', 'arrival_time_Night', 'destination_city_Bangalore',
    'destination_city_Chennai', 'destination_city_Delhi', 'destination_city_Hyderabad',
    'destination_city_Kolkata', 'destination_city_Mumbai', 'class_Business', 'class_Economy'
]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that includes preprocessing and model training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(
        n_estimators=3000,        # 增加树的数量
        learning_rate=0.003,      # 降低学习率以提高稳定性
        max_depth=12,             # 增加树的深度
        num_leaves=63,            # 增加叶子节点数量
        min_child_samples=15,     # 减少最小样本数
        subsample=0.85,           # 增加样本使用比例
        colsample_bytree=0.85,    # 增加特征使用比例
        reg_alpha=0.05,           # 减少L1正则化
        reg_lambda=0.05,          # 减少L2正则化
        random_state=42,
        n_jobs=-1                 # 使用所有CPU核心
    ))
])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("\n交叉验证R2分数:")
print(f"平均: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Predict prices
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R^2 Score: {r2:.4f}")

# 打印一些预测结果的统计信息
print("\n预测结果统计:")
print(f"最小预测值: {y_pred.min():.2f}")
print(f"最大预测值: {y_pred.max():.2f}")
print(f"平均预测值: {y_pred.mean():.2f}")
print(f"预测值标准差: {y_pred.std():.2f}")

# 打印真实值的统计信息
print("\n真实值统计:")
print(f"最小真实值: {y_test.min():.2f}")
print(f"最大真实值: {y_test.max():.2f}")
print(f"平均真实值: {y_test.mean():.2f}")
print(f"真实值标准差: {y_test.std():.2f}")

# 计算预测误差的分布
errors = y_test - y_pred
print("\n预测误差统计:")
print(f"平均误差: {errors.mean():.2f}")
print(f"误差标准差: {errors.std():.2f}")
print(f"最大正误差: {errors.max():.2f}")
print(f"最大负误差: {errors.min():.2f}")

# Visualize the comparison between actual and predicted prices
font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("真实票价与预测票价对比", fontproperties=font_prop)
plt.xlabel("真实票价", fontproperties=font_prop)
plt.ylabel("预测票价", fontproperties=font_prop)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figure_dir, 'comparison_plot.png'))
plt.show()

# Save the entire pipeline
joblib.dump(model, os.path.join(notebooks_dir, 'flight_price_model.pkl'))