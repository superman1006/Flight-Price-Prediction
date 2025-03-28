import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

# 加载预处理后的数据
df = pd.read_csv('../data/processed/clean_processed_data.csv')

# 查看数据的相关性
correlation_matrix = df.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 7})
plt.title("Feature correlation heat map", fontproperties='SimHei')  # 设置字体
# Save the plot as an image file in the figure directory
plt.savefig('../data/figure/feature_correlation_heatmap.png')
plt.show()

# 选择与目标变量（票价）相关性强的特征
X = df.drop(columns=['price'])  # 假设'price'是票价列
y = df['price']

# 使用SelectKBest进行特征选择
selector = SelectKBest(f_regression, k=10)  # 选择10个最重要的特征
X_new = selector.fit_transform(X, y)

# 显示选择的特征
selected_features = X.columns[selector.get_support()]
print(f"选中的特征：{selected_features}")
