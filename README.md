# Flight Price Prediction

本项目旨在预测航班价格，帮助消费者找到最佳价格，同时协助航空公司优化收入管理策略。通过分析影响航班价格的关键因素，构建机器学习模型以预测未来价格。

## 目录

- [项目概述](#项目概述)
- [数据结构](#数据结构)
- [如何运行代码](#如何运行代码)
- [项目组织](#项目组织)
- [依赖项](#依赖项)
- [参考文献](#参考文献)

## 项目概述

本项目包括以下主要任务：

1. **数据预处理**：清洗原始数据，处理缺失值、编码分类变量、缩放数值变量。
2. **特征选择与工程**：分析影响航班价格的因素，如航空公司、航线距离、出发时间、到达城市和座位等级等，选择相关特征。
3. **模型训练**：使用 LightGBM 模型训练数据，预测航班价格。
4. **模型评估与优化**：通过均方误差 (MSE)、平均绝对误差 (MAE) 和 R² 等指标评估模型性能，并进行优化。
5. **结果分析**：分析模型输出，理解影响航班价格的主要因素，并提出改进建议。

## 如何运行代码

按照以下步骤在本地运行项目：

1. 克隆代码库：

   ```bash
   git clone https://github.com/yourusername/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. 安装所需依赖项：

   ```bash
   pip install -r requirements.txt
   ```

3. 数据预处理：
   - 打开 `notebooks/1_data_preprocessing.ipynb`。
   - 运行所有单元格以清洗和预处理数据。

4. 特征选择：
   - 打开 `notebooks/2_feature_selection.ipynb`。
   - 运行所有单元格以选择模型的相关特征。

5. 模型训练：
   - 打开 `notebooks/3_model_training.ipynb`。
   - 运行所有单元格以训练 LightGBM 模型。

6. 结果分析：
   - 打开 `notebooks/4_result_analysis.ipynb`。
   - 运行所有单元格以可视化模型性能并分析特征重要性。

## 项目组织

### `data/`

- `raw/`：包含原始数据集文件（CSV 格式）。
- `processed/`：包含清洗和预处理后的数据。

### `notebooks/`

- 包含 Jupyter Notebook 文件：
  - `1_data_preprocessing.ipynb`：数据清洗与预处理。
  - `2_feature_selection.ipynb`：特征选择与工程。
  - `3_model_training.ipynb`：模型训练。
  - `4_result_analysis.ipynb`：结果分析与可视化。

### `src/`

- 包含 Python 源代码文件：
  - `data_preprocessing.py`：数据清洗与预处理模块。
  - `feature_engineering.py`：特征选择与工程模块。
  - `model.py`：模型构建与训练模块。
  - `evaluation.py`：模型评估模块。
  - `utils.py`：辅助函数模块。

### `reports/`

- 包含项目报告和参考文献。

### `requirements.txt`

- 列出了运行项目所需的 Python 库。

### `.gitignore`

- 指定 Git 应忽略的文件和目录。

## 依赖项

运行项目需要以下 Python 库：

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- lightgbm
- jupyter

通过以下命令安装所有依赖项：

```bash
pip install -r requirements.txt
```

## 参考文献

1. Abdella J A, Zaki N M, Shuaib K, et al. "Airline ticket price and demand prediction: A survey." *Journal of King Saud University-Computer and Information Sciences*, 2021, 33(4): 375-391.
2. Panigrahi A, Sharma R, Chakravarty S, et al. "Flight Price Prediction Using Machine Learning." *ACI@ ISIC*, 2022: 172-178.
3. Samunderu E, Farrugia M. "Predicting customer purpose of travel in a low-cost travel environment—A Machine Learning Approach." *Machine Learning with Applications*, 2022, 9: 100379.

## 许可证

本项目基于 MIT 许可证开源，详情请参阅 [LICENSE](LICENSE) 文件。
