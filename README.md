
# Flight Price Prediction using Linear Regression

This project aims to predict flight prices using linear regression techniques. The goal is to help consumers find the best prices and assist airlines in optimizing their revenue management strategies. By predicting flight prices, this project explores the key determinants affecting airfare and builds a model to forecast future prices.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Structure](#data-structure)
- [How to Run the Code](#how-to-run-the-code)
- [Project Organization](#project-organization)
- [Requirements](#requirements)
- [References](#references)

## Project Overview

The project involves the following major tasks:
1. **Data Collection & Feature Selection**: Identify and evaluate factors such as airline, route distance, flight time, departure and destination cities, and seat class that impact flight costs.
2. **Model Construction**: Develop a linear regression model to predict flight prices and explore the relationships between various features and the prices.
3. **Model Evaluation & Optimization**: Evaluate the model using metrics like Mean Squared Error (MSE) and R², and refine it to enhance performance.
4. **Results Analysis**: Analyze the model’s output to understand the main determinants of flight prices and suggest potential improvements.



## How to Run the Code

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess the data:
   - Open `notebooks/1_data_preprocessing.ipynb` in Jupyter Notebook.
   - Run all cells to clean and preprocess the data.

4. Perform feature selection:
   - Open `notebooks/2_feature_selection.ipynb`.
   - Run all cells to select relevant features for the model.

5. Train the linear regression model:
   - Open `notebooks/3_model_training.ipynb`.
   - Run all cells to train the model on the selected features.

6. Analyze the results:
   - Open `notebooks/4_result_analysis.ipynb`.
   - Run all cells to visualize the model’s performance and analyze the impact of different features on flight prices.

## Project Organization

### `data/`
- `raw/` contains the original dataset files in CSV format.
- `processed/` contains the cleaned and preprocessed data after handling missing values, encoding categorical features, and scaling numerical variables.
- `external/` can contain external data sources, if applicable.

### `notebooks/`
- Jupyter Notebooks for each major phase of the project: data preprocessing, feature selection, model training, and results analysis.

### `src/`
- Python source code files for each module in the project:
  - `data_preprocessing.py` handles data cleaning and preprocessing.
  - `feature_engineering.py` performs feature selection and engineering.
  - `model.py` builds and trains the linear regression model.
  - `evaluation.py` evaluates the model using performance metrics.
  - `utils.py` contains utility functions to assist with various tasks.

### `reports/`
- The thesis proposal and final report PDFs, as well as a Bibtex file containing references.

### `requirements.txt`
- Lists the necessary Python libraries to run the project. It includes packages like pandas, scikit-learn, numpy, and matplotlib.

### `.gitignore`
- Specifies which files and directories Git should ignore, including data files and other non-source files.

## Requirements

To run the project, you'll need the following Python libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install all the dependencies by running:
```bash
pip install -r requirements.txt
```

## References

1. Abdella J A, Zaki N M, Shuaib K, et al. "Airline ticket price and demand prediction: A survey." *Journal of King Saud University-Computer and Information Sciences*, 2021, 33(4): 375-391.
2. Panigrahi A, Sharma R, Chakravarty S, et al. "Flight Price Prediction Using Machine Learning." *ACI@ ISIC*, 2022: 172-178.
3. Samunderu E, Farrugia M. "Predicting customer purpose of travel in a low-cost travel environment—A Machine Learning Approach." *Machine Learning with Applications*, 2022, 9: 100379.
4. He K, Zhang X, Ren S, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016: 770-778.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

请按原样复制，确保没有渲染，保持纯文本格式。这应该可以避免任何格式上的自动渲染问题。如果有其他需要，随时告诉我！