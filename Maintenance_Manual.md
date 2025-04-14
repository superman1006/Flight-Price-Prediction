# 📘 Flight Price Prediction - Maintenance Manual

This Maintenance Manual provides developers and maintainers with essential instructions for sustaining, debugging, and extending the Flight Price Prediction System over time. This system is built with Python and is structured around a modular machine learning pipeline and a Flask-based deployment interface.

---

## 🛠️ 1. System Overview

The system includes the following components:

- **Data Preprocessing**: Cleaning, normalizing, encoding, and transforming raw flight datasets.
- **Feature Selection & Engineering**: Selecting statistically and business-relevant features.
- **Model Training**: Training LightGBM and Linear Regression models for price prediction.
- **Model Evaluation**: Testing and performance validation.
- **Deployment**: A RESTful API using Flask for serving prediction results.
- **Visualization & Interpretation**: SHAP values, correlation maps, etc.

---

## 🔄 2. Maintenance Tasks

### 2.1 Data Updates

**Steps:**
1. Place new raw data into `data/raw/`.
2. Run `1_data_preprocessing.ipynb` to clean and preprocess.
3. Update processed files in `data/processed/`.

**Notes:**
- Ensure column formats match original structure.
- Update any feature engineering logic if the new dataset schema changes.

---

### 2.2 Model Retraining

**Steps:**
1. Execute `2_feature_selection.ipynb` and `3_model_training.ipynb`.
2. Check evaluation metrics (R², MAE, MSE).
3. Save new model as `flight_price_model.pkl`.

**Notes:**
- Update hyperparameters if model performance degrades.
- Ensure training/validation split maintains 80/20 ratio.

---

### 2.3 API Maintenance

**Location:** `api.py`

**Tasks:**
- Restart Flask API if file structure or model changes.
- Ensure `joblib.load()` points to the correct model path.
- Monitor memory usage under load.

**Versioning Tip:** Update endpoint (e.g. `/predict/v2`) if major changes occur.

---

### 2.4 Error Logging & Debugging

**Where to Check:**
- Flask error logs
- JSON schema validation in API input
- Traceback errors from model predictions

**Enhancements:**
- Add `try-except` around model predict block
- Log invalid input to a `.log` file

---

### 2.5 Dependencies

**File:** `requirements.txt`

**Update Process:**
1. Install new package with `pip install xxx`.
2. Freeze current environment:
   ```
   pip freeze > requirements.txt
   ```

**Common packages:**
- `pandas`, `scikit-learn`, `lightgbm`, `joblib`, `matplotlib`, `seaborn`, `flask`

---

## 🧪 3. Test Protocol

1. Unit test preprocessing (check null values, encoded features).
2. Validate model outputs against baseline predictions.
3. Check that the API returns a valid JSON with expected price range.

---

## 🧱 4. Folder Structure Consistency

Maintain the following structure:

```
project-root/
├── data/
├── notebooks/
├── src/
├── reports/
├── api.py
├── requirements.txt
```

Never remove `src/` modules or modify function names unless updated in all notebooks.

---

## 👥 5. Roles & Responsibilities

| Role              | Responsibility                        |
|-------------------|----------------------------------------|
| Maintainer        | Oversees model integrity & deployment |
| Data Engineer     | Updates raw data & preprocessing logic |
| API Developer     | Maintains REST interface and endpoints |
| Analyst           | Interprets model results and SHAP logs |

---

## 📤 6. Backup & Version Control

- Use Git for all source files.
- Commit after each significant notebook run or API update.
- Tag versions (e.g. `v1.0.0`) when model is finalized.
- Back up model files (`*.pkl`) and processed datasets.

---

## 🛡️ 7. Future Maintenance Tips

- Add Docker support for reproducible deployment.
- Use `mlflow` or `DVC` for model versioning and tracking.
- Schedule model refresh using Airflow or CronJobs if necessary.
- Add test cases with `pytest` for all critical modules.

---

Maintained by: **Chen Lv**  
Email: `a35lc21@abdn.ac.uk`  
Graduation Year: 2025  
University of Aberdeen & SCNU Joint Institute