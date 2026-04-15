📊 MLmodelStreamlit

A Machine Learning-powered Streamlit web application that predicts Superstore sales using advanced regression techniques. This project demonstrates end-to-end ML workflow — from data preprocessing to deployment.

🚀 Project Overview

This project focuses on building a predictive analytics solution using machine learning and deploying it via an interactive Streamlit dashboard.

It transforms raw retail data into actionable insights and real-time predictions, making it highly useful for business decision-making and forecasting.

🎯 Key Features

✔️ Interactive UI for user input
✔️ Real-time prediction using trained ML model
✔️ Clean and intuitive dashboard with Streamlit
✔️ End-to-end pipeline: Data → Model → Deployment
✔️ Scalable and deployment-ready architecture
✔️ **CRITICAL: Production-grade strict input validation (no silent data corruption)**
✔️ **Engineered features only (13 numerical features, zero raw dataset columns)**

🧠 Tech Stack
Python 🐍
Pandas, NumPy – Data Processing
Scikit-learn / XGBoost – Machine Learning
Streamlit – Web App Deployment
Joblib – Model Serialization
Git & GitHub – Version Control

---

## 📊 Model Architecture

### Engineered Features (13 total)
The model is trained on **ONLY engineered numerical features** — no raw dataset columns:

1. **Date-based**: `day`, `month`, `weekday`, `weekofyear`, `is_weekend`
2. **Lag features**: `lag_1`, `lag_7`, `lag_14`, `lag_30` (historical sales)
3. **Rolling metrics**: `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`
4. **Trend**: `trend` (time counter)

### Why This Approach?
✅ **Strict Feature Alignment** - Training and prediction use identical feature structure
✅ **No Silent Data Corruption** - Invalid inputs rejected immediately (not converted to 0)
✅ **Type Safety** - All features guaranteed float64 (no object dtype)
✅ **Production Ready** - Pre-validated data reaches model

### Feature Matching
```python
# In model.pkl:
expected_feature_columns = [
    "day", "month", "weekday", "is_weekend", "weekofyear",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7", "trend"
]

# In app.py:
ENGINEERED_FEATURES = [  # MUST MATCH model.pkl exactly
    "day", "month", "weekday", "is_weekend", "weekofyear",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7", "trend"
]
```

**Validation**: App checks feature alignment on startup and fails if mismatch detected.

---

## 🔒 Input Validation (Production Grade)

### Three-Layer Validation
1. **Individual Input Validation** (`validate_numeric_input`)
   - Type checking
   - NaN/Infinity detection
   - Min/Max constraints
   - Fails fast with clear error

2. **Date Feature Extraction** (`extract_date_features`)
   - Safe date parsing
   - Range validation (day 1-31, month 1-12, etc.)
   - Raises ValueError on invalid input

3. **Strict DataFrame Creation** (`create_input_dataframe_strict`)
   - 8-step validation pipeline
   - Validates BEFORE conversion (no silent coercion)
   - Guarantees float64 only
   - Verifies feature order matches model

### Key Difference: Old vs New
```python
# ❌ OLD (Silent Data Corruption)
pd.to_numeric(errors='coerce')  # Invalid → NaN
fillna(0.0)                      # NaN → 0 (silent corruption!)

# ✅ NEW (Strict Validation)
validate_numeric_input()         # Check first, raise error immediately
# No silent conversions, no data corruption
```

See [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) for detailed validation architecture.

---

## 📁 Project Structure
```
MLmodelStreamlit/
│── app.py                      # Production Streamlit app
│── retrain_model.py            # Script to retrain model
│── model.pkl                   # Trained XGBoost (engineered features only)
│── ML_minor_project.ipynb      # Model development notebook
│── Sample - Superstore.csv     # Dataset
│── requirements.txt            # Dependencies
│── VALIDATION_GUIDE.md         # Validation architecture guide
│── README.md                   # This file
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain Model (Optional)
```bash
python retrain_model.py
```
This creates `model.pkl` with engineered features only.

### 3. Run Streamlit App
```bash
streamlit run app.py
```

### 4. Use the App
- Select order date in sidebar
- Enter historical sales data (lag features, rolling metrics)
- Click "Predict Sales" button
- View validation panel and prediction results

---

## 🔍 Validation Example

**Valid Input:**
```
Date: 2026-04-15
lag_1: 100.00
lag_7: 95.50
lag_14: 92.00
lag_30: 88.50
rolling_mean_7: 96.00
rolling_mean_14: 91.00
rolling_std_7: 3.50
trend: 500

Result: ✅ PASS
Output: "Predicted Sales: $322.40"
```

**Invalid Input (Negative lag):**
```
lag_1: -50.00  ← INVALID (must be ≥ 0)

Result: ❌ FAIL
Error: "lag_1: Value -50.0 is less than minimum 0"
Output: No prediction attempted
```

---

## 📊 Model Performance

- **Algorithm**: XGBoost (Gradient Boosting)
- **Training Samples**: 7,971
- **Test Samples**: 1,993
- **Mean Absolute Error (MAE)**: $258.84
- **Features**: 13 engineered features
- **Training Time**: ~2 minutes

---

## 🛡️ Production Guarantees

✅ **No Silent Data Corruption** - Invalid inputs rejected with clear error
✅ **No Type Errors** - All features validated to float64 before model
✅ **No Feature Mismatch** - App enforces exact feature alignment with model
✅ **Clear Error Messages** - Users know exactly what's wrong
✅ **Comprehensive Logging** - All errors logged for debugging
✅ **Defense-in-Depth** - Multiple validation layers catch edge cases

---

## 📝 Git History

Recent commits (feature alignment + strict validation):
- `03a2bce`: Production-grade strict input validation
- Previous: scipy.sparse fixes, type conversion improvements, model training

See full history: `git log`

---

## 🤝 Contributing

To retrain the model with new data:
1. Add data to `Sample - Superstore.csv`
2. Run `python retrain_model.py`
3. Verify model.pkl loads successfully
4. Test with Streamlit app
5. Commit changes to git

---

## ⚙️ Workflow

Data Collection → Feature Engineering → Model Training → Validation → Deployment

1. **Data**: Raw Superstore sales data
2. **Engineering**: Create date, lag, and rolling statistics features
3. **Training**: XGBoost on 13 engineered features
4. **Validation**: Strict 3-layer input validation
5. **Deployment**: Streamlit app with real-time predictions

---

## 📚 Documentation

- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) - Detailed validation architecture
- [ML_minor_project.ipynb](ML_minor_project.ipynb) - Model training notebook
- [app.py](app.py) - Streamlit application code
