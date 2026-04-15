# 📊 SalesPulse AI – Intelligent Sales Analytics Dashboard

A **production-ready AI analytics platform** for sales forecasting and business intelligence. Powered by XGBoost machine learning, Streamlit, and advanced time-series analysis.

---

## 🎯 Overview

**SalesPulse AI** transforms raw sales data into actionable intelligence. Predict daily sales, forecast 7-day trends, analyze patterns, and make data-driven decisions with confidence.

### Key Features

✅ **Single-Day Predictions** – AI-powered sales forecasting with 13 engineered features  
✅ **7-Day Forecasting** – Iterative predictions with trend analysis  
✅ **KPI Dashboard** – Real-time metrics and feature importance visualization  
✅ **Batch Processing** – Upload CSV files for large-scale predictions  
✅ **Business Insights** – Automatic analysis and actionable recommendations  
✅ **Professional UI** – Clean, intuitive design with responsive layout  
✅ **Production Grade** – Strict validation, error handling, and logging  

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone/Download the project**
   ```bash
   cd SalesPulse_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## 📊 Features & Capabilities

### 1️⃣ Single Prediction Tab

Predict sales for a specific date with instant results.

**Inputs:**
- 📅 Prediction date
- 📈 Lag features (sales 1, 7, 14, 30 days ago)
- 📊 Rolling metrics (7-day & 14-day averages)
- 📈 Trend counter

**Outputs:**
- 🔮 Predicted sales value
- 📊 Trend visualization (lag_30 → predicted)
- 📈 Comparison bar chart
- 💡 Business insights & recommendations
- 📥 CSV download

**Business Logic:**
```python
if prediction > lag_1 × 1.05:
    → "Sales expected to GROW" ✅
elif prediction < lag_1 × 0.95:
    → "Sales may DECLINE" ⚠️
else:
    → "Sales STABLE" ℹ️
```

---

### 2️⃣ Trend Analysis Tab

Generate 7-day forecast with iterative predictions and trend analysis.

**Forecasting Algorithm:**
1. Make prediction for Day 1
2. Use Day 1 prediction as lag_1 for Day 2
3. Update rolling metrics based on recent predictions
4. Repeat for Days 3-7

**Outputs:**
- 📊 Line chart with 7-day forecast
- 📋 Detailed forecast table
- 📈 Summary statistics (avg, min, max, trend)
- 📥 CSV download

---

### 3️⃣ KPI Dashboard

Monitor key performance indicators and model explainability.

**KPI Metrics:**
- 💰 Yesterday's sales
- 📊 7-day average
- 📈 14-day average
- 🎯 Predicted sales
- 📊 Feature importance ranking

**Feature Importance:**
Visualizes which features have the most impact on predictions.
- Top 5 most important features
- Full feature ranking
- Importance distribution analysis

---

### 4️⃣ Batch Predictions

Upload CSV files for bulk sales predictions.

**Required CSV Columns:**
```
order_date          (YYYY-MM-DD format)
lag_1, lag_7, lag_14, lag_30  (Historical sales)
rolling_mean_7, rolling_mean_14, rolling_std_7  (Rolling metrics)
trend               (Sequential counter)
```

**Example CSV:**
```
order_date,lag_1,lag_7,lag_14,lag_30,rolling_mean_7,rolling_mean_14,rolling_std_7,trend
2024-01-15,150.50,145.30,148.20,140.00,147.50,145.00,3.50,100
2024-01-16,155.20,150.50,145.30,148.20,150.25,146.00,4.20,101
```

**Outputs:**
- ✅ Success/error status for each row
- 📊 Summary statistics
- 📋 Results table
- 📥 CSV download with predictions

---

## 🎨 User Interface

### Tab Structure

| Tab | Purpose |
|-----|---------|
| 🔮 Single Prediction | One-day forecast with insights |
| 📈 Trend Analysis | 7-day forecast & trend visualization |
| 📊 Dashboard | KPI metrics & model explainability |
| 📁 Batch Predictions | Bulk CSV processing |

### Design Features

✨ **Professional Layout**
- Clean, modern design with Streamlit
- Responsive columns for mobile/desktop
- Organized sections with clear headings

🎨 **Color Scheme**
- Primary: #1f77b4 (Professional blue)
- Background: #f5f7fa (Light gray)
- Status: ✅ Green, ⚠️ Yellow, ❌ Red

📱 **Interactive Components**
- Number inputs with validation
- Date pickers with defaults
- File uploaders for CSV
- Download buttons for results

---

## 🧠 Model Architecture

### XGBoost Model

**Algorithm:** XGBoost Regressor  
**Training Samples:** 7,971 rows  
**Test MAE:** $258.84  
**Features:** 13 engineered numerical features

### Engineered Features (13 Total)

#### Date Features (5)
```
day          → Day of month (1-31)
month        → Month (1-12)
weekday      → Day of week (0=Mon, 6=Sun)
is_weekend   → Binary flag (1=weekend, 0=weekday)
weekofyear   → Week number (1-53)
```

#### Lag Features (4)
```
lag_1        → Sales from 1 day ago
lag_7        → Sales from 7 days ago
lag_14       → Sales from 14 days ago
lag_30       → Sales from 30 days ago
```

#### Rolling Statistics (3)
```
rolling_mean_7      → 7-day average sales
rolling_mean_14     → 14-day average sales
rolling_std_7       → 7-day standard deviation
```

#### Trend Feature (1)
```
trend        → Sequential counter (captures long-term growth/decline)
```

### Training Process

1. Load data from `Sample - Superstore.csv`
2. Feature engineering with date extraction & rolling calculations
3. Train-test split (80-20, time-ordered)
4. XGBoost training with optimized hyperparameters
5. Model evaluation with MAE metric
6. Model serialization to `model.pkl`

---

## 🛡️ Validation & Error Handling

### 3-Layer Validation Pipeline

#### Layer 1: Input Validation
```python
validate_numeric_input()
├─ Type conversion (to float)
├─ NaN/Infinity check
├─ Range validation
└─ Clear error messages
```

#### Layer 2: DataFrame Validation
```python
create_input_dataframe_strict()
├─ Individual value validation
├─ DataFrame dtype enforcement (all float64)
├─ Feature count verification
├─ NaN/Infinity sweep
└─ Feature order matching
```

#### Layer 3: Model Validation
```python
make_prediction()
├─ Pre-prediction data checks
├─ Prediction output validation
├─ Result sanity checks
└─ Exception handling
```

### Guarantees

✅ **No Silent Corruption** – All errors reported immediately  
✅ **No Object Dtypes** – All features converted to float64  
✅ **No Invalid Inputs** – Range-checked before model  
✅ **No Type Errors** – Strict type enforcement throughout  
✅ **Traceable Errors** – Logging of all validation steps

---

## 📁 Project Structure

```
SalesPulse_AI/
├── app.py                           # Main Streamlit application
├── model.pkl                        # Trained XGBoost model + features
├── retrain_model.py                 # Model retraining script
├── ML_minor_project.ipynb           # Jupyter notebook with analysis
├── Sample - Superstore.csv          # Training dataset
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── __pycache__/                     # Python cache files
```

---

## 🔧 Dependencies

```
streamlit==1.28.0          # Web app framework
pandas==2.0.0              # Data manipulation
numpy==1.24.0              # Numerical computing
xgboost==2.0.0             # Machine learning model
joblib==1.3.0              # Model serialization
scikit-learn==1.3.0        # ML utilities
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📊 Example Workflow

### 1. Single Prediction

**Input:**
- Date: 2024-01-20
- Sales Yesterday: $150
- 7-Day Average: $145
- 14-Day Average: $148

**Output:**
```
Predicted Sales: $155.42
% Change: +3.6%
vs 7-Day Avg: +$10.42

💡 Insights:
  📈 Strong growth expected
  Recommendation: Scale inventory and staffing
```

### 2. 7-Day Forecast

**Input:** Today's sales data

**Output:**
```
Day 1: $155.42
Day 2: $158.20
Day 3: $161.50
Day 4: $159.80
Day 5: $162.10
Day 6: $165.40
Day 7: $168.90

📈 7-Day Trend: +$13.48
Average: $160.34
```

### 3. Batch Upload

**Input:** CSV with 100 rows

**Output:**
```
✅ Total Rows: 100
✅ Successful: 98
❌ Errors: 2

[Table with all predictions and error messages]
```

---

## 🚀 Deployment

### Local Development

```bash
streamlit run app.py
```

### Production Deployment

**Streamlit Cloud:**
```bash
streamlit run app.py --logger.level=error
```

**Docker:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Algorithm | XGBoost |
| Mean Absolute Error (MAE) | $258.84 |
| Training Samples | 7,971 |
| Features Used | 13 |
| Feature Dtypes | All float64 |
| Validation Layers | 3 |
| Error Coverage | 100% |

---

## 🔄 Model Retraining

Update the model with new data:

```bash
python retrain_model.py
```

**Process:**
1. Loads latest `Sample - Superstore.csv`
2. Recreates engineered features
3. Trains new XGBoost model
4. Evaluates on test set
5. Saves to `model.pkl`
6. Logs all metrics

---

## 🐛 Troubleshooting

### Issue: "model.pkl not found"
**Solution:** Run the Jupyter notebook or `retrain_model.py` to train the model first.

### Issue: "Feature mismatch error"
**Solution:** Ensure all 13 features are present in input. Use the batch upload with proper CSV format.

### Issue: App crashes with "object dtype"
**Solution:** All inputs must be numeric. Check CSV for string/text values.

### Issue: Predictions are negative
**Solution:** This is unusual but possible. Check if lag values are realistic. Model may need retraining with new data.

---

## 📚 Technical Documentation

### Feature Engineering

```python
df['day'] = df['Order Date'].dt.day
df['month'] = df['Order Date'].dt.month
df['weekday'] = df['Order Date'].dt.weekday
df['lag_1'] = df['Sales'].shift(1)
df['rolling_mean_7'] = df['Sales'].rolling(7).mean()
df['trend'] = range(len(df))
```

### Model Training

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
```

### Prediction Pipeline

```python
# 1. Validate inputs
date_features = extract_date_features(date)
input_df = create_input_dataframe_strict(...)

# 2. Make prediction
prediction = model.predict(input_df)

# 3. Generate insights
insights = generate_insights(prediction, lag_1, rolling_7, rolling_14)
```

---

## 🎯 Use Cases

### Sales Management
- Daily sales forecasting
- Inventory planning
- Revenue projection
- Performance tracking

### Business Strategy
- Trend analysis
- Seasonality detection
- Growth monitoring
- Risk assessment

### Operational Planning
- Staff scheduling
- Resource allocation
- Marketing timing
- Promotion optimization

---

## 📝 License & Credits

**Created:** MLminorStreamlit Project  
**Built with:** Streamlit, XGBoost, Pandas, NumPy  
**Data Source:** Sample Superstore Dataset  

---

## 📧 Support & Feedback

For issues, suggestions, or improvements:
- Check the troubleshooting section
- Review error messages in logs
- Validate input data format
- Run `retrain_model.py` to update the model

---

## 🎓 Learning Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **XGBoost Guide:** https://xgboost.readthedocs.io
- **Time Series Forecasting:** https://pandas.pydata.org/docs
- **Machine Learning:** https://scikit-learn.org/stable

---

## ✨ Version History

### v2.0 (Current) - Professional Dashboard
- ✅ Single prediction tab
- ✅ 7-day forecasting
- ✅ KPI dashboard with feature importance
- ✅ Batch CSV predictions
- ✅ Business insights engine
- ✅ Professional UI with responsive layout
- ✅ Advanced error handling & validation

### v1.0 - Basic Prediction
- Single prediction interface
- Basic validation
- Simple UI layout

---

## 🚀 Future Enhancements

🔄 Real-time data integration  
📊 Advanced analytics & dashboards  
🤖 Ensemble model comparison  
💾 Model versioning & A/B testing  
📧 Automated alerts & notifications  
📱 Mobile app  

---

**Made with ❤️ for data-driven decision making.**

---

*Last Updated: April 16, 2026*  
*SalesPulse AI – Intelligent Sales Analytics Platform*
