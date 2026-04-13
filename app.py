import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Title
st.title("📊 Sales Prediction App")
st.write("Predict sales using trained XGBoost model")

# Load model and expected columns
try:
    loaded_content = joblib.load("model.pkl")
    model_pipeline = loaded_content['model_pipeline']
    expected_columns = loaded_content['expected_feature_columns']
    st.success("✓ Model loaded successfully")
except FileNotFoundError:
    st.error("❌ model.pkl not found. Please run the notebook first to generate it.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Enter Sales Prediction Features")

def user_input():
    # Get current date
    order_date = st.sidebar.date_input("Order Date", datetime.now())
    lag_1 = st.sidebar.number_input("Sales (Yesterday - lag_1)", 0.0, step=10.0)
    lag_7 = st.sidebar.number_input("Sales (7 days ago - lag_7)", 0.0, step=10.0)
    lag_14 = st.sidebar.number_input("Sales (14 days ago - lag_14)", 0.0, step=10.0)
    lag_30 = st.sidebar.number_input("Sales (30 days ago - lag_30)", 0.0, step=10.0)
    rolling_mean_7 = st.sidebar.number_input("Rolling Mean (7 days)", 0.0, step=10.0)
    rolling_mean_14 = st.sidebar.number_input("Rolling Mean (14 days)", 0.0, step=10.0)
    rolling_std_7 = st.sidebar.number_input("Rolling Std Dev (7 days)", 0.0, step=1.0)
    trend = st.sidebar.number_input("Trend (sequential counter)", 0, step=1)

    # Calculate date features
    order_date_obj = pd.Timestamp(order_date)
    day = order_date_obj.day
    month = order_date_obj.month
    weekday = order_date_obj.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    weekofyear = order_date_obj.isocalendar().week

    # Create DataFrame with all required features
    data = {
        "day": day,
        "month": month,
        "weekday": weekday,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_30": lag_30,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_14": rolling_mean_14,
        "rolling_std_7": rolling_std_7,
        "is_weekend": is_weekend,
        "weekofyear": weekofyear,
        "trend": trend
    }
    return pd.DataFrame(data, index=[0]), order_date

input_df, selected_date = user_input()

# Show inputs
st.subheader("Input Features:")
st.write(input_df)

# Add missing categorical columns if they exist in expected columns
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Default value for missing categorical features

# Reorder columns to match training data
input_df = input_df[expected_columns]

# Prediction
if st.button("🔮 Predict Sales"):
    try:
        prediction = model_pipeline.predict(input_df)
        st.success(f"✓ Predicted Sales for {selected_date}: **${prediction[0]:.2f}**")
        
        # Show prediction confidence
        st.info(f"Model: XGBoost with {len(expected_columns)} features")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")