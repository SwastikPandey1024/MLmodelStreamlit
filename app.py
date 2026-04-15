import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Sales Prediction", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("📊 Sales Prediction App")
st.markdown("Predict daily sales using a trained **XGBoost machine learning model**")

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and configuration."""
    try:
        loaded_content = joblib.load("model.pkl")
        return loaded_content['model_pipeline'], loaded_content['expected_feature_columns']
    except FileNotFoundError:
        st.error("❌ **model.pkl** not found. Please run the notebook first to generate it.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model_pipeline, expected_columns = load_model()
st.success("✓ Model loaded successfully")

# ============================================================================
# USER INPUT SECTION
# ============================================================================
st.sidebar.header("📝 Prediction Input")

# Date input
order_date = st.sidebar.date_input("📅 Order Date", value=datetime.now())

# Numeric feature inputs grouped logically
st.sidebar.subheader("Historical Sales Data")
lag_1 = st.sidebar.number_input("Sales (Yesterday - lag_1)", min_value=0.0, value=100.0, step=10.0)
lag_7 = st.sidebar.number_input("Sales (7 days ago - lag_7)", min_value=0.0, value=100.0, step=10.0)
lag_14 = st.sidebar.number_input("Sales (14 days ago - lag_14)", min_value=0.0, value=100.0, step=10.0)
lag_30 = st.sidebar.number_input("Sales (30 days ago - lag_30)", min_value=0.0, value=100.0, step=10.0)

st.sidebar.subheader("Rolling Metrics")
rolling_mean_7 = st.sidebar.number_input("7-Day Rolling Mean", min_value=0.0, value=100.0, step=10.0)
rolling_mean_14 = st.sidebar.number_input("14-Day Rolling Mean", min_value=0.0, value=100.0, step=10.0)
rolling_std_7 = st.sidebar.number_input("7-Day Rolling Std Dev", min_value=0.0, value=5.0, step=1.0)

st.sidebar.subheader("Trend")
trend = st.sidebar.number_input("Trend Counter (sequential)", min_value=0, value=100, step=1)

# ============================================================================
# FEATURE ENGINEERING AND VALIDATION
# ============================================================================
def prepare_prediction_data(date, lag_1, lag_7, lag_14, lag_30, rolling_mean_7, 
                           rolling_mean_14, rolling_std_7, trend, expected_cols):
    """
    Prepare and validate prediction data.
    
    Args:
        date: pandas-compatible date
        lag_1-lag_30: historical sales values
        rolling_mean_7-rolling_std_7: rolling statistics
        trend: trend counter
        expected_cols: list of column names from training
        
    Returns:
        DataFrame with proper types and column order, or None if validation fails
    """
    try:
        # Convert date to pandas Timestamp
        date_obj = pd.Timestamp(date)
        
        # Extract date features
        day = int(date_obj.day)
        month = int(date_obj.month)
        weekday = int(date_obj.weekday())  # 0=Monday, 6=Sunday
        is_weekend = 1 if weekday >= 5 else 0
        weekofyear = int(date_obj.isocalendar().week)
        
        # Create initial data dictionary with numeric features
        data = {
            "day": day,
            "month": month,
            "weekday": weekday,
            "lag_1": float(lag_1),
            "lag_7": float(lag_7),
            "lag_14": float(lag_14),
            "lag_30": float(lag_30),
            "rolling_mean_7": float(rolling_mean_7),
            "rolling_mean_14": float(rolling_mean_14),
            "rolling_std_7": float(rolling_std_7),
            "is_weekend": is_weekend,
            "weekofyear": weekofyear,
            "trend": int(trend)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Add any missing categorical columns with default string values
        # This prevents type mismatches during preprocessing
        for col in expected_cols:
            if col not in df.columns:
                # For unknown categorical columns, use a default category
                df[col] = "Unknown"
        
        # Ensure numeric columns are float type (critical for model compatibility)
        numeric_cols = ["day", "month", "weekday", "lag_1", "lag_7", "lag_14", "lag_30",
                       "rolling_mean_7", "rolling_mean_14", "rolling_std_7", 
                       "is_weekend", "weekofyear", "trend"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Reorder columns to exactly match training data
        df = df[expected_cols]
        
        return df
        
    except Exception as e:
        logger.error(f"Error in prepare_prediction_data: {str(e)}")
        return None


# ============================================================================
# DISPLAY AND PREDICT
# ============================================================================
# Prepare input data
input_data = prepare_prediction_data(
    order_date, lag_1, lag_7, lag_14, lag_30, 
    rolling_mean_7, rolling_mean_14, rolling_std_7, 
    trend, expected_columns
)

if input_data is not None:
    # Display input data with debugging information
    with st.expander("📋 Input Data Validation", expanded=False):
        st.write("**Raw Input Features:**")
        st.dataframe(input_data, use_container_width=True)
        
        st.write("**Data Types Check:**")
        dtype_info = pd.DataFrame({
            "Column": input_data.columns,
            "Data Type": input_data.dtypes.astype(str)
        })
        st.dataframe(dtype_info, use_container_width=True)

# Prediction button
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔮 Predict Sales", use_container_width=True):
        if input_data is None:
            st.error("❌ Data preparation failed. Please check your inputs.")
        else:
            try:
                # Make prediction
                prediction = model_pipeline.predict(input_data)
                predicted_sales = float(prediction[0])
                
                # Display result
                st.success("✓ Prediction Successful!")
                
                # Large prominent display of prediction
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.metric("📈 Predicted Sales", f"${predicted_sales:.2f}")
                
                with col_result2:
                    st.info(f"📅 Date: {order_date.strftime('%B %d, %Y')}\n\n"
                           f"⚙️ Model: XGBoost\n\n"
                           f"🔢 Features: {len(expected_columns)}")
                
                # Additional context
                st.markdown("---")
                st.subheader("💡 Prediction Context")
                col_context1, col_context2, col_context3 = st.columns(3)
                with col_context1:
                    st.metric("Previous Day (lag_1)", f"${lag_1:.2f}")
                with col_context2:
                    st.metric("7-Day Avg", f"${rolling_mean_7:.2f}")
                with col_context3:
                    st.metric("14-Day Avg", f"${rolling_mean_14:.2f}")
                
            except ValueError as e:
                st.error(f"❌ **Data Type Error**: {str(e)}")
                st.warning("⚠️ Please ensure all numerical inputs are valid numbers.")
                logger.error(f"ValueError during prediction: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ **Prediction Error**: {str(e)}")
                st.info("Please verify all inputs are correctly filled and try again.")
                logger.error(f"Exception during prediction: {str(e)}")

with col2:
    st.info("← Click the button to generate a prediction")