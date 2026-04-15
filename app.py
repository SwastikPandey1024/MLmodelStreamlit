"""
Streamlit Sales Prediction App
==============================
A robust machine learning application for predicting daily sales using a trained XGBoost model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Sales Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ML Sales Prediction Model"}
)

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load trained model and expected feature columns."""
    try:
        with open("model.pkl", "rb") as f:
            loaded_content = joblib.load(f)
        
        model = loaded_content['model_pipeline']
        features = loaded_content['expected_feature_columns']
        
        st.success("✓ Model loaded successfully")
        return model, features
    
    except FileNotFoundError:
        st.error("❌ model.pkl not found. Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.stop()


model_pipeline, expected_columns = load_model()

# ============================================================================
# PAGE LAYOUT
# ============================================================================
st.title("📊 Sales Prediction App")
st.markdown("""
Predict daily sales using a trained **XGBoost** machine learning model.
Provide historical sales data and the model will forecast tomorrow's sales.
""")

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================
st.sidebar.header("📝 Input Features")

# Date selection
order_date = st.sidebar.date_input(
    "📅 Order Date",
    value=datetime.now()
)

# Historical sales data
st.sidebar.subheader("📈 Historical Sales (Lag Features)")
lag_1 = st.sidebar.number_input(
    "Sales (Yesterday - lag_1)",
    min_value=0.0,
    value=100.0,
    step=10.0,
    help="Sales value from 1 day ago"
)
lag_7 = st.sidebar.number_input(
    "Sales (7 days ago - lag_7)",
    min_value=0.0,
    value=100.0,
    step=10.0,
    help="Sales value from 7 days ago"
)
lag_14 = st.sidebar.number_input(
    "Sales (14 days ago - lag_14)",
    min_value=0.0,
    value=100.0,
    step=10.0,
    help="Sales value from 14 days ago"
)
lag_30 = st.sidebar.number_input(
    "Sales (30 days ago - lag_30)",
    min_value=0.0,
    value=100.0,
    step=10.0,
    help="Sales value from 30 days ago"
)

# Rolling statistics
st.sidebar.subheader("📊 Rolling Metrics")
rolling_mean_7 = st.sidebar.number_input(
    "7-Day Rolling Mean",
    min_value=0.0,
    value=100.0,
    step=10.0,
    help="Average sales over last 7 days"
)
rolling_mean_14 = st.sidebar.number_input(
    "14-Day Rolling Mean",
    min_value=0.0,
    value=100.0,
    step=10.0,
    help="Average sales over last 14 days"
)
rolling_std_7 = st.sidebar.number_input(
    "7-Day Rolling Std Dev",
    min_value=0.0,
    value=5.0,
    step=1.0,
    help="Standard deviation of sales over last 7 days"
)

# Trend
st.sidebar.subheader("📈 Trend")
trend = st.sidebar.number_input(
    "Trend Counter",
    min_value=0,
    value=100,
    step=1,
    help="Sequential counter (e.g., day number in dataset)"
)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def extract_date_features(date_input):
    """Extract numerical features from date."""
    try:
        date_obj = pd.Timestamp(date_input)
        return {
            "day": int(date_obj.day),
            "month": int(date_obj.month),
            "weekday": int(date_obj.weekday()),  # 0=Monday, 6=Sunday
            "is_weekend": 1 if int(date_obj.weekday()) >= 5 else 0,
            "weekofyear": int(date_obj.isocalendar().week)
        }
    except Exception as e:
        logger.error(f"Date feature extraction error: {str(e)}")
        return None


def create_input_dataframe(date_features, lag_1, lag_7, lag_14, lag_30,
                          rolling_mean_7, rolling_mean_14, rolling_std_7, trend):
    """
    Create input DataFrame with proper types for model prediction.
    
    Parameters:
    -----------
    date_features : dict
        Dictionary with date-derived features
    lag_* : float
        Lag features (historical sales)
    rolling_mean_* : float
        Rolling average metrics
    rolling_std_7 : float
        Rolling standard deviation
    trend : int
        Trend counter
    
    Returns:
    --------
    pd.DataFrame or None
        Input data with all features, properly typed as float64
    """
    try:
        if date_features is None:
            return None
        
        # Build input data with explicit float conversion
        input_dict = {
            "day": float(date_features["day"]),
            "month": float(date_features["month"]),
            "weekday": float(date_features["weekday"]),
            "is_weekend": float(date_features["is_weekend"]),
            "weekofyear": float(date_features["weekofyear"]),
            "lag_1": float(lag_1),
            "lag_7": float(lag_7),
            "lag_14": float(lag_14),
            "lag_30": float(lag_30),
            "rolling_mean_7": float(rolling_mean_7),
            "rolling_mean_14": float(rolling_mean_14),
            "rolling_std_7": float(rolling_std_7),
            "trend": float(trend)
        }
        
        # Create DataFrame
        df = pd.DataFrame(input_dict, index=[0])
        
        # Force all columns to float64 (CRITICAL for sklearn compatibility)
        df = df.astype("float64")
        
        # Reorder to match expected column order from training
        df = df[expected_columns]
        
        return df
    
    except Exception as e:
        logger.error(f"DataFrame creation error: {str(e)}")
        return None


# ============================================================================
# MAIN PREDICTION LOGIC
# ============================================================================
# Extract date features
date_features = extract_date_features(order_date)

# Create input DataFrame
input_data = create_input_dataframe(
    date_features, lag_1, lag_7, lag_14, lag_30,
    rolling_mean_7, rolling_mean_14, rolling_std_7, trend
)

# ============================================================================
# DISPLAY SECTION
# ============================================================================
st.markdown("---")

# Show input validation if data is ready
if input_data is not None:
    with st.expander("🔍 Input Data Validation", expanded=False):
        col_val1, col_val2 = st.columns(2)
        
        with col_val1:
            st.write("**Features:**")
            st.dataframe(input_data.T, use_container_width=True)
        
        with col_val2:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                "Feature": input_data.columns,
                "Type": input_data.dtypes.astype(str),
                "Value": input_data.iloc[0].values
            })
            st.dataframe(dtype_df, use_container_width=True)

# ============================================================================
# PREDICTION BUTTON AND RESULTS
# ============================================================================
col_btn, col_spacer = st.columns([1, 3])

with col_btn:
    predict_button = st.button(
        "🔮 Predict Sales",
        use_container_width=True,
        type="primary"
    )

if predict_button:
    if input_data is None:
        st.error("❌ Error: Failed to prepare input data. Please check your inputs.")
    else:
        try:
            # Make prediction
            prediction = model_pipeline.predict(input_data)
            predicted_sales = float(prediction[0])
            
            # Validate prediction
            if np.isnan(predicted_sales) or np.isinf(predicted_sales):
                st.error("❌ Invalid prediction value. Please review your inputs.")
            else:
                st.success("✓ Prediction Generated Successfully!")
                
                # Display main result
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    st.metric(
                        "📈 Predicted Sales",
                        f"${predicted_sales:.2f}",
                        delta=f"{(predicted_sales - lag_1):.2f}" if lag_1 > 0 else None,
                        delta_color="normal"
                    )
                
                with col_result2:
                    st.metric(
                        "📅 Prediction Date",
                        order_date.strftime('%b %d, %Y')
                    )
                
                with col_result3:
                    st.metric(
                        "🎯 Model",
                        "XGBoost",
                        f"{len(expected_columns)} features"
                    )
                
                # Context metrics
                st.markdown("---")
                st.subheader("📊 Context Metrics")
                
                ctx_col1, ctx_col2, ctx_col3, ctx_col4 = st.columns(4)
                
                with ctx_col1:
                    st.metric("Yesterday (lag_1)", f"${lag_1:.2f}")
                with ctx_col2:
                    st.metric("7-Day Avg", f"${rolling_mean_7:.2f}")
                with ctx_col3:
                    st.metric("14-Day Avg", f"${rolling_mean_14:.2f}")
                with ctx_col4:
                    pct_change = ((predicted_sales - lag_1) / lag_1 * 100) if lag_1 > 0 else 0
                    st.metric("% Change", f"{pct_change:.1f}%")
        
        except ValueError as ve:
            st.error(f"❌ **Data Type Error**: {str(ve)}")
            st.warning("⚠️ Ensure all inputs are valid numbers.")
            logger.error(f"ValueError: {str(ve)}")
        
        except Exception as e:
            st.error(f"❌ **Prediction Failed**: {str(e)}")
            st.info("Please verify all inputs and try again.")
            logger.error(f"Prediction exception: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em;'>
💡 <strong>Tip:</strong> Use realistic values from your recent sales history for accurate predictions.
</div>
""", unsafe_allow_html=True)