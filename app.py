"""
Streamlit Sales Prediction App (Production Grade)
===================================================
A production-ready ML application for predicting daily sales using a trained XGBoost model.

VALIDATION APPROACH (Strict, No Silent Corruption):
- validate_numeric_input(): Validates each input before any conversion
- extract_date_features(): Safely extracts date components with range validation
- create_input_dataframe_strict(): Strict validation pipeline (no silent coercion)
  * Checks each value BEFORE conversion
  * Fails fast with clear error messages
  * Ensures exact feature match with training data
  * NEVER silently converts invalid inputs to 0
  * GUARANTEES all output is float64 with no object dtype

ERROR HANDLING:
- Input validation errors: Stop execution with clear message
- Prediction errors: Comprehensive error diagnosis
- Type errors: Detected and reported before model call

This approach ensures: No silent data corruption, No invalid predictions, Clean/verified data only
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
    """
    Extract numerical features from date.
    
    Returns dict with day, month, weekday, is_weekend, weekofyear.
    Raises ValueError if date cannot be parsed.
    """
    try:
        date_obj = pd.Timestamp(date_input)
        
        day = int(date_obj.day)
        month = int(date_obj.month)
        weekday = int(date_obj.weekday())  # 0=Monday, 6=Sunday
        is_weekend = 1 if weekday >= 5 else 0
        weekofyear = int(date_obj.isocalendar().week)
        
        # Validate ranges
        if not (1 <= day <= 31):
            raise ValueError(f"Day must be 1-31, got {day}")
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be 1-12, got {month}")
        if not (0 <= weekday <= 6):
            raise ValueError(f"Weekday must be 0-6, got {weekday}")
        
        return {
            "day": day,
            "month": month,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "weekofyear": weekofyear
        }
    
    except TypeError as te:
        raise ValueError(f"Invalid date format: {str(te)}")
    except Exception as e:
        raise ValueError(f"Date feature extraction error: {str(e)}")


def validate_numeric_input(value, name, min_val=None, max_val=None):
    """
    Strictly validate a single numeric input.
    
    Parameters:
    -----------
    value : any
        Value to validate
    name : str
        Name of the input (for error messages)
    min_val : float, optional
        Minimum allowed value
    max_val : float, optional
        Maximum allowed value
    
    Raises:
    -------
    ValueError if validation fails
    
    Returns:
    --------
    float : validated value
    """
    try:
        # Attempt conversion to float
        numeric_val = float(value)
        
        # Check for NaN and infinities
        if np.isnan(numeric_val):
            raise ValueError(f"{name}: NaN value not allowed")
        if np.isinf(numeric_val):
            raise ValueError(f"{name}: Infinite value not allowed")
        
        # Check range constraints
        if min_val is not None and numeric_val < min_val:
            raise ValueError(f"{name}: Value {numeric_val} is less than minimum {min_val}")
        if max_val is not None and numeric_val > max_val:
            raise ValueError(f"{name}: Value {numeric_val} exceeds maximum {max_val}")
        
        return numeric_val
    
    except (TypeError, ValueError) as e:
        if "could not convert" in str(e).lower():
            raise ValueError(f"{name}: Must be a valid number, got {value}")
        raise


def create_input_dataframe_strict(date_features, lag_1, lag_7, lag_14, lag_30,
                                  rolling_mean_7, rolling_mean_14, rolling_std_7, 
                                  trend, expected_cols):
    """
    Create input DataFrame with STRICT validation.
    
    NO silent coercion, NO data corruption.
    Fails fast with clear error messages.
    
    Parameters:
    -----------
    date_features : dict
        Validated date-derived features
    lag_* : float
        Historical sales values (must be numeric)
    rolling_mean_* : float
        Rolling average metrics (must be numeric)
    rolling_std_7 : float
        Rolling standard deviation (must be numeric)
    trend : int
        Trend counter (must be numeric)
    expected_cols : list
        Expected column names from training
    
    Returns:
    --------
    pd.DataFrame : Validated input data (all float64)
    
    Raises:
    -------
    ValueError : If any validation fails
    """
    try:
        if date_features is None:
            raise ValueError("Date features are None")
        
        # STEP 1: Validate each input value BEFORE any conversion
        # This prevents silent data corruption
        validated_data = {
            "day": validate_numeric_input(date_features["day"], "day", min_val=1, max_val=31),
            "month": validate_numeric_input(date_features["month"], "month", min_val=1, max_val=12),
            "weekday": validate_numeric_input(date_features["weekday"], "weekday", min_val=0, max_val=6),
            "is_weekend": validate_numeric_input(date_features["is_weekend"], "is_weekend", min_val=0, max_val=1),
            "weekofyear": validate_numeric_input(date_features["weekofyear"], "weekofyear", min_val=1, max_val=53),
            "lag_1": validate_numeric_input(lag_1, "lag_1", min_val=0),
            "lag_7": validate_numeric_input(lag_7, "lag_7", min_val=0),
            "lag_14": validate_numeric_input(lag_14, "lag_14", min_val=0),
            "lag_30": validate_numeric_input(lag_30, "lag_30", min_val=0),
            "rolling_mean_7": validate_numeric_input(rolling_mean_7, "rolling_mean_7", min_val=0),
            "rolling_mean_14": validate_numeric_input(rolling_mean_14, "rolling_mean_14", min_val=0),
            "rolling_std_7": validate_numeric_input(rolling_std_7, "rolling_std_7", min_val=0),
            "trend": validate_numeric_input(trend, "trend", min_val=0),
        }
        
        # STEP 2: Create DataFrame from validated data
        df = pd.DataFrame(validated_data, index=[0])
        
        # STEP 3: STRICT type conversion (no silent coercion)
        # Use astype instead of to_numeric to fail on conversion errors
        df = df.astype(np.float64)
        
        # STEP 4: Final validation - check for object dtypes
        object_cols = [col for col in df.columns if df[col].dtype == 'object']
        if object_cols:
            raise ValueError(f"Object dtype found in columns: {object_cols}")
        
        # STEP 5: Check for any NaN or infinities
        nan_mask = df.isnull()
        if nan_mask.any().any():
            nan_cols = df.columns[nan_mask.any()].tolist()
            raise ValueError(f"NaN values found in: {nan_cols}")
        
        inf_mask = ~np.isfinite(df.values)
        if inf_mask.any():
            inf_cols = df.columns[inf_mask.any()].tolist()
            raise ValueError(f"Infinite values found in: {inf_cols}")
        
        # STEP 6: Verify feature columns match expected training features
        actual_cols = list(df.columns)
        if actual_cols != expected_cols:
            raise ValueError(
                f"Feature mismatch:\n"
                f"Expected: {expected_cols}\n"
                f"Got: {actual_cols}"
            )
        
        # STEP 7: Reorder to match expected order (should already be correct)
        df = df[expected_cols]
        
        # STEP 8: Final sanity check
        if df.shape != (1, len(expected_cols)):
            raise ValueError(
                f"Shape mismatch: expected (1, {len(expected_cols)}), got {df.shape}"
            )
        
        return df
    
    except ValueError as ve:
        # Re-raise validation errors with context
        logger.error(f"Input validation failed: {str(ve)}")
        raise
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"Unexpected error in create_input_dataframe_strict: {str(e)}")
        raise ValueError(f"Input processing error: {str(e)}")


# ============================================================================
# MAIN PREDICTION LOGIC
# ============================================================================
# Extract and validate date features
try:
    date_features = extract_date_features(order_date)
except ValueError as date_error:
    date_features = None
    date_error_msg = str(date_error)
else:
    date_error_msg = None

# Create validated input DataFrame (only if date features are valid)
if date_features is not None:
    try:
        input_data = create_input_dataframe_strict(
            date_features, lag_1, lag_7, lag_14, lag_30,
            rolling_mean_7, rolling_mean_14, rolling_std_7, 
            trend, expected_columns
        )
    except ValueError as df_error:
        input_data = None
        df_error_msg = str(df_error)
else:
    input_data = None
    df_error_msg = date_error_msg
# ============================================================================
st.markdown("---")

# Show input validation if data is ready
if input_data is not None:
    with st.expander("✅ Input Data Validation (Valid)", expanded=False):
        col_val1, col_val2 = st.columns(2)
        
        with col_val1:
            st.write("**Features (Validated):**")
            st.dataframe(input_data.T, use_container_width=True)
        
        with col_val2:
            st.write("**Data Types (All Numeric):**")
            dtype_df = pd.DataFrame({
                "Feature": input_data.columns,
                "Type": input_data.dtypes.astype(str),
                "Value": input_data.iloc[0].values
            })
            st.dataframe(dtype_df, use_container_width=True)
            st.success("✅ All columns are float64 - Ready for prediction")
else:
    # Data is invalid - show error
    st.error(f"❌ **Input Validation Failed**")
    st.error(f"Error: {df_error_msg}")
    st.warning(
        "⚠️ Please review your inputs:\n"
        "- All fields must be valid numbers\n"
        "- Lag and rolling values must be non-negative\n"
        "- Date must be valid"
    )


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
        st.error("❌ **Cannot Predict**: Input validation failed")
        st.stop()
    
    try:
        # FINAL SAFETY CHECK before prediction
        # This should never fail because data was already validated,
        # but we check anyway for defense-in-depth
        
        # 1. Check for object dtypes
        object_cols = [col for col in input_data.columns if input_data[col].dtype == 'object']
        if object_cols:
            raise ValueError(f"Unexpected object dtype in: {object_cols}")
        
        # 2. Check for NaN
        if input_data.isnull().any().any():
            raise ValueError("Unexpected NaN values in input")
        
        # 3. Check for infinities
        if not np.isfinite(input_data.values).all():
            raise ValueError("Unexpected infinite values in input")
        
        # 4. Verify shape
        if input_data.shape != (1, len(expected_columns)):
            raise ValueError(f"Shape mismatch: got {input_data.shape}")
        
        # 5. Make prediction with validated data
        prediction = model_pipeline.predict(input_data)
        predicted_sales = float(prediction[0])
        
        # 6. Validate prediction output
        if np.isnan(predicted_sales):
            raise ValueError("Model returned NaN prediction")
        if np.isinf(predicted_sales):
            raise ValueError("Model returned infinite prediction")
        if predicted_sales < 0:
            st.warning("⚠️ Model predicted negative sales (unusual)")
        
        # SUCCESS - Display results
        st.success("✅ Prediction Generated Successfully!")
        
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
        
        logger.info(f"✓ Prediction successful: ${predicted_sales:.2f}")

    except ValueError as ve:
        st.error(f"❌ **Validation Error**: {str(ve)}")
        logger.error(f"Validation error during prediction: {str(ve)}")
        st.stop()
    
    except TypeError as te:
        st.error(f"❌ **Type Error**: {str(te)}")
        st.warning("⚠️ This indicates a data type incompatibility.")
        logger.error(f"TypeError during prediction: {str(te)}")
        st.stop()
    
    except Exception as e:
        st.error(f"❌ **Prediction Failed**: {str(e)}")
        logger.error(f"Unexpected error during prediction: {str(e)}")
        
        # Diagnose scipy.sparse error specifically
        if "scipy.sparse" in str(e).lower():
            st.error("🔍 **Diagnosis**: scipy.sparse error detected")
            st.error("This indicates object dtype was passed to model preprocessing.")
        
        st.stop()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em;'>
💡 <strong>Tip:</strong> Use realistic values from your recent sales history for accurate predictions.
</div>
""", unsafe_allow_html=True)