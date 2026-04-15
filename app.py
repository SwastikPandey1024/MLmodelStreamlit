"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   📊 SalesPulse AI                                           ║
║        Intelligent Sales Analytics Dashboard & Forecasting Engine            ║
║                                                                              ║
║  A production-ready ML analytics platform for:                              ║
║  • Single-day sales predictions using XGBoost                               ║
║  • 7-day sales forecasting with iterative predictions                       ║
║  • Batch predictions via CSV upload                                         ║
║  • Trend analysis and business insights                                     ║
║  • Feature importance and model explainability                              ║
║                                                                              ║
║  Validation: Strict 3-layer validation prevents all silent data corruption  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import logging
import io

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE CONSTANTS (MUST MATCH model.pkl EXACTLY)
# ============================================================================
ENGINEERED_FEATURES = [
    "day",
    "month",
    "weekday",
    "is_weekend",
    "weekofyear",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "trend"
]

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SalesPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Intelligent Sales Analytics Dashboard by SalesPulse"}
)

# Custom theme
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================
@st.cache_resource
def load_model():
    """Load trained model and expected feature columns."""
    try:
        with open("model.pkl", "rb") as f:
            loaded_content = joblib.load(f)
        
        model = loaded_content['model_pipeline']
        features = loaded_content['expected_feature_columns']
        
        return model, features
    
    except FileNotFoundError:
        st.error("❌ model.pkl not found. Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.stop()

model_pipeline, expected_columns = load_model()

# Validate features match
if expected_columns != ENGINEERED_FEATURES:
    st.error(f"❌ CRITICAL: Feature mismatch!\nExpected: {ENGINEERED_FEATURES}\nLoaded: {expected_columns}")
    st.stop()

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def extract_date_features(date_input):
    """Extract date features from date object."""
    try:
        date_obj = pd.Timestamp(date_input)
        
        day = int(date_obj.day)
        month = int(date_obj.month)
        weekday = int(date_obj.weekday())
        is_weekend = 1 if weekday >= 5 else 0
        weekofyear = int(date_obj.isocalendar().week)
        
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
    """Validate single numeric input."""
    try:
        numeric_val = float(value)
        
        if np.isnan(numeric_val):
            raise ValueError(f"{name}: NaN value not allowed")
        if np.isinf(numeric_val):
            raise ValueError(f"{name}: Infinite value not allowed")
        
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
    """Create and validate input DataFrame."""
    try:
        if date_features is None:
            raise ValueError("Date features are None")
        
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
        
        df = pd.DataFrame(validated_data, index=[0])
        df = df.astype(np.float64)
        
        object_cols = [col for col in df.columns if df[col].dtype == 'object']
        if object_cols:
            raise ValueError(f"Object dtype found in columns: {object_cols}")
        
        nan_mask = df.isnull()
        if nan_mask.any().any():
            nan_cols = df.columns[nan_mask.any()].tolist()
            raise ValueError(f"NaN values found in: {nan_cols}")
        
        inf_mask = ~np.isfinite(df.values)
        if inf_mask.any():
            inf_cols = df.columns[inf_mask.any()].tolist()
            raise ValueError(f"Infinite values found in: {inf_cols}")
        
        actual_cols = list(df.columns)
        if actual_cols != expected_cols:
            raise ValueError(f"Feature mismatch:\nExpected: {expected_cols}\nGot: {actual_cols}")
        
        df = df[expected_cols]
        
        if df.shape != (1, len(expected_cols)):
            raise ValueError(f"Shape mismatch: expected (1, {len(expected_cols)}), got {df.shape}")
        
        return df
    
    except ValueError as ve:
        logger.error(f"Input validation failed: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error in create_input_dataframe_strict: {str(e)}")
        raise ValueError(f"Input processing error: {str(e)}")


def make_prediction(input_data, model):
    """Make prediction with validation."""
    try:
        object_cols = [col for col in input_data.columns if input_data[col].dtype == 'object']
        if object_cols:
            raise ValueError(f"Unexpected object dtype in: {object_cols}")
        
        if input_data.isnull().any().any():
            raise ValueError("Unexpected NaN values in input")
        
        if not np.isfinite(input_data.values).all():
            raise ValueError("Unexpected infinite values in input")
        
        prediction = model.predict(input_data)
        predicted_sales = float(prediction[0])
        
        if np.isnan(predicted_sales):
            raise ValueError("Model returned NaN prediction")
        if np.isinf(predicted_sales):
            raise ValueError("Model returned infinite prediction")
        
        return predicted_sales
    
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")


# ============================================================================
# INSIGHT & ANALYSIS FUNCTIONS
# ============================================================================

def generate_insights(predicted_sales, lag_1, rolling_mean_7, rolling_mean_14):
    """Generate business insights from prediction."""
    insights = []
    
    # Trend analysis
    if predicted_sales > lag_1 * 1.05:
        trend_icon = "📈"
        trend_text = "Strong growth expected"
        trend_color = "success"
        recommendation = "Consider scaling inventory and staffing."
        insights.append({
            "type": "growth",
            "icon": trend_icon,
            "text": trend_text,
            "color": trend_color,
            "recommendation": recommendation
        })
    elif predicted_sales < lag_1 * 0.95:
        trend_icon = "📉"
        trend_text = "Decline expected"
        trend_color = "warning"
        recommendation = "Consider promotional activities or cost adjustments."
        insights.append({
            "type": "decline",
            "icon": trend_icon,
            "text": trend_text,
            "color": trend_color,
            "recommendation": recommendation
        })
    else:
        trend_icon = "➡️"
        trend_text = "Sales stable"
        trend_color = "info"
        recommendation = "Maintain current strategy."
        insights.append({
            "type": "stable",
            "icon": trend_icon,
            "text": trend_text,
            "color": trend_color,
            "recommendation": recommendation
        })
    
    # Comparison with rolling averages
    if predicted_sales > rolling_mean_7:
        comp_icon = "⬆️"
        comp_text = f"Above 7-day average (${rolling_mean_7:.2f})"
        insights.append({
            "type": "comparison",
            "icon": comp_icon,
            "text": comp_text,
            "value": f"+${predicted_sales - rolling_mean_7:.2f}"
        })
    elif predicted_sales < rolling_mean_7:
        comp_icon = "⬇️"
        comp_text = f"Below 7-day average (${rolling_mean_7:.2f})"
        insights.append({
            "type": "comparison",
            "icon": comp_icon,
            "text": comp_text,
            "value": f"-${rolling_mean_7 - predicted_sales:.2f}"
        })
    
    # Volatility check
    if rolling_mean_14 > 0:
        volatility_ratio = rolling_mean_7 / rolling_mean_14
        if volatility_ratio > 1.1:
            insights.append({
                "type": "volatility",
                "icon": "⚡",
                "text": "High volatility detected",
                "detail": "Sales trending above biweekly average"
            })
        elif volatility_ratio < 0.9:
            insights.append({
                "type": "volatility",
                "icon": "🔻",
                "text": "Low volatility detected",
                "detail": "Sales trending below biweekly average"
            })
    
    return insights


def get_feature_importance():
    """Extract feature importance from XGBoost model."""
    try:
        trained_model = model_pipeline.named_steps['regressor']
        importance = trained_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': ENGINEERED_FEATURES,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return None


def forecast_7_days(initial_data, lag_1_val):
    """Forecast next 7 days iteratively."""
    try:
        forecast_dates = []
        forecast_values = []
        current_date = datetime.now()
        current_data = initial_data.copy()
        
        # Keep track of recent predictions for lag features
        recent_predictions = [lag_1_val]
        
        for day_offset in range(1, 8):
            pred_date = current_date + timedelta(days=day_offset)
            forecast_dates.append(pred_date)
            
            # Update date features
            date_features = extract_date_features(pred_date)
            
            # Update lag features (use recent predictions)
            lag_1_new = recent_predictions[-1] if len(recent_predictions) > 0 else lag_1_val
            lag_7_new = recent_predictions[-7] if len(recent_predictions) > 7 else lag_1_val
            lag_14_new = recent_predictions[-14] if len(recent_predictions) > 14 else lag_1_val
            lag_30_new = recent_predictions[-30] if len(recent_predictions) > 30 else lag_1_val
            
            # Calculate rolling metrics (approximate)
            recent_for_mean = recent_predictions[-7:] if len(recent_predictions) >= 7 else recent_predictions
            rolling_mean_7_new = np.mean(recent_for_mean) if recent_for_mean else lag_1_val
            rolling_mean_14_new = np.mean(recent_predictions[-14:]) if len(recent_predictions) >= 14 else rolling_mean_7_new
            rolling_std_7_new = np.std(recent_for_mean) if len(recent_for_mean) > 1 else 1.0
            
            # Trend counter
            trend_new = current_data['trend'].iloc[0] + day_offset
            
            # Create prediction input
            pred_input = create_input_dataframe_strict(
                date_features,
                lag_1_new, lag_7_new, lag_14_new, lag_30_new,
                rolling_mean_7_new, rolling_mean_14_new, rolling_std_7_new,
                trend_new,
                expected_columns
            )
            
            # Make prediction
            prediction = make_prediction(pred_input, model_pipeline)
            forecast_values.append(prediction)
            recent_predictions.append(prediction)
        
        return pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted Sales': forecast_values
        })
    
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        return None