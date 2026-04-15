"""
PRODUCTION MODEL RETRAINING SCRIPT

Retrains XGBoost using ONLY engineered features.
Features match EXACTLY what app.py sends.
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import os

# Set path
path = r'C:\Users\Swastik Pandey\OneDrive\Documents\MLminorStreamlit'
os.chdir(path)

# Load data
print('Loading data...')
df = pd.read_csv('Sample - Superstore.csv', encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')
df = df.ffill()

# Create all engineered features
print('Creating engineered features...')
df['day'] = df['Order Date'].dt.day
df['month'] = df['Order Date'].dt.month
df['weekday'] = df['Order Date'].dt.weekday
df['lag_1'] = df['Sales'].shift(1)
df['lag_7'] = df['Sales'].shift(7)
df['lag_14'] = df['Sales'].shift(14)
df['lag_30'] = df['Sales'].shift(30)
df['rolling_mean_7'] = df['Sales'].rolling(7).mean()
df['rolling_mean_14'] = df['Sales'].rolling(14).mean()
df['rolling_std_7'] = df['Sales'].rolling(7).std()
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['weekofyear'] = df['Order Date'].dt.isocalendar().week.astype(int)
df['trend'] = range(len(df))
df = df.dropna()

# Define engineered features (MUST MATCH app.py)
ENGINEERED_FEATURES = [
    'day', 'month', 'weekday', 'is_weekend', 'weekofyear',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7', 'trend'
]

print('=' * 80)
print('PRODUCTION MODEL RETRAINING')
print('=' * 80)
print(f'\nUsing {len(ENGINEERED_FEATURES)} engineered features')
for i, feat in enumerate(ENGINEERED_FEATURES, 1):
    print(f'  {i:2d}. {feat}')

# Select only engineered features
print('\nBuilding training dataset...')
X = df[ENGINEERED_FEATURES].copy()
y = df['Sales'].copy()

print(f'✓ X shape: {X.shape}')
print(f'✓ y shape: {y.shape}')

# Check for object dtypes
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    raise ValueError(f'Object dtype columns found: {object_cols}')
print('✓ All features are numeric (no object dtype)')

# Train-test split
train_size = int(len(X) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print(f'\nTrain-test split:')
print(f'   Training: {len(X_train)} rows')
print(f'   Testing: {len(X_test)} rows')

# Train XGBoost
print(f'\nTraining XGBoost model...')
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
print('✓ Model trained successfully')

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f'\nModel Performance:')
print(f'   Mean Absolute Error: ${mae:.2f}')
print(f'   Test samples: {len(y_test)}')

# Save model
print(f'\nSaving production model...')
joblib.dump({
    'model_pipeline': model,
    'expected_feature_columns': ENGINEERED_FEATURES,
    'model_type': 'XGBoost',
    'mae': mae,
    'features': ENGINEERED_FEATURES
}, 'model.pkl')

print('✅ Model saved as model.pkl')

# Verify
loaded = joblib.load('model.pkl')
print(f'\nModel Verification:')
print(f'   Type: {loaded["model_type"]}')
print(f'   Features: {len(loaded["expected_feature_columns"])} columns')
print(f'   MAE: ${loaded["mae"]:.2f}')

# Test prediction
test_sample = X_test.iloc[:1]
test_pred = model.predict(test_sample)
print(f'   Test prediction: ${test_pred[0]:.2f}')

print('\n' + '=' * 80)
print('🎯 PRODUCTION MODEL READY FOR STREAMLIT APP')
print('=' * 80)
print('\nModel.pkl updated with:')
print(f'  • XGBoost trained on {len(X_train)} samples')
print(f'  • {len(ENGINEERED_FEATURES)} engineered features')
print(f'  • MAE: ${mae:.2f}')
print(f'  • Ready for app.py')
