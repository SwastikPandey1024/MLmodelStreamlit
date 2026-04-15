# Strict Input Validation Guide

## Overview
This document explains the production-grade validation system implemented in `app.py`.

---

## Validation Architecture

### Layer 1: Individual Input Validation
**Function**: `validate_numeric_input(value, name, min_val, max_val)`

- ✅ Converts value to float
- ✅ Checks for NaN
- ✅ Checks for infinity
- ✅ Enforces min/max constraints
- ❌ Raises ValueError immediately on failure (no silent coercion)

**Example**:
```python
validate_numeric_input(100, "lag_1", min_val=0)  # ✓ Returns 100.0
validate_numeric_input("abc", "lag_1")            # ✗ Raises ValueError
validate_numeric_input(-50, "lag_1", min_val=0)  # ✗ Below minimum
```

### Layer 2: Date Feature Extraction
**Function**: `extract_date_features(date_input)`

- ✅ Parses date safely
- ✅ Extracts: day, month, weekday, is_weekend, weekofyear
- ✅ Validates ranges:
  - day: 1-31
  - month: 1-12
  - weekday: 0-6
  - weekofyear: 1-53
- ❌ Raises ValueError with clear message on failure

**Example**:
```python
extract_date_features("2026-04-15")   # ✓ Returns {'day': 15, ...}
extract_date_features("invalid")      # ✗ Raises ValueError
```

### Layer 3: DataFrame Creation (Strict Pipeline)
**Function**: `create_input_dataframe_strict(..., expected_cols)`

8-step validation pipeline:
1. Validate each input using `validate_numeric_input()`
2. Create DataFrame from validated Python floats
3. Force astype(float64) - STRICT, no coercion
4. Check for object dtypes
5. Check for NaN values
6. Check for infinite values
7. Verify column names match training features
8. Final shape sanity check

**Key Difference from Old Approach**:
```python
# ❌ OLD (Silent Corruption)
pd.to_numeric(errors='coerce')  # Converts invalid to NaN
fillna(0.0)                      # Converts NaN to 0 silently
# Result: Invalid input becomes 0 (wrong prediction)

# ✅ NEW (Strict Validation)
validate_numeric_input()         # Raises error immediately
# Result: User knows input was invalid, prediction not attempted
```

### Layer 4: Final Safety Checks Before Prediction
In the prediction block:
- Check for object dtypes (defense-in-depth)
- Check for NaN values
- Check for infinite values
- Verify shape matches expected

---

## Validation Results

### When Validation PASSES ✅
1. User sees: "✅ Input Data Validation (Valid)"
2. Debug panel shows:
   - All features with their values
   - All dtypes are float64
   - Success message: "✅ All columns are float64 - Ready for prediction"
3. Predict button is enabled and works

### When Validation FAILS ❌
1. User sees: "❌ **Input Validation Failed**"
2. Error message displays: Exactly what went wrong
3. Warning with guidance:
   - "All fields must be valid numbers"
   - "Lag and rolling values must be non-negative"
   - "Date must be valid"
4. Predict button still works, but will fail with error

---

## Examples

### Example 1: Valid Input
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
Output: "Predicted Sales: $105.32"
```

### Example 2: Invalid Lag Value
```
Date: 2026-04-15
lag_1: -50.00  ← NEGATIVE (invalid)
lag_7: 95.50
...

Result: ❌ FAIL
Error: "lag_1: Value -50.0 is less than minimum 0"
Output: No prediction attempted
```

### Example 3: Invalid Date
```
Date: 2026-13-32  ← Invalid date
lag_1: 100.00
...

Result: ❌ FAIL
Error: "day: Day must be 1-31, got 32"
Output: No prediction attempted
```

### Example 4: NaN/Inf Values
```
Date: 2026-04-15
lag_1: inf  ← INFINITE
lag_7: 95.50
...

Result: ❌ FAIL
Error: "lag_1: Infinite value not allowed"
Output: No prediction attempted
```

---

## Error Messages (User Friendly)

| Scenario | User Sees |
|----------|-----------|
| Non-numeric input | `"[field]: Must be a valid number, got [value]"` |
| Negative value | `"[field]: Value [x] is less than minimum [min]"` |
| Too large value | `"[field]: Value [x] exceeds maximum [max]"` |
| NaN value | `"[field]: NaN value not allowed"` |
| Infinite value | `"[field]: Infinite value not allowed"` |
| Invalid date | `"day: Day must be 1-31, got [x]"` |
| Column mismatch | `"Feature mismatch: Expected [...], Got [...]"` |

---

## Production Guarantees

✅ **No Silent Data Corruption**
- Invalid inputs are NOT converted to 0
- Invalid inputs are NOT filled with defaults
- Invalid inputs STOP execution with clear error

✅ **No Incorrect Predictions**
- Prediction only happens with clean, validated data
- All values are verified before model call
- Output is checked for validity

✅ **Strict Type Safety**
- Only float64 reaches the model
- No object dtypes ever passed to preprocessing
- No scipy.sparse errors from bad types

✅ **Clear User Feedback**
- Every error has a clear message
- Users know exactly what's wrong
- No silent failures or mysterious errors

✅ **Comprehensive Logging**
- All errors logged for debugging
- Timestamps and error context preserved
- Can trace validation failures

---

## Comparison: Old vs New

| Aspect | Old | New |
|--------|-----|-----|
| Invalid input handling | Silent → 0 | Raises error |
| Type conversion | coerce + fillna | Strict validation |
| Error messages | Generic | Specific (field + constraint) |
| Data corruption | Possible | Impossible |
| Performance | Slightly faster | Slightly slower (worth it) |
| Production ready | No | Yes |

---

## Debugging Tips

### Check Validation Panel
1. Click expander: "✅ Input Data Validation (Valid)"
2. See all feature values
3. Verify dtypes are float64
4. Look for any unexpected values

### Check Error Message
1. Read the specific error
2. Find which field failed
3. Check constraint (min/max/type)
4. Correct input and try again

### Check Logs
```python
# Errors are logged to logger
# Can be checked with: streamlit run app.py
```

### Enable Streamlit Debug
```bash
streamlit run app.py --logger.level=debug
```

---

## Migration from Old Code

If you had existing code using the old approach:

```python
# ❌ OLD: create_input_dataframe()
input_data = create_input_dataframe(date_features, ...)

# ✅ NEW: create_input_dataframe_strict()
input_data = create_input_dataframe_strict(date_features, ..., expected_columns)
```

The new function requires `expected_columns` parameter (list of feature names).

---

## Future Improvements

1. Add data type hints
2. Add unit tests for validation functions
3. Add validation statistics dashboard
4. Add CSV batch validation
5. Add custom constraint rules
