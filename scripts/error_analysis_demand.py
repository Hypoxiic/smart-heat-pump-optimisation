"""
Error analysis for the tuned 1-hour-ahead apparent-temperature model.

▪ Re-creates the train/test split exactly as in `train_demand_predictor.py`
▪ Generates fresh test-set predictions
▪ Computes residuals + key metrics
▪ Builds a suite of one-figure-per-file matplotlib plots
▪ Writes everything under `reports/figures/`
"""

# ---------------------------------------------------------------------
# 1. Imports & constants
# ---------------------------------------------------------------------
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------  repo-specific paths --------------------------
DATA_PATH = "data/processed/final_dataset_with_forecasts.csv"
MODEL_PATH = "models/demand_predictor_xgb_tuned_with_forecasts.joblib"

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# must match training script
TARGET_COLUMN = "apparent_temperature"
SHIFT_PERIODS = -1          # predict 1 hour ahead
TEST_SIZE = 0.20            # 80 / 20 split, no shuffling
RANDOM_STATE = 42           # (only used by train_test_split when shuffle=False → reproducible)

# ---------------------------------------------------------------------
# Feature Engineering Functions (copied from training script)
# ---------------------------------------------------------------------
def create_time_features(df, target_datetime_col):
    """Creates time-based features from a datetime column for the target prediction time."""
    # Ensure the target_datetime_col is present and is datetime type
    if target_datetime_col not in df.columns:
        raise ValueError(f"Column {target_datetime_col} not found in DataFrame.")
    if not pd.api.types.is_datetime64_any_dtype(df[target_datetime_col]):
        df[target_datetime_col] = pd.to_datetime(df[target_datetime_col])

    df_feat = df.copy()
    df_feat['hour'] = df_feat[target_datetime_col].dt.hour
    df_feat['dayofweek'] = df_feat[target_datetime_col].dt.dayofweek
    df_feat['month'] = df_feat[target_datetime_col].dt.month
    df_feat['dayofyear'] = df_feat[target_datetime_col].dt.dayofyear
    df_feat['quarter'] = df_feat[target_datetime_col].dt.quarter
    return df_feat

def create_lag_features(df, column_name, lags=[1, 2, 3, 6, 12, 24]):
    """Creates lag features for a specified column."""
    df_feat = df.copy()
    for lag in lags:
        df_feat[f'{column_name}_lag{lag}'] = df_feat[column_name].shift(lag)
    return df_feat

# ---------------------------------------------------------------------
# 2. Load model payload  (contains model + feature list)
# ---------------------------------------------------------------------
payload = joblib.load(MODEL_PATH)
model       = payload["model"]
feature_cols = payload["features"]      # list of feature names in correct order

# ---------------------------------------------------------------------
# 3. Load data & recreate target, features, split
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["time"]).set_index("time").sort_index()

# target for t+1 h
df["target"] = df[TARGET_COLUMN].shift(SHIFT_PERIODS)

# Create a datetime column for the target prediction time to generate relevant time features
df['target_time'] = df.index.to_series().shift(SHIFT_PERIODS)
df['target_time'] = pd.to_datetime(df['target_time']) # Ensure it's datetime

# Apply Feature Engineering (as in training script)
df = create_time_features(df, target_datetime_col='target_time')

# Enhanced time features
df['hour_target_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_target_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['dayofweek_target_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
df['dayofweek_target_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
df['month_target_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
df['month_target_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
df['dayofyear_target_sin'] = np.sin(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
df['dayofyear_target_cos'] = np.cos(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
df['is_weekend_target'] = (df['dayofweek'] >= 5).astype(int)
def get_season(month_col):
    conditions = [
        month_col.isin([12, 1, 2]), # Winter
        month_col.isin([3, 4, 5]),   # Spring
        month_col.isin([6, 7, 8]),   # Summer
        month_col.isin([9, 10, 11]) # Autumn
    ]
    choices = [0, 1, 2, 3] # Winter, Spring, Summer, Autumn
    return np.select(conditions, choices, default=np.nan)
df['season_target'] = get_season(df['month'])

# Lag features
weather_cols_for_lags = ['apparent_temperature', 'temperature', 'humidity', 'windspeed',
                         'sw_radiation_forecast', 'cloud_cover_forecast', 'precipitation_forecast']
for col in weather_cols_for_lags:
    if col in df.columns:
        df = create_lag_features(df, col, lags=[1, 2, 3, 6, 12, 24])
    else:
        print(f"Warning: Column {col} for lag feature creation not found in error analysis script.")


# Interaction features
if 'temperature_forecast' in df.columns and 'humidity_forecast' in df.columns:
    df['temp_X_humidity_forecast'] = df['temperature_forecast'] * df['humidity_forecast']
if 'temperature_forecast' in df.columns and 'windspeed_forecast' in df.columns:
    df['temp_X_windspeed_forecast'] = df['temperature_forecast'] * df['windspeed_forecast']
if 'sw_radiation_forecast' in df.columns and 'cloud_cover_forecast' in df.columns:
    cloud_cover_fraction = (df['cloud_cover_forecast'] / 100.0).clip(0,1)
    df['sw_rad_eff_forecast'] = df['sw_radiation_forecast'] * (1 - cloud_cover_fraction)

# Polynomial features
if 'temperature_forecast' in df.columns:
    df['temperature_forecast_sq'] = df['temperature_forecast']**2
if 'humidity_forecast' in df.columns:
    df['humidity_forecast_sq'] = df['humidity_forecast']**2
if 'sw_radiation_forecast' in df.columns:
    df['sw_radiation_forecast_sq'] = df['sw_radiation_forecast']**2

# keep only the columns we need
needed = feature_cols + ["target"]
df_model = df[needed].dropna(subset=["target"]) # Only drop rows if target is NaN
df_model = df_model.dropna(subset=feature_cols, how='any') # Drop rows if any feature is NaN

X = df_model[feature_cols]
y = df_model["target"]

# identical split (time-ordered, no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

# ---------------------------------------------------------------------
# 4. Predict & basic metrics
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTest-set metrics  (N = {len(y_test)})")
print(f"  MAE : {mae:7.3f} °C")
print(f"  RMSE: {rmse:7.3f} °C")

# residuals Series aligned with index
residuals = y_test - y_pred
residuals.name = "residual"

# ---------------------------------------------------------------------
# 5. Plot helpers
# ---------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved → {path}")

# ---------------------------------------------------------------------
# 6. Residual-over-time
# ---------------------------------------------------------------------
plt.figure(figsize=(15, 4))
plt.plot(residuals.index, residuals, lw=0.8)
plt.axhline(0, ls="--")
plt.xlabel("Time")
plt.ylabel("Residual (°C)")
plt.title("Residuals vs. Time")
savefig("demand_residuals_over_time.png")

# ---------------------------------------------------------------------
# 7. Histogram of residuals
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=60)
plt.xlabel("Residual (°C)")
plt.ylabel("Frequency")
plt.title("Residual Distribution (test set)")
savefig("demand_residual_histogram.png")

# ---------------------------------------------------------------------
# 8. Actual vs. predicted scatter
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, s=10, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, ls="--")
plt.xlabel("Actual °C")
plt.ylabel("Predicted °C")
plt.title("Actual vs. Predicted")
savefig("demand_actual_vs_pred_scatter.png")

# ---------------------------------------------------------------------
# 9. MAE by hour-of-day / month / season / weekday
# ---------------------------------------------------------------------
# rebuild time features for the *target time* (t+1 h)
target_time = y_test.index + pd.Timedelta(hours=1)
hour   = target_time.hour
month  = target_time.month
dow    = target_time.dayofweek
season = pd.cut(month, bins=[0,2,5,8,11,12],
                labels=["Winter","Spring","Summer","Autumn","Winter"], right=True, ordered=False)

groups = {
    "hour_of_day" : hour,
    "month"       : month,
    "weekday"     : dow,
    "season"      : season
}

for label, grp in groups.items():
    mae_grp = residuals.abs().groupby(grp).mean()
    plt.figure(figsize=(7,4))
    mae_grp.plot.bar()
    plt.ylabel("MAE (°C)")
    plt.title(f"MAE by {label.replace('_',' ').title()}")
    savefig(f"demand_MAE_by_{label}.png")

# ---------------------------------------------------------------------
# 10. Save numeric error summaries
# ---------------------------------------------------------------------
summary = pd.DataFrame({
    "MAE_by_hour"   : residuals.abs().groupby(hour).mean(),
    "MAE_by_month"  : residuals.abs().groupby(month).mean(),
    "MAE_by_weekday": residuals.abs().groupby(dow).mean(),
    "MAE_by_season" : residuals.abs().groupby(season).mean()
})

summary_path = os.path.join(FIG_DIR, "demand_error_summary.csv")
summary.to_csv(summary_path)
print(f"\nNumeric summaries → {summary_path}")

print("\nDone – check the figures & CSV in reports/figures/")
