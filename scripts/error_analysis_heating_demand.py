import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import os

# Configuration
MODEL_PATH = "models/heating_demand_predictor_xgb_tuned.joblib"
DATA_PATH = "data/processed/final_dataset_with_forecasts.csv"
FIGURES_OUTPUT_DIR = "reports/figures"

def calculate_heating_demand(row, base_temp=18.0):
    """Calculate heating demand in degree-hours."""
    apparent_temp = row['apparent_temperature']
    if pd.isna(apparent_temp):
        return np.nan
    return max(0, base_temp - apparent_temp)

def load_and_prepare_data():
    """Load data and recreate the exact same preprocessing as training."""
    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    df = df.set_index('time').sort_index()
    
    # Create heating demand target
    df['heating_demand_current'] = df.apply(calculate_heating_demand, axis=1)
    df['target_heating_demand'] = df['heating_demand_current'].shift(-1)
    df['target_time'] = df.index.to_series().shift(-1)
    
    return df

def analyze_heating_demand_errors():
    """Comprehensive error analysis for heating demand prediction model."""
    print("🔥 HEATING DEMAND MODEL ERROR ANALYSIS")
    print("=" * 60)
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_columns = model_data['features']
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    # Apply same feature engineering as training script
    # (This is a simplified version - in production, should modularize this)
    
    # Time features
    if not pd.api.types.is_datetime64_any_dtype(df['target_time']):
        df['target_time'] = pd.to_datetime(df['target_time'])
    
    df['hour'] = df['target_time'].dt.hour
    df['dayofweek'] = df['target_time'].dt.dayofweek
    df['month'] = df['target_time'].dt.month
    df['dayofyear'] = df['target_time'].dt.dayofyear
    df['quarter'] = df['target_time'].dt.quarter
    
    # Enhanced time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    df['dayofyear_sin'] = np.sin(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    df['dayofyear_cos'] = np.cos(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Season
    def get_season(month_col):
        conditions = [
            month_col.isin([12, 1, 2]), # Winter
            month_col.isin([3, 4, 5]),   # Spring
            month_col.isin([6, 7, 8]),   # Summer
            month_col.isin([9, 10, 11]) # Autumn
        ]
        choices = [0, 1, 2, 3]
        return np.select(conditions, choices, default=np.nan)
    df['season'] = get_season(df['month'])
    
    # Create lag features (simplified)
    weather_cols = ['apparent_temperature', 'temperature', 'humidity', 'windspeed', 'heating_demand_current']
    for col in weather_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Forecast features and interactions (add only if they exist)
    forecast_cols = ['temperature_forecast', 'humidity_forecast', 'windspeed_forecast', 
                    'sw_radiation_forecast', 'cloud_cover_forecast', 'precipitation_forecast',
                    'apparent_temperature_forecast']
    
    for col in forecast_cols:
        if col not in df.columns:
            continue
    
    # Create derived heating demand from forecast
    if 'apparent_temperature_forecast' in df.columns:
        df['heating_demand_forecast'] = df['apparent_temperature_forecast'].apply(
            lambda x: max(0, 18.0 - x) if pd.notna(x) else np.nan
        )
    
    # Basic interactions
    if 'temperature_forecast' in df.columns and 'windspeed_forecast' in df.columns:
        df['wind_chill_factor'] = df['temperature_forecast'] - (df['windspeed_forecast'] * 0.5)
    
    if 'temperature_forecast' in df.columns and 'humidity_forecast' in df.columns:
        df['temp_humidity_interaction'] = df['temperature_forecast'] * (df['humidity_forecast'] / 100.0)
    
    if 'sw_radiation_forecast' in df.columns and 'cloud_cover_forecast' in df.columns:
        cloud_fraction = (df['cloud_cover_forecast'] / 100.0).clip(0, 1)
        df['solar_heating_effect'] = df['sw_radiation_forecast'] * (1 - cloud_fraction)
    
    # Filter data and features
    df_clean = df.dropna(subset=feature_columns + ['target_heating_demand'])
    
    X = df_clean[feature_columns]
    y_true = df_clean['target_heating_demand']
    
    # Recreate the same train/test split (time-ordered, no shuffle)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, shuffle=False)
    
    print(f"Test set size: {len(y_test)} samples")
    print(f"Test period: {y_test.index[0]} to {y_test.index[-1]}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
    
    # Calculate overall metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  MAE:  {mae:.4f} degree-hours")
    print(f"  RMSE: {rmse:.4f} degree-hours")
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Analysis by heating demand levels
    print(f"\n🌡️  PERFORMANCE BY HEATING DEMAND LEVEL:")
    demand_ranges = [
        ("No heating (0)", y_test == 0),
        ("Low (0-5)", (y_test > 0) & (y_test <= 5)),
        ("Medium (5-10)", (y_test > 5) & (y_test <= 10)),
        ("High (10-15)", (y_test > 10) & (y_test <= 15)),
        ("Very High (>15)", y_test > 15)
    ]
    
    for label, mask in demand_ranges:
        if mask.sum() > 0:
            mae_subset = mean_absolute_error(y_test[mask], y_pred[mask])
            count = mask.sum()
            pct = mask.mean() * 100
            print(f"  {label:15}: MAE={mae_subset:.3f}, Count={count:4d} ({pct:4.1f}%)")
    
    # Temporal analysis
    test_data = df_clean.loc[y_test.index].copy()
    test_data['y_true'] = y_test
    test_data['y_pred'] = y_pred
    test_data['residuals'] = residuals
    test_data['abs_residuals'] = np.abs(residuals)
    
    # Hour of day analysis
    print(f"\n🕐 PERFORMANCE BY HOUR OF DAY:")
    hourly_mae = test_data.groupby('hour')['abs_residuals'].mean()
    for hour in range(0, 24, 3):  # Every 3 hours
        if hour in hourly_mae.index:
            print(f"  Hour {hour:2d}: MAE={hourly_mae[hour]:.3f}")
    
    # Monthly analysis
    print(f"\n📅 PERFORMANCE BY MONTH:")
    monthly_mae = test_data.groupby('month')['abs_residuals'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in range(1, 13):
        if month in monthly_mae.index:
            print(f"  {month_names[month-1]:3}: MAE={monthly_mae[month]:.3f}")
    
    # Seasonal analysis
    print(f"\n🍂 PERFORMANCE BY SEASON:")
    seasonal_mae = test_data.groupby('season')['abs_residuals'].mean()
    season_names = ['Winter', 'Spring', 'Summer', 'Autumn']
    for season in range(4):
        if season in seasonal_mae.index:
            print(f"  {season_names[season]:6}: MAE={seasonal_mae[season]:.3f}")
    
    # Generate plots
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
    
    # 1. Time series plot
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    time_subset = slice(0, min(500, len(y_test)))  # First 500 points for clarity
    plt.plot(y_test.iloc[time_subset], label='Actual Heating Demand', alpha=0.8, color='blue')
    plt.plot(y_pred[time_subset], label='Predicted Heating Demand', alpha=0.8, color='red', linestyle='--')
    plt.title('Heating Demand: Actual vs Predicted (First 500 Test Points)')
    plt.ylabel('Heating Demand (degree-hours)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(residuals.iloc[time_subset], alpha=0.8, color='green')
    plt.title('Prediction Residuals')
    plt.ylabel('Residual (degree-hours)')
    plt.xlabel('Test Sample')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_time_series_analysis.png'), dpi=300)
    plt.close()
    
    # 2. Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=1)
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Heating Demand (degree-hours)')
    plt.ylabel('Predicted Heating Demand (degree-hours)')
    plt.title('Heating Demand: Actual vs Predicted Scatter Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_scatter_plot.png'), dpi=300)
    plt.close()
    
    # 3. MAE by hour
    plt.figure(figsize=(12, 6))
    hourly_mae.plot(kind='bar')
    plt.title('Mean Absolute Error by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('MAE (degree-hours)')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_mae_by_hour.png'), dpi=300)
    plt.close()
    
    # 4. MAE by month
    plt.figure(figsize=(12, 6))
    monthly_mae.plot(kind='bar')
    plt.title('Mean Absolute Error by Month')
    plt.xlabel('Month')
    plt.ylabel('MAE (degree-hours)')
    plt.xticks([i-1 for i in range(1, 13)], month_names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_mae_by_month.png'), dpi=300)
    plt.close()
    
    # 5. Residual distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Residual (degree-hours)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals[y_test > 0], bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Residual (degree-hours)')
    plt.ylabel('Frequency')
    plt.title('Residuals (Non-zero Heating Demand Only)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_residual_distribution.png'), dpi=300)
    plt.close()
    
    print(f"\n✅ Analysis complete! Plots saved to {FIGURES_OUTPUT_DIR}/")
    
    # Summary insights
    print(f"\n🎯 KEY INSIGHTS:")
    zero_demand_mask = (y_test == 0)
    nonzero_demand_mask = (y_test > 0)
    
    if zero_demand_mask.sum() > 0:
        mae_zero = mean_absolute_error(y_test[zero_demand_mask], y_pred[zero_demand_mask])
        print(f"  • Zero heating demand periods: {zero_demand_mask.sum()} samples, MAE={mae_zero:.3f}")
    
    if nonzero_demand_mask.sum() > 0:
        mae_nonzero = mean_absolute_error(y_test[nonzero_demand_mask], y_pred[nonzero_demand_mask])
        print(f"  • Non-zero heating demand periods: {nonzero_demand_mask.sum()} samples, MAE={mae_nonzero:.3f}")
    
    # Check prediction range
    print(f"  • Actual heating demand range: {y_test.min():.2f} to {y_test.max():.2f}")
    print(f"  • Predicted heating demand range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    
    # Check if model predicts zero correctly
    pred_zero_mask = (y_pred < 0.1)  # Nearly zero predictions
    actual_zero_mask = (y_test < 0.1)  # Nearly zero actual
    correct_zero_predictions = (pred_zero_mask & actual_zero_mask).sum()
    total_zero_actual = actual_zero_mask.sum()
    
    if total_zero_actual > 0:
        zero_accuracy = correct_zero_predictions / total_zero_actual * 100
        print(f"  • Zero-demand prediction accuracy: {zero_accuracy:.1f}% ({correct_zero_predictions}/{total_zero_actual})")

if __name__ == '__main__':
    analyze_heating_demand_errors() 