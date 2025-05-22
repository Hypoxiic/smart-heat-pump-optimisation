import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import os
import optuna
from tqdm.auto import tqdm

# Define constants
MODEL_OUTPUT_DIR = "models"
FIGURES_OUTPUT_DIR = "reports/figures"
DATA_PATH = "data/processed/final_dataset_with_forecasts.csv"
MODEL_NAME_TUNED = "heating_demand_predictor_xgb_tuned.joblib"
N_OPTUNA_TRIALS = 100
N_CV_SPLITS = 5

def calculate_heating_demand(row, base_temp=18.0):
    """
    Calculate heating demand in degree-hours based on temperature difference.
    
    Heating demand = max(0, base_temperature - apparent_temperature)
    
    This gives:
    - 0 when apparent temp >= 18°C (no heating needed)
    - Positive values when apparent temp < 18°C (heating needed)
    - Higher values for colder temperatures
    """
    apparent_temp = row['apparent_temperature']
    if pd.isna(apparent_temp):
        return np.nan
    
    heating_demand = max(0, base_temp - apparent_temp)
    return heating_demand

def create_time_features(df, target_datetime_col):
    """Creates time-based features from a datetime column."""
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

def train_heating_demand_model():
    """Trains an XGBoost model to predict heating demand (degree-hours)."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    df = df.set_index('time').sort_index()

    print("Creating heating demand target variable...")
    # Current heating demand based on current apparent temperature
    df['heating_demand_current'] = df.apply(calculate_heating_demand, axis=1)
    
    # Target: heating demand 1 hour ahead
    df['target_heating_demand'] = df['heating_demand_current'].shift(-1)
    df['target_time'] = df.index.to_series().shift(-1)

    print("Analyzing heating demand distribution...")
    heating_stats = df['target_heating_demand'].describe()
    print("Heating Demand Statistics:")
    print(heating_stats)
    print(f"Zero demand (no heating): {(df['target_heating_demand'] == 0).sum()} samples ({(df['target_heating_demand'] == 0).mean()*100:.1f}%)")
    print(f"High demand (>10°C-hrs): {(df['target_heating_demand'] > 10).sum()} samples ({(df['target_heating_demand'] > 10).mean()*100:.1f}%)")

    print("Creating time features for target prediction time...")
    df = create_time_features(df, target_datetime_col='target_time') 

    print("Creating enhanced time features...")
    # Hour sin/cos
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    # Day of week sin/cos
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    # Month sin/cos
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    # Day of year sin/cos
    df['dayofyear_sin'] = np.sin(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    df['dayofyear_cos'] = np.cos(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    # is_weekend
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Season (categorical)
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

    print("Creating lag features...")
    # Lag features for historical context
    weather_cols_for_lags = ['apparent_temperature', 'temperature', 'humidity', 'windspeed', 'heating_demand_current']
    
    for col in weather_cols_for_lags:
        if col in df.columns:
            df = create_lag_features(df, col, lags=[1, 2, 3, 6, 12, 24])

    print("Creating forecast-based features...")
    # Use 1-hour ahead forecasts to predict heating demand
    forecast_features = []
    for col in ['temperature_forecast', 'humidity_forecast', 'windspeed_forecast', 
                'sw_radiation_forecast', 'cloud_cover_forecast', 'precipitation_forecast',
                'apparent_temperature_forecast']:
        if col in df.columns:
            forecast_features.append(col)
    
    # Create derived heating demand from forecasted apparent temperature
    if 'apparent_temperature_forecast' in df.columns:
        df['heating_demand_forecast'] = df['apparent_temperature_forecast'].apply(
            lambda x: max(0, 18.0 - x) if pd.notna(x) else np.nan
        )
        forecast_features.append('heating_demand_forecast')

    print("Creating interaction features...")
    interaction_features = []
    
    # Temperature-based interactions that affect heating demand
    if 'temperature_forecast' in df.columns and 'windspeed_forecast' in df.columns:
        # Wind chill effect - higher wind makes it feel colder
        df['wind_chill_factor'] = df['temperature_forecast'] - (df['windspeed_forecast'] * 0.5)
        interaction_features.append('wind_chill_factor')
    
    if 'temperature_forecast' in df.columns and 'humidity_forecast' in df.columns:
        # Humidity affects perceived temperature
        df['temp_humidity_interaction'] = df['temperature_forecast'] * (df['humidity_forecast'] / 100.0)
        interaction_features.append('temp_humidity_interaction')
    
    if 'sw_radiation_forecast' in df.columns and 'cloud_cover_forecast' in df.columns:
        # Solar heating effect
        cloud_fraction = (df['cloud_cover_forecast'] / 100.0).clip(0, 1)
        df['solar_heating_effect'] = df['sw_radiation_forecast'] * (1 - cloud_fraction)
        interaction_features.append('solar_heating_effect')

    # Combine all feature categories
    time_features = ['hour', 'dayofweek', 'month', 'dayofyear', 'quarter',
                     'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                     'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
                     'is_weekend', 'season']
    
    lagged_features = []
    for col in weather_cols_for_lags:
        if col in df.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                lagged_features.append(f'{col}_lag{lag}')
    
    all_features = time_features + lagged_features + forecast_features + interaction_features
    
    # Filter to only existing columns
    feature_columns = [f for f in all_features if f in df.columns]
    
    print(f"Total features selected for heating demand prediction: {len(feature_columns)}")
    print("Feature categories:")
    print(f"  Time features: {len(time_features)}")
    print(f"  Lagged features: {len(lagged_features)}")
    print(f"  Forecast features: {len(forecast_features)}")
    print(f"  Interaction features: {len(interaction_features)}")
    
    print("Dropping NaN values...")
    df_model = df.dropna(subset=feature_columns + ['target_heating_demand'])
    
    X = df_model[feature_columns]
    y = df_model['target_heating_demand']

    print(f"Final dataset shape: X-{X.shape}, y-{y.shape}")
    
    print("Analyzing target distribution after cleaning...")
    print(f"Target range: {y.min():.2f} to {y.max():.2f} degree-hours")
    print(f"Mean heating demand: {y.mean():.2f} degree-hours")
    print(f"Zero heating demand: {(y == 0).sum()} samples ({(y == 0).mean()*100:.1f}%)")

    print("Creating time-ordered train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Train set: {len(y_train)} samples")
    print(f"  Heating demand range: {y_train.min():.2f} to {y_train.max():.2f}")
    print(f"  Mean: {y_train.mean():.2f} degree-hours")
    print(f"  Zero demand: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    
    print(f"Test set: {len(y_test)} samples")  
    print(f"  Heating demand range: {y_test.min():.2f} to {y_test.max():.2f}")
    print(f"  Mean: {y_test.mean():.2f} degree-hours")
    print(f"  Zero demand: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")

    # Optuna hyperparameter tuning
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'random_state': 42,
            'early_stopping_rounds': 20
        }
        
        # GPU detection
        try:
            xgb.XGBRegressor(device='cuda')
            params['device'] = 'cuda'
        except:
            params['device'] = 'cpu'

        model = xgb.XGBRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        fold_rmses = []

        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      verbose=False)

            preds = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    print(f"Starting Optuna optimisation with {N_OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction='minimize')
    
    for _ in tqdm(range(N_OPTUNA_TRIALS), desc="Optuna Trials"):
        study.optimize(objective, n_trials=1, timeout=1800)

    print("Optuna study finished.")
    print(f"Best RMSE (CV): {study.best_value:.4f} degree-hours")
    
    # Train final model
    best_params = study.best_params
    
    try:
        xgb.XGBRegressor(device='cuda').fit(X_train.iloc[:2], y_train.iloc[:2])
        best_params['device'] = 'cuda'
    except:
        if 'device' in best_params: 
            del best_params['device']

    print("Training final heating demand model...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        **best_params,
        random_state=42,
        early_stopping_rounds=50
    )
    
    # Create evaluation set
    X_train_part, X_eval_part, y_train_part, y_eval_part = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )

    final_model.fit(X_train_part, y_train_part,
                    eval_set=[(X_eval_part, y_eval_part)],
                    verbose=50)

    print("Making predictions on test set...")
    y_pred = final_model.predict(X_test)

    # Ensure predictions are non-negative (heating demand can't be negative)
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"HEATING DEMAND MODEL RESULTS:")
    print(f"Test MAE: {mae:.4f} degree-hours")
    print(f"Test RMSE: {rmse:.4f} degree-hours")
    
    # Calculate percentage errors for non-zero demands
    non_zero_mask = y_test > 0
    if non_zero_mask.sum() > 0:
        mape_non_zero = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        print(f"MAPE (non-zero demands): {mape_non_zero:.1f}%")

    # Save model
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

    model_payload = {
        'model': final_model, 
        'features': feature_columns,
        'best_hyperparameters': study.best_params,
        'optuna_best_cv_rmse': study.best_value,
        'target_type': 'heating_demand_degree_hours',
        'base_temperature': 18.0
    }
    joblib.dump(model_payload, os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_TUNED))
    
    # Generate plots
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test, label='Actual Heating Demand', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted Heating Demand', linestyle='--', alpha=0.8)
    plt.title('Heating Demand Prediction: Actual vs. Predicted (degree-hours)')
    plt.xlabel('Time')
    plt.ylabel('Heating Demand (degree-hours)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_actual_vs_predicted.png'))
    plt.close()

    # Feature importance
    if hasattr(final_model, 'feature_importances_'):
        plt.figure(figsize=(12, max(6, len(feature_columns) // 3)))
        sorted_idx = final_model.feature_importances_.argsort()
        top_features = sorted_idx[-20:]  # Top 20 features
        plt.barh(np.array(feature_columns)[top_features], final_model.feature_importances_[top_features])
        plt.xlabel("XGBoost Feature Importance")
        plt.title("Heating Demand Predictor - Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_feature_importance.png'))
        plt.close()
        
    print("Heating demand predictor training completed!")
    print(f"Model saved to: {os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_TUNED)}")

if __name__ == '__main__':
    train_heating_demand_model() 