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
MODEL_NAME_TUNED = "demand_predictor_xgb_tuned_stratified.joblib"
TARGET_COLUMN = "apparent_temperature"
SHIFT_PERIODS = -1 # Predicting 1 hour ahead
N_OPTUNA_TRIALS = 100  # Reduced for testing
N_CV_SPLITS = 5
OPTUNA_EARLY_STOPPING_ROUNDS = 20
XGB_EARLY_STOPPING_ROUNDS_FINAL = 50

def create_time_features(df, target_datetime_col):
    """Creates time-based features from a datetime column for the target prediction time."""
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

def create_stratified_split(df, test_size=0.2, random_state=42):
    """Create a stratified split that ensures similar temperature distributions in train/test."""
    
    # Create temperature bins for stratification
    df['temp_bin'] = pd.cut(df['target_apparent_temperature'], 
                           bins=[-10, 0, 5, 10, 15, 20, 25, 40], 
                           labels=['very_cold', 'cold', 'cool', 'mild', 'warm', 'hot', 'very_hot'])
    
    # Create month bins for seasonal balance
    df['season'] = df['month'] % 12 // 3  # 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    
    # Combine temp and season for stratification
    df['strata'] = df['temp_bin'].astype(str) + '_' + df['season'].astype(str)
    
    # Ensure we have enough samples in each stratum
    strata_counts = df['strata'].value_counts()
    valid_strata = strata_counts[strata_counts >= 10].index  # At least 10 samples per stratum
    df_valid = df[df['strata'].isin(valid_strata)]
    
    print(f"Using {len(valid_strata)} strata out of {len(strata_counts)} total")
    print(f"Valid samples: {len(df_valid)} out of {len(df)}")
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    train_indices, test_indices = train_test_split(
        df_valid.index, 
        test_size=test_size, 
        stratify=df_valid['strata'],
        random_state=random_state
    )
    
    return train_indices, test_indices

def train_demand_model():
    """Trains an XGBoost model with proper stratified split to handle temperature distribution."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    df = df.set_index('time').sort_index()

    print("Creating target variable...")
    df['target_apparent_temperature'] = df[TARGET_COLUMN].shift(SHIFT_PERIODS)
    df['target_time'] = df.index.to_series().shift(SHIFT_PERIODS)

    print("Creating time features for target prediction time...")
    df = create_time_features(df, target_datetime_col='target_time') 

    print("Creating enhanced time features...")
    # Hour sin/cos
    df['hour_target_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_target_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    # Day of week sin/cos
    df['dayofweek_target_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_target_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    # Month sin/cos
    df['month_target_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['month_target_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    # Day of year sin/cos
    df['dayofyear_target_sin'] = np.sin(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    df['dayofyear_target_cos'] = np.cos(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    # is_weekend
    df['is_weekend_target'] = (df['dayofweek'] >= 5).astype(int)
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
    df['season_target'] = get_season(df['month'])

    print("Creating lag features...")
    weather_cols_for_lags = ['apparent_temperature', 'temperature', 'humidity', 'windspeed']
    if 'sw_radiation_forecast' in df.columns:
        weather_cols_for_lags.append('sw_radiation_forecast')
    if 'cloud_cover_forecast' in df.columns:
        weather_cols_for_lags.append('cloud_cover_forecast')
    if 'precipitation_forecast' in df.columns:
         weather_cols_for_lags.append('precipitation_forecast')

    for col in weather_cols_for_lags:
        if col in df.columns:
            df = create_lag_features(df, col, lags=[1, 2, 3, 6, 12, 24])

    # Define features
    time_features_for_model = ['hour', 'dayofweek', 'month', 'dayofyear', 'quarter',
                               'hour_target_sin', 'hour_target_cos', 
                               'dayofweek_target_sin', 'dayofweek_target_cos',
                               'month_target_sin', 'month_target_cos',
                               'dayofyear_target_sin', 'dayofyear_target_cos',
                               'is_weekend_target', 'season_target']
    
    lagged_weather_features = []
    for col in weather_cols_for_lags: 
        if col in df.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                lagged_weather_features.append(f'{col}_lag{lag}')
    
    actual_forecast_features = [
        'temperature_forecast', 
        'humidity_forecast', 
        'windspeed_forecast', 
        'sw_radiation_forecast', 
        'cloud_cover_forecast', 
        'precipitation_forecast',
        'apparent_temperature_forecast'
    ]
    actual_forecast_features = [f for f in actual_forecast_features if f in df.columns]

    print("Creating interaction features...")
    interaction_features = []
    if 'temperature_forecast' in df.columns and 'humidity_forecast' in df.columns:
        df['temp_X_humidity_forecast'] = df['temperature_forecast'] * df['humidity_forecast']
        interaction_features.append('temp_X_humidity_forecast')
    if 'temperature_forecast' in df.columns and 'windspeed_forecast' in df.columns:
        df['temp_X_windspeed_forecast'] = df['temperature_forecast'] * df['windspeed_forecast']
        interaction_features.append('temp_X_windspeed_forecast')
    if 'sw_radiation_forecast' in df.columns and 'cloud_cover_forecast' in df.columns:
        cloud_cover_fraction = (df['cloud_cover_forecast'] / 100.0).clip(0,1)
        df['sw_rad_eff_forecast'] = df['sw_radiation_forecast'] * (1 - cloud_cover_fraction)
        interaction_features.append('sw_rad_eff_forecast')

    print("Creating polynomial features...")
    polynomial_features = []
    if 'temperature_forecast' in df.columns:
        df['temperature_forecast_sq'] = df['temperature_forecast']**2
        polynomial_features.append('temperature_forecast_sq')
    if 'humidity_forecast' in df.columns: 
        df['humidity_forecast_sq'] = df['humidity_forecast']**2
        polynomial_features.append('humidity_forecast_sq')
    if 'sw_radiation_forecast' in df.columns:
        df['sw_radiation_forecast_sq'] = df['sw_radiation_forecast']**2
        polynomial_features.append('sw_radiation_forecast_sq')
            
    base_feature_categories = [
        time_features_for_model, 
        lagged_weather_features, 
        actual_forecast_features, 
        interaction_features, 
        polynomial_features
    ]
    
    feature_columns = []
    for category in base_feature_categories:
        for feature in category:
            if feature not in feature_columns:
                 feature_columns.append(feature)
    
    feature_columns = [f for f in feature_columns if f in df.columns]
    
    print(f"Total features selected: {len(feature_columns)}")
    
    print("Dropping NaN values...")
    df_model = df.dropna(subset=feature_columns + ['target_apparent_temperature'])
    
    print(f"Dataset shape for modeling: {df_model.shape}")
    
    # **CRITICAL FIX: Use stratified split instead of chronological**
    print("Creating stratified train/test split...")
    train_indices, test_indices = create_stratified_split(df_model, test_size=0.2, random_state=42)
    
    X = df_model[feature_columns]
    y = df_model['target_apparent_temperature']
    
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    
    print(f"Train set: {len(y_train)} samples")
    print(f"  Temperature range: {y_train.min():.1f}°C to {y_train.max():.1f}°C")
    print(f"  >20°C: {(y_train > 20).sum()} ({(y_train > 20).mean()*100:.1f}%)")
    
    print(f"Test set: {len(y_test)} samples")
    print(f"  Temperature range: {y_test.min():.1f}°C to {y_test.max():.1f}°C")
    print(f"  >20°C: {(y_test > 20).sum()} ({(y_test > 20).mean()*100:.1f}%)")

    # No sample weights needed with proper split
    print("Training XGBoost with stratified data...")

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
            'early_stopping_rounds': OPTUNA_EARLY_STOPPING_ROUNDS
        }
        
        # GPU detection
        try:
            xgb.XGBRegressor(device='cuda')
            params['device'] = 'cuda'
        except xgb.core.XGBoostError:
            params['device'] = 'cpu'

        model = xgb.XGBRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      verbose=False)

            preds = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    study = optuna.create_study(direction='minimize')
    
    print(f"Starting Optuna optimization with {N_OPTUNA_TRIALS} trials...")
    for _ in tqdm(range(N_OPTUNA_TRIALS), desc="Optuna Trials"):
        study.optimize(objective, n_trials=1, timeout=1800)

    print("Optuna study finished.")
    print(f"Best RMSE (CV): {study.best_value:.4f}")
    
    best_params = study.best_params
    
    # GPU detection for final model
    try:
        xgb.XGBRegressor(device='cuda').fit(X_train.iloc[:2], y_train.iloc[:2])
        best_params['device'] = 'cuda'
    except Exception:
        if 'device' in best_params: 
            del best_params['device']

    print("Training final model...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        **best_params,
        random_state=42,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS_FINAL 
    )
    
    # Create evaluation set for early stopping
    X_train_part, X_eval_part, y_train_part, y_eval_part = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=True, random_state=42
    )

    final_model.fit(X_train_part, y_train_part,
                    eval_set=[(X_eval_part, y_eval_part)],
                    verbose=50)

    print("Making predictions...")
    y_pred = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"STRATIFIED SPLIT RESULTS:")
    print(f"Test MAE: {mae:.4f}°C")
    print(f"Test RMSE: {rmse:.4f}°C")

    # Save model
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

    model_payload = {
        'model': final_model, 
        'features': feature_columns,
        'best_hyperparameters': study.best_params,
        'optuna_best_cv_rmse': study.best_value,
        'split_type': 'stratified',
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist()
    }
    joblib.dump(model_payload, os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_TUNED))
    
    # Generate comparison plots
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.values, label='Actual', alpha=0.8)  # Remove time index since it's not sequential
    plt.plot(y_pred, label='Predicted', linestyle='--', alpha=0.8)
    plt.title('Stratified Split: Actual vs. Predicted Apparent Temperature')
    plt.xlabel('Sample Index (Random Order)')
    plt.ylabel('Apparent Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'demand_stratified_actual_vs_pred.png'))
    plt.close()

    print("Stratified split model training completed!")

if __name__ == '__main__':
    train_demand_model() 