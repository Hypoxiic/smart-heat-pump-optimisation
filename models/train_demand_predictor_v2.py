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
MODEL_NAME_TUNED = "demand_predictor_xgb_tuned_v2_monotonic.joblib"
TARGET_COLUMN = "apparent_temperature"
SHIFT_PERIODS = -1 # Predicting 1 hour ahead
N_OPTUNA_TRIALS = 200  # Reduced for faster testing
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

def train_demand_model():
    """Trains an XGBoost model with monotonic constraints to predict future apparent temperature."""
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    if 'time' not in df.columns:
        print("Error: 'time' column not found in the dataset.")
        return
        
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.sort_index()

    print("Creating target variable...")
    df['target_apparent_temperature'] = df[TARGET_COLUMN].shift(SHIFT_PERIODS)
    df['target_time'] = df.index.to_series().shift(SHIFT_PERIODS)

    print("Creating time features for target prediction time...")
    df = create_time_features(df, target_datetime_col='target_time') 

    print("Creating enhanced time features (sin/cos, categorical) for target prediction time...")
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

    print("Creating lag features (historical actuals)...")
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

    # Define features for demand prediction
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

    print("Creating interaction features from forecast data...")
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

    print("Creating polynomial features from forecast data...")
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
    
    # Final check: ensure all features actually exist in df.columns
    feature_columns = [f for f in feature_columns if f in df.columns]
    
    print(f"Total features selected for demand prediction ({len(feature_columns)}): {feature_columns}")
    
    print("Dropping NaN values based on the selected features and target...")
    df_model = df.dropna(subset=feature_columns + ['target_apparent_temperature'])
    
    X = df_model[feature_columns]
    y = df_model['target_apparent_temperature']

    print(f"Dataset shape for modeling: X-{X.shape}, y-{y.shape}")

    print("Splitting data into train and test sets (80/20, time-ordered)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Calculating sample weights for training (4x weight for temperatures > 18°C)...")
    k = 3
    sample_weight_train = np.ones(len(y_train))
    sample_weight_train[y_train > 18] = 1 + k

    # --- Optuna Hyperparameter Tuning ---
    print(f"Starting Optuna hyperparameter tuning with {N_CV_SPLITS}-fold TimeSeriesSplit and {N_OPTUNA_TRIALS} trials...")

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),  # Increased range for deeper trees
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'random_state': 42,
            'early_stopping_rounds': OPTUNA_EARLY_STOPPING_ROUNDS
        }
        
        # Add monotonic constraints to prevent flattening at high temperatures
        monotone_constraints_dict = {}
        if 'temperature_forecast' in feature_columns:
            monotone_constraints_dict['temperature_forecast'] = 1
        if 'apparent_temperature_forecast' in feature_columns:
            monotone_constraints_dict['apparent_temperature_forecast'] = 1
        if 'sw_radiation_forecast' in feature_columns:
            monotone_constraints_dict['sw_radiation_forecast'] = 1
        
        if monotone_constraints_dict:
            params['monotone_constraints'] = monotone_constraints_dict
        
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
            
            current_sample_weight_train_fold = sample_weight_train[train_idx]

            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      sample_weight=current_sample_weight_train_fold,
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
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best RMSE (CV): {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    best_params = study.best_params
    
    # GPU detection for final model
    try:
        xgb.XGBRegressor(device='cuda').fit(X_train.iloc[:2], y_train.iloc[:2])
        best_params['device'] = 'cuda'
        print("Final model training will attempt to use GPU.")
    except Exception:
        print("Final model training will use CPU.")
        if 'device' in best_params: 
            del best_params['device']

    print("Training final XGBoost Regressor model with best hyperparameters...")
    
    # Add monotonic constraints to final model
    final_monotone_constraints_dict = {}
    if 'temperature_forecast' in feature_columns:
        final_monotone_constraints_dict['temperature_forecast'] = 1
    if 'apparent_temperature_forecast' in feature_columns:
        final_monotone_constraints_dict['apparent_temperature_forecast'] = 1
    if 'sw_radiation_forecast' in feature_columns:
        final_monotone_constraints_dict['sw_radiation_forecast'] = 1
    
    if final_monotone_constraints_dict:
        best_params['monotone_constraints'] = final_monotone_constraints_dict
        print(f"Applied monotonic constraints to final model: {final_monotone_constraints_dict}")
    
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        **best_params,
        random_state=42,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS_FINAL 
    )
    
    # Create evaluation set for early stopping
    X_train_final_part, X_eval_final_part, y_train_final_part, y_eval_final_part = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )
    
    # Sample weights for final training
    sample_weight_train_final_part = sample_weight_train[:len(X_train_final_part)]

    final_model.fit(X_train_final_part, y_train_final_part,
                    eval_set=[(X_eval_final_part, y_eval_final_part)],
                    sample_weight=sample_weight_train_final_part,
                    verbose=50)

    print("Making predictions on the test set with the tuned model...")
    y_pred = final_model.predict(X_test)

    print("Evaluating model...")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Create output directories
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

    print(f"Saving tuned model to {os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_TUNED)}...")
    model_payload = {
        'model': final_model, 
        'features': feature_columns,
        'best_hyperparameters': study.best_params,
        'optuna_best_cv_rmse': study.best_value,
        'monotonic_constraints': final_monotone_constraints_dict,
        'optuna_study_summary': study.trials_dataframe().to_dict()
    }
    joblib.dump(model_payload, os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_TUNED))
    print("Tuned model saved successfully.")

    # Generate plots
    print("Generating plots...")
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test, label='Actual Apparent Temperature', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted Apparent Temperature', linestyle='--', alpha=0.8)
    plt.title('Demand Prediction V2 (Monotonic): Actual vs. Predicted Apparent Temperature (Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Apparent Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path_actual_vs_pred = os.path.join(FIGURES_OUTPUT_DIR, 'demand_actual_vs_predicted_test_v2_monotonic.png')
    plt.savefig(plot_path_actual_vs_pred)
    print(f"Actual vs. Predicted plot saved to {plot_path_actual_vs_pred}")
    plt.close()

    # Feature Importance Plot
    if hasattr(final_model, 'feature_importances_'):
        plt.figure(figsize=(12, max(6, len(feature_columns) // 2)))
        sorted_idx = final_model.feature_importances_.argsort()
        plt.barh(np.array(feature_columns)[sorted_idx], final_model.feature_importances_[sorted_idx])
        plt.xlabel("XGBoost Feature Importance")
        plt.title("Demand Predictor V2 (Monotonic) Feature Importance")
        plt.tight_layout()
        plot_path_feat_imp = os.path.join(FIGURES_OUTPUT_DIR, 'demand_tuned_feature_importance_v2_monotonic.png')
        plt.savefig(plot_path_feat_imp)
        print(f"Feature importance plot saved to {plot_path_feat_imp}")
        plt.close()
        
    print("Tuned demand predictor V2 (monotonic) training script finished.")

if __name__ == '__main__':
    train_demand_model() 