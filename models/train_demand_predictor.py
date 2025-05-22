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
MODEL_NAME_TUNED = "demand_predictor_xgb_tuned_with_forecasts.joblib"
TARGET_COLUMN = "apparent_temperature"
SHIFT_PERIODS = -1 # Predicting 1 hour ahead
N_OPTUNA_TRIALS = 500
N_CV_SPLITS = 5
OPTUNA_EARLY_STOPPING_ROUNDS = 20 # Early stopping within Optuna trials
XGB_EARLY_STOPPING_ROUNDS_FINAL = 50 # Early stopping for the final model fit

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
    # df_feat['weekofyear'] = df_feat[target_datetime_col].dt.isocalendar().week.astype(int) # pandas >= 1.1.0
    return df_feat

def create_lag_features(df, column_name, lags=[1, 2, 3, 6, 12, 24]):
    """Creates lag features for a specified column."""
    df_feat = df.copy()
    for lag in lags:
        df_feat[f'{column_name}_lag{lag}'] = df_feat[column_name].shift(lag)
    return df_feat

def create_lead_features(df, column_name, leads=[ -1 ]):
    """Creates lead features for a specified column. Default lead is -1 (1 step ahead)."""
    df_feat = df.copy()
    for lead_val in leads:
        # Lead feature names indicate the source column and how many steps ahead it is looking
        # e.g., temperature_lead1 (for lead_val = -1)
        df_feat[f'{column_name}_lead{abs(lead_val)}'] = df_feat[column_name].shift(lead_val)
    return df_feat

def train_demand_model():
    """Trains an XGBoost model to predict future apparent temperature using actual forecast features."""
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

    # Create a datetime column for the target prediction time to generate relevant time features
    df['target_time'] = df.index.to_series().shift(SHIFT_PERIODS)


    print("Creating time features for target prediction time...")
    df = create_time_features(df, target_datetime_col='target_time') 

    print("Creating enhanced time features (sin/cos, categorical) for target prediction time...")
    # Base features 'hour', 'dayofweek', 'month', 'dayofyear', 'quarter' are already created by create_time_features
    # using df['target_time'] and added to df. We will use them here.
    
    # Hour sin/cos
    df['hour_target_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_target_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    # Day of week sin/cos
    df['dayofweek_target_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_target_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    # Month sin/cos
    df['month_target_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0) # Month is 1-12
    df['month_target_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    # Day of year sin/cos
    df['dayofyear_target_sin'] = np.sin(2 * np.pi * (df['dayofyear'] - 1) / 365.0) # Dayofyear is 1-365/366
    df['dayofyear_target_cos'] = np.cos(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    # is_weekend (using 'dayofweek' which is Monday=0 to Sunday=6)
    df['is_weekend_target'] = (df['dayofweek'] >= 5).astype(int)
    # Season (using 'month')
    def get_season(month_col): # month_col is a Series
        conditions = [
            month_col.isin([12, 1, 2]), # Winter
            month_col.isin([3, 4, 5]),   # Spring
            month_col.isin([6, 7, 8]),   # Summer
            month_col.isin([9, 10, 11]) # Autumn
        ]
        choices = [0, 1, 2, 3] # Winter, Spring, Summer, Autumn
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
        else:
            print(f"Warning: Column {col} for lag feature creation not found.")

    # Explicitly define features for demand prediction
    # Base time features from create_time_features are already 'hour', 'dayofweek', 'month', 'dayofyear', 'quarter'
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
    if not actual_forecast_features:
        print("Warning: No actual forecast features found in the loaded DataFrame. Check column names in final_dataset_with_forecasts.csv")
    else:
        print(f"Identified actual forecast features: {actual_forecast_features}")

    print("Creating interaction features from forecast data...")
    interaction_features = []
    if 'temperature_forecast' in df.columns and 'humidity_forecast' in df.columns:
        df['temp_X_humidity_forecast'] = df['temperature_forecast'] * df['humidity_forecast']
        interaction_features.append('temp_X_humidity_forecast')
    if 'temperature_forecast' in df.columns and 'windspeed_forecast' in df.columns:
        df['temp_X_windspeed_forecast'] = df['temperature_forecast'] * df['windspeed_forecast']
        interaction_features.append('temp_X_windspeed_forecast')
    if 'sw_radiation_forecast' in df.columns and 'cloud_cover_forecast' in df.columns:
        # Assuming cloud_cover_forecast is 0-100 (percentage), convert to 0-1 fraction
        # Clip to ensure the fraction is within [0,1] after division if source data is noisy
        cloud_cover_fraction = (df['cloud_cover_forecast'] / 100.0).clip(0,1)
        df['sw_rad_eff_forecast'] = df['sw_radiation_forecast'] * (1 - cloud_cover_fraction)
        interaction_features.append('sw_rad_eff_forecast')
    
    print(f"Created interaction features: {interaction_features}")

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
    
    print(f"Created polynomial features: {polynomial_features}")
            
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
            if feature not in feature_columns: # Add only if not already present
                 feature_columns.append(feature)
    
    # Final check: ensure all features in feature_columns actually exist in df.columns
    # This step is crucial if some optional features (like specific forecasts) were not present.
    feature_columns = [f for f in feature_columns if f in df.columns]
    
    print(f"Total features selected for demand prediction ({len(feature_columns)}): {feature_columns}")
    if not feature_columns:
        print("Error: No features selected for modeling. Exiting.")
        return

    print("Dropping NaN values based on the selected features and target...")
    df_model = df.dropna(subset=feature_columns + ['target_apparent_temperature'])
    
    if df_model.empty:
        print("Error: DataFrame is empty after dropping NaN values. Check feature creation and data.")
        return

    X = df_model[feature_columns]
    y = df_model['target_apparent_temperature']

    print(f"Dataset shape for modeling: X-{X.shape}, y-{y.shape}")

    print("Splitting data into train and test sets (80/20, time-ordered)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Calculating sample weights for training...")
    k = 3
    sample_weight_train = np.ones(len(y_train))
    sample_weight_train[y_train > 18] = 1 + k # (y_train > 18) is a boolean, direct multiplication is fine.
                                             # No, it's not. (y_train > 18) is boolean.
                                             # We want to add k, not multiply by k.
                                             # Correct: sample_weight_train[y_train > 18] = 1 + k
    # Ensure y_train > 18 is treated as a condition to apply 1+k
    # The previous comment was a bit misleading. The calculation for sample_weight_train should be:
    # sample_weight_train = np.where(y_train > 18, 1 + k, 1).astype(float)
    # Let's use the simpler assignment for clarity as originally intended by the user request.
    # The condition y_train > 18 will select indices where the condition is true.
    # sample_weight_train[y_train > 18] directly modifies those elements.
    # For elements where y_train <= 18, their weight remains 1.0 as initialized.
    # So, sample_weight_train[y_train > 18] = 1 + k is correct.

    # --- Optuna Hyperparameter Tuning ---
    print(f"Starting Optuna hyperparameter tuning with {N_CV_SPLITS}-fold TimeSeriesSplit and {N_OPTUNA_TRIALS} trials...")

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),  # Increased from 10 to 12 for deeper trees
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 reg
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),   # L1 reg
            'random_state': 42,
            'early_stopping_rounds': OPTUNA_EARLY_STOPPING_ROUNDS # Moved here
            # 'tree_method': 'hist' 
        }
        
        # Add monotonic constraints to prevent flattening at high temperatures
        # Ensure temperature and apparent temperature forecasts have positive relationships with target
        monotone_constraints_dict = {}
        if 'temperature_forecast' in feature_columns:
            monotone_constraints_dict['temperature_forecast'] = 1
        if 'apparent_temperature_forecast' in feature_columns:
            monotone_constraints_dict['apparent_temperature_forecast'] = 1
        if 'sw_radiation_forecast' in feature_columns:
            monotone_constraints_dict['sw_radiation_forecast'] = 1  # More solar radiation = higher apparent temp
        
        if monotone_constraints_dict:
            # Convert to XGBoost format: dict with feature names as keys
            params['monotone_constraints'] = monotone_constraints_dict
            print(f"Applied monotonic constraints: {monotone_constraints_dict}")
        
        # Attempt to use GPU if available for Optuna trials
        try:
            # Check if CUDA is available
            xgb.XGBRegressor(device='cuda')
            params['device'] = 'cuda'
            print("Optuna trial will attempt to use GPU.")
        except xgb.core.XGBoostError:
            params['device'] = 'cpu'
            # print("GPU not available for Optuna trial, using CPU.")

        model = xgb.XGBRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        fold_rmses = []

        # Use tqdm for progress bar on folds if desired, but can be verbose
        # for fold, (train_idx, val_idx) in tqdm(enumerate(tscv.split(X_train)), total=N_CV_SPLITS, desc="CV Folds"):
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Get sample weights for the current training fold
            current_sample_weight_train_fold = sample_weight_train[train_idx]

            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      sample_weight=current_sample_weight_train_fold, # Pass sample weights here
                      verbose=False) # Suppress verbose output during Optuna CV

            preds = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    study = optuna.create_study(direction='minimize')
    
    print(f"Starting Optuna optimization with {N_OPTUNA_TRIALS} trials...")
    # Loop with tqdm for trial-level progress bar
    for _ in tqdm(range(N_OPTUNA_TRIALS), desc="Optuna Trials"):
        study.optimize(objective, n_trials=1, timeout=1800) # Optimize one trial at a time
        # Check if study should stop (e.g. if timeout was hit globally for the loop, though less direct here)
        # Optuna handles internal timeout for the optimize call if set globally in create_study or optimize, 
        # but here we are calling optimize for each trial.
        # The timeout in optimize(n_trials=1, timeout=1800) applies to that single trial.
        # A global timeout for the whole loop would need to be handled externally if desired.

    # study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=1800) # Original call

    print("Optuna study finished.")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best RMSE (CV): {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    best_params = study.best_params
    # Add device preference based on detection during trials, or ensure it is not present if GPU wasn't consistently usable
    try:
        xgb.XGBRegressor(device='cuda').fit(X_train.iloc[:2], y_train.iloc[:2])
        best_params['device'] = 'cuda'
        print("Final model training will attempt to use GPU.")
    except Exception:
        print("Final model training will use CPU.")
        if 'device' in best_params: del best_params['device']
        # best_params['tree_method'] = 'hist'

    print("Training final XGBoost Regressor model with best hyperparameters...")        # Add monotonic constraints to final model as well    final_monotone_constraints_dict = {}    if 'temperature_forecast' in feature_columns:        final_monotone_constraints_dict['temperature_forecast'] = 1    if 'apparent_temperature_forecast' in feature_columns:        final_monotone_constraints_dict['apparent_temperature_forecast'] = 1    if 'sw_radiation_forecast' in feature_columns:        final_monotone_constraints_dict['sw_radiation_forecast'] = 1        if final_monotone_constraints_dict:        best_params['monotone_constraints'] = final_monotone_constraints_dict        print(f"Applied monotonic constraints to final model: {final_monotone_constraints_dict}")        final_model = xgb.XGBRegressor(        objective='reg:squarederror',        eval_metric='rmse', # Good to have eval_metric for final fit too        **best_params,        random_state=42,        # n_estimators is set by best_params, early stopping will adjust it if eval_set is provided        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS_FINAL     )
    
    # Create an evaluation set for early stopping for the final model
    # This uses a portion of the full training set
    X_train_final_part, X_eval_final_part, y_train_final_part, y_eval_final_part = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )

    final_model.fit(X_train_final_part, y_train_final_part,
                    eval_set=[(X_eval_final_part, y_eval_final_part)],
                    verbose=50) # Can be True or a number to see training progress

    print("Making predictions on the test set with the tuned model...")
    y_pred = final_model.predict(X_test)

    print("Evaluating model...")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Create output directories if they don't exist
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

    print(f"Saving tuned model to {os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_TUNED)}...")
    model_payload = {
        'model': final_model, 
        'features': feature_columns,
        'best_hyperparameters': study.best_params,
        'optuna_best_cv_rmse': study.best_value,
        'optuna_study_summary': study.trials_dataframe().to_dict() # Save study summary
    }
    joblib.dump(model_payload, os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_TUNED))
    print("Tuned model saved successfully.")

    # --- Plotting ---
    print("Generating plots...")
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test, label='Actual Apparent Temperature', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted Apparent Temperature', linestyle='--', alpha=0.8)
    plt.title('Demand Prediction: Actual vs. Predicted Apparent Temperature (Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Apparent Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path_actual_vs_pred = os.path.join(FIGURES_OUTPUT_DIR, 'demand_actual_vs_predicted_test.png')
    plt.savefig(plot_path_actual_vs_pred)
    print(f"Actual vs. Predicted plot saved to {plot_path_actual_vs_pred}")
    plt.close()

    # Feature Importance Plot
    if hasattr(final_model, 'feature_importances_'):
        plt.figure(figsize=(12, max(6, len(feature_columns) // 2))) # Adjust height
        sorted_idx = final_model.feature_importances_.argsort()
        plt.barh(np.array(feature_columns)[sorted_idx], final_model.feature_importances_[sorted_idx])
        plt.xlabel("XGBoost Feature Importance")
        plt.title("Tuned Demand Predictor Feature Importance")
        plt.tight_layout()
        plot_path_feat_imp = os.path.join(FIGURES_OUTPUT_DIR, 'demand_tuned_feature_importance.png')
        plt.savefig(plot_path_feat_imp)
        print(f"Feature importance plot saved to {plot_path_feat_imp}")
        plt.close()

    # Optuna plots (if plotly is available)
    try:
        import plotly
        fig_opt_history = optuna.visualization.plot_optimization_history(study)
        fig_opt_history.write_image(os.path.join(FIGURES_OUTPUT_DIR, 'demand_optuna_optimization_history.png'))
        print(f"Optuna optimization history plot saved to {os.path.join(FIGURES_OUTPUT_DIR, 'demand_optuna_optimization_history.png')}")
        
        fig_param_importance = optuna.visualization.plot_param_importances(study)
        fig_param_importance.write_image(os.path.join(FIGURES_OUTPUT_DIR, 'demand_optuna_param_importances.png'))
        print(f"Optuna parameter importance plot saved to {os.path.join(FIGURES_OUTPUT_DIR, 'demand_optuna_param_importances.png')}")
    except ImportError:
        print("Plotly is not installed. Skipping Optuna diagnostic plots. Install with: pip install plotly kaleido")
    except Exception as e:
        print(f"Could not generate Optuna plots: {e}")
        
    print("Tuned demand predictor training script finished.")

if __name__ == '__main__':
    train_demand_model() 