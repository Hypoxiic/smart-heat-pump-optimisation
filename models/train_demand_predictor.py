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
DATA_PATH = "data/processed/featured_dataset_phase3_imputed.csv"
MODEL_NAME_TUNED = "demand_predictor_xgb_tuned.joblib"
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

def train_demand_model():
    """Trains an XGBoost model to predict future apparent temperature."""
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
    # Pass df['target_time'] to ensure features are for the future prediction point
    df = create_time_features(df, target_datetime_col='target_time') 

    print("Creating lag features...")
    weather_cols_for_lags = ['apparent_temperature', 'temperature', 'humidity', 'windspeed']
    for col in weather_cols_for_lags:
        if col in df.columns:
            df = create_lag_features(df, col, lags=[1, 2, 3, 6, 12, 24]) # Lags based on current time's data
        else:
            print(f"Warning: Column {col} not found for lag feature creation.")

    # Explicitly define features for demand prediction
    time_features_for_model = ['hour', 'dayofweek', 'month', 'dayofyear', 'quarter']
    
    lagged_weather_features = []
    for col in weather_cols_for_lags: # weather_cols_for_lags = ['apparent_temperature', 'temperature', 'humidity', 'windspeed']
        for lag in [1, 2, 3, 6, 12, 24]:
            lagged_weather_features.append(f'{col}_lag{lag}')
            
    feature_columns = time_features_for_model + lagged_weather_features
    
    # Ensure all selected features actually exist in the dataframe before proceeding
    missing_model_features = [f for f in feature_columns if f not in df.columns]
    if missing_model_features:
        print(f"Error: The following expected model features are missing from the DataFrame: {missing_model_features}")
        print(f"Debug: Available columns in DataFrame after feature creation: {df.columns.tolist()}")
        return
    
    # Define features (excluding original weather columns that are not lagged, target, and temporary time columns)
    # features_to_exclude = [TARGET_COLUMN, 'target_apparent_temperature', 'temperature', 
    #                        'humidity', 'windspeed', 'carbon_intensity_forecast', 
    #                        'carbon_intensity_actual', 'carbon_intensity_index', 
    #                        'cost_p_per_kwh', 'is_price_imputed', 'target_time']
                           
    # All columns that are not in features_to_exclude and not the target itself
    # feature_columns = [col for col in df.columns if col not in features_to_exclude]
    
    print(f"Explicitly selected features for demand prediction: {feature_columns}")

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

    # --- Optuna Hyperparameter Tuning ---
    print(f"Starting Optuna hyperparameter tuning with {N_CV_SPLITS}-fold TimeSeriesSplit and {N_OPTUNA_TRIALS} trials...")

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 reg
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),   # L1 reg
            'random_state': 42,
            'early_stopping_rounds': OPTUNA_EARLY_STOPPING_ROUNDS # Moved here
            # 'tree_method': 'hist' 
        }
        
        # Attempt to use GPU if available for Optuna trials
        try:
            # Check if XGBoost was built with CUDA support and a GPU is detected
            # This is a proxy check; a more robust check might involve cupy or nvidia-smi
            xgb.XGBRegressor(device='cuda').fit(X_train.iloc[:2], y_train.iloc[:2]) # Test with a tiny bit of data
            params['device'] = 'cuda'
            print("Optuna trial will attempt to use GPU.")
        except Exception:
            print("Optuna trial will use CPU.")
            if 'device' in params: del params['device'] # Ensure it is not set if GPU fails
            # params['tree_method'] = 'hist' # Fallback for CPU if needed

        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        rmses = []

        # Add tqdm progress bar for the cross-validation folds
        for fold, (train_idx, val_idx) in enumerate(tqdm(tscv.split(X_train), total=N_CV_SPLITS, desc=f'Trial {trial.number} CV', leave=False)):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model_cv = xgb.XGBRegressor(**params) # early_stopping_rounds is now in params
            model_cv.fit(X_train_fold, y_train_fold,
                         eval_set=[(X_val_fold, y_val_fold)],
                         verbose=False)
            
            preds = model_cv.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
            rmses.append(rmse)
        
        return np.mean(rmses)

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

    print("Training final XGBoost Regressor model with best hyperparameters...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse', # Good to have eval_metric for final fit too
        **best_params,
        random_state=42,
        # n_estimators is set by best_params, early stopping will adjust it if eval_set is provided
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS_FINAL 
    )
    
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