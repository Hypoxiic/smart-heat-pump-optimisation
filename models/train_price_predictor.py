import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import optuna
import matplotlib.pyplot as plt
import os

# Attempt to import cupy for GPU data handling
try:
    import cupy
    CUPY_AVAILABLE = True
    print("CuPy library found. Will use for GPU data transfer if XGBoost is using GPU.")
except ImportError:
    CUPY_AVAILABLE = False
    cupy = None  # Define cupy as None if not available to avoid runtime errors on access
    print("CuPy library not found. GPU data transfer will rely on XGBoost's internal handling for CPU-based arrays.")

# --- Configuration ---
DATA_FILE_PATH = 'data/processed/featured_dataset_phase2.csv'
MODEL_OUTPUT_DIR = 'models'
REPORTS_DIR = 'reports/figures' # For saving plots
MODEL_FILENAME = 'price_predictor_xgb_tuned.joblib'
TARGET_COLUMN = 'cost_p_per_kwh'
N_CV_SPLITS = 5  # Number of splits for TimeSeriesSplit
OPTUNA_N_TRIALS = 100 # User confirmed to run 100 trials.
EARLY_STOPPING_ROUNDS_OPTUNA = 50
EARLY_STOPPING_ROUNDS_FINAL = 50 # For final model training, using in constructor
VALIDATION_SET_SIZE_FINAL_TRAIN = 0.1 # Proportion of training data to use for validation during final model fit

# Ensure output directories exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Feature Selection ---
FEATURE_COLUMNS = [
    'temperature', 'humidity', 'windspeed', 'apparent_temperature',
    'carbon_intensity_forecast', 'carbon_intensity_actual',
    'carbon_intensity_index_encoded',
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
    'cost_p_per_kwh_lag1h', 'cost_p_per_kwh_lag2h', 'cost_p_per_kwh_lag3h',
    'cost_p_per_kwh_lag24h', 'cost_p_per_kwh_lag48h', 'cost_p_per_kwh_lag168h',
    'carbon_intensity_actual_lag1h', 'carbon_intensity_actual_lag2h', 'carbon_intensity_actual_lag24h',
    'carbon_intensity_forecast_lag1h', 'carbon_intensity_forecast_lag2h', 'carbon_intensity_forecast_lag24h'
]

def get_gpu_params():
    try:
        # Attempt to initialize XGBoost with GPU to check availability
        # Create a minimal dataset for the check
        minimal_x = np.array([[1, 2], [3, 4]])
        minimal_y = np.array([1, 0])
        model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
        model.fit(minimal_x, minimal_y)
        print("GPU available for XGBoost. Using tree_method='hist' and device='cuda'.")
        return {'tree_method': 'hist', 'device': 'cuda'}
    except Exception as e:
        print(f"GPU not available or error during GPU check for XGBoost: {e}. Using tree_method='hist' (CPU).")
        return {'tree_method': 'hist'}

GPU_PARAMS = get_gpu_params()

# Determine if data should be moved to GPU (if model is on GPU and cupy is available)
SHOULD_MOVE_DATA_TO_GPU = GPU_PARAMS.get('device') == 'cuda' and CUPY_AVAILABLE

if GPU_PARAMS.get('device') == 'cuda' and not CUPY_AVAILABLE:
    print("Warning: XGBoost is configured to use GPU, but CuPy is not available. \n         Data will remain on CPU, which may lead to performance warnings or slower execution due to data transfer.")

# Helper function to convert data to CuPy array if needed
def to_gpu_if_needed(data_df_or_series):
    if SHOULD_MOVE_DATA_TO_GPU and data_df_or_series is not None:
        if hasattr(data_df_or_series, 'values'): # Check if it's a DataFrame or Series
            return cupy.asarray(data_df_or_series.values)
        else: # Assuming it's already a NumPy array or similar
            return cupy.asarray(data_df_or_series)
    return data_df_or_series

def load_and_preprocess_data(file_path, target_column, feature_columns):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    print(f"Data loaded. Shape: {df.shape}")

    df_trainable = df.dropna(subset=[target_column])
    print(f"Shape after dropping NaNs in target ('{target_column}'): {df_trainable.shape}")
    
    missing_features = [col for col in feature_columns if col not in df_trainable.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in DataFrame: {missing_features}")

    X = df_trainable[feature_columns]
    y = df_trainable[target_column]

    initial_rows = X.shape[0]
    X = X.dropna()
    y = y.loc[X.index]
    print(f"Dropped {initial_rows - X.shape[0]} rows from X due to NaNs in features.")
    print(f"Final shape for X: {X.shape}, y: {y.shape}")
    
    return X, y

def objective(trial, X_train, y_train, tscv):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse', # Explicitly set eval_metric
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA, # Moved to constructor
        **GPU_PARAMS
    }
    
    fold_rmse_scores = []
    try:
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            X_fold_train_processed = to_gpu_if_needed(X_fold_train)
            X_fold_val_processed = to_gpu_if_needed(X_fold_val)
            # y_fold_train and y_fold_val are typically fine as NumPy/Pandas series for XGBoost

            model = xgb.XGBRegressor(**params)
            model.fit(X_fold_train_processed, y_fold_train,
                      eval_set=[(X_fold_val_processed, y_fold_val)],
                      verbose=False)
            
            preds = model.predict(X_fold_val_processed)
            rmse = np.sqrt(mean_squared_error(y_fold_val.values, cupy.asnumpy(preds) if SHOULD_MOVE_DATA_TO_GPU and isinstance(preds, cupy.ndarray) else preds))
            fold_rmse_scores.append(rmse)
        
        return np.mean(fold_rmse_scores)
    except Exception as e:
        trial.report(float('inf'), step=0) # Report a very bad value
        # Optuna can prune this trial if configured, or just record it as failed
        print(f"Trial {trial.number} failed with error: {e}. Returning inf.")
        return float('inf') # Return a large value if a trial fails

def train_final_model(X_train, y_train, best_params):
    print("Training final XGBoost model with best parameters...")
    
    # Create a time-series split for validation during final training
    # The last VALIDATION_SET_SIZE_FINAL_TRAIN of X_train will be used for validation
    n_val_samples = int(len(X_train) * VALIDATION_SET_SIZE_FINAL_TRAIN)
    if n_val_samples == 0 and len(X_train) > 0: # Ensure at least 1 sample if dataset is tiny but not empty
        n_val_samples = 1
    elif len(X_train) == 0:
        raise ValueError("X_train is empty, cannot create validation set for final model training.")
    
    n_train_only_samples = len(X_train) - n_val_samples
    
    X_train_fold, X_val_fold = X_train.iloc[:n_train_only_samples], X_train.iloc[n_train_only_samples:]
    y_train_fold, y_val_fold = y_train.iloc[:n_train_only_samples], y_train.iloc[n_train_only_samples:]

    print(f"Final model training: X_train_fold shape: {X_train_fold.shape}, X_val_fold shape: {X_val_fold.shape}")

    X_train_fold_processed = to_gpu_if_needed(X_train_fold)
    X_val_fold_processed = to_gpu_if_needed(X_val_fold)

    final_params = {
        **GPU_PARAMS, 
        'objective': 'reg:squarederror', 
        'eval_metric': 'rmse', 
        'random_state': 42, 
        'n_jobs': -1,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL, # Moved to constructor
        **best_params
    }
    
    # Ensure n_estimators is present from best_params, if not, set a default (though Optuna should provide it)
    if 'n_estimators' not in final_params:
        print("Warning: 'n_estimators' not in best_params. Defaulting to 100 for final model.")
        final_params['n_estimators'] = 100

    final_model = xgb.XGBRegressor(**final_params)
    
    final_model.fit(X_train_fold_processed, y_train_fold, 
                      eval_set=[(X_train_fold_processed, y_train_fold), (X_val_fold_processed, y_val_fold)], 
                      verbose=False) # Set verbose to True or a number to see XGBoost's training output
                     
    print("Final model training complete.")
    # Retrieve evals_result. If data was on GPU, results keys might reflect that, but values are typically numpy arrays.
    evals_result = final_model.evals_result()
    return final_model, evals_result

def evaluate_model(model, X_test, y_test, model_name="Final Model"):
    print(f"Evaluating {model_name} on the test set...")
    X_test_processed = to_gpu_if_needed(X_test)
    predictions = model.predict(X_test_processed)

    # If predictions are on GPU (cupy array), convert them back to numpy for scikit-learn metrics
    if SHOULD_MOVE_DATA_TO_GPU and isinstance(predictions, cupy.ndarray):
        predictions_np = cupy.asnumpy(predictions)
    else:
        predictions_np = predictions

    mae = mean_absolute_error(y_test, predictions_np)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_np))
    
    print(f"  Test MAE: {mae:.4f} p/kWh")
    print(f"  Test RMSE: {rmse:.4f} p/kWh")
    return {'mae': mae, 'rmse': rmse, 'predictions': predictions_np}

def save_model_artifacts(model, feature_columns, best_params, study, model_dir, model_filename):
    output_path = os.path.join(model_dir, model_filename)
    print(f"Saving model, features, best_params, and Optuna study to {output_path}...")
    artifacts = {
        'model': model,
        'feature_columns': feature_columns,
        'best_hyperparameters': best_params,
        # 'optuna_study': study # Saving the study can be very large. Consider omitting or saving parts.
    }
    try:
        artifacts['optuna_best_value'] = study.best_value
        artifacts['optuna_best_trial_number'] = study.best_trial.number
    except Exception:
        pass # If study has no best_trial yet

    joblib.dump(artifacts, output_path)
    print("Model artifacts saved.")

def plot_actual_vs_predicted(y_test_df, predictions_series, save_path):
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_df.index, y_test_df.values, label='Actual Prices', alpha=0.7)
    plt.plot(predictions_series.index, predictions_series.values, label='Predicted Prices', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price (p/kWh)')
    plt.title('Actual vs. Predicted Prices on Test Set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Actual vs. Predicted plot saved to {save_path}")

def plot_feature_importance(model, feature_columns, save_path):
    plt.figure(figsize=(12, 8)) 
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    N_top_features = min(len(feature_columns), 25) 
    
    plt.title(f'Top {N_top_features} Feature Importances')
    plt.bar(range(N_top_features), importances[sorted_indices][:N_top_features], align='center')
    plt.xticks(range(N_top_features), np.array(feature_columns)[sorted_indices][:N_top_features], rotation=90, ha='right')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

def plot_optuna_diagnostics(study, save_dir):
    if not study or not hasattr(study, 'trials') or not study.trials:
        print("Optuna study is empty or invalid. Skipping diagnostic plots.")
        return
    try:
        if optuna.visualization.is_available():
                        fig_history = optuna.visualization.plot_optimization_history(study)            fig_history.write_image(os.path.join(save_dir, "optuna_optimisation_history.png"))            print(f"Optuna optimisation history plot saved to {save_dir}")

            fig_param_importance = optuna.visualization.plot_param_importances(study)
            fig_param_importance.write_image(os.path.join(save_dir, "optuna_param_importances.png"))
            print(f"Optuna parameter importances plot saved to {save_dir}")
        else:
            print("Optuna visualization is not available. Install plotly and kaleido. Skipping diagnostic plots.")
    except Exception as e:
        print(f"Could not generate Optuna plots: {e}")

def plot_training_loss_curve(evals_result, save_path):
    results = evals_result
    # Ensure metrics are numpy arrays for plotting if they came from cupy
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']

    if SHOULD_MOVE_DATA_TO_GPU:
        if isinstance(train_rmse, cupy.ndarray):
            train_rmse = cupy.asnumpy(train_rmse)
        if isinstance(val_rmse, cupy.ndarray):
            val_rmse = cupy.asnumpy(val_rmse)

    epochs = len(train_rmse)
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_rmse, label='Train RMSE')
    plt.plot(x_axis, val_rmse, label='Validation RMSE')
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Epoch (Boosting Round)')
    plt.title('XGBoost Training and Validation RMSE')
    plt.tight_layout()
    try:
        plt.savefig(save_path)
        plt.close()
        print(f"Training loss curve saved to {save_path}")
    except Exception as e:
        print(f"Error saving training loss curve: {e}")

def main():
    X, y = load_and_preprocess_data(DATA_FILE_PATH, TARGET_COLUMN, FEATURE_COLUMNS)

    if X.empty or y.empty or X.shape[0] < N_CV_SPLITS * 2: # Ensure enough samples for CV splits
        print("No data or insufficient data available for training after preprocessing. Exiting.")
        return

    n_samples = len(X)
    n_test_samples = int(n_samples * 0.2)
    # Ensure n_test_samples is not too small, and train set is adequate for N_CV_SPLITS
    if n_test_samples < 50 or (n_samples - n_test_samples) < N_CV_SPLITS * 10: # Heuristic checks
        print(f"Dataset size ({n_samples}) is too small for reliable train/test split and {N_CV_SPLITS}-fold CV. Adjust parameters or get more data.")
        # Fallback to a simpler split or abort
        if n_samples < 100: return # Abort if extremely small
        n_test_samples = max(10, int(n_samples * 0.1)) # Smaller test set if needed
    
    n_train_samples = n_samples - n_test_samples
    
    X_train, X_test = X.iloc[:n_train_samples], X.iloc[n_train_samples:]
    y_train, y_test = y.iloc[:n_train_samples], y.iloc[n_train_samples:]
    
    print(f"Initial split: Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Initial split: Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    print(f"\nStarting Optuna hyperparameter optimisation ({OPTUNA_N_TRIALS} trials)...")
    study = None
    try:
        study = optuna.create_study(direction='minimize', study_name='xgb_price_prediction')
        # Use a lambda to pass additional arguments to the objective function that Optuna calls.
        study.optimize(lambda trial: objective(trial, X_train, y_train, tscv), 
                         n_trials=OPTUNA_N_TRIALS, 
                         show_progress_bar=True)
        
        best_params_optuna = study.best_params
        print("\nBest hyperparameters found by Optuna:")
        for key, value in best_params_optuna.items():
            print(f"  {key}: {value}")
        print(f"  Best CV RMSE: {study.best_value:.4f}")

        final_model, final_model_history = train_final_model(X_train, y_train, best_params_optuna)
        test_set_evaluation = evaluate_model(final_model, X_test, y_test)
        save_model_artifacts(final_model, FEATURE_COLUMNS, best_params_optuna, study, MODEL_OUTPUT_DIR, MODEL_FILENAME)

        print("\nGenerating and saving plots...")
        # Create Series for predictions to carry index for plotting
        predictions_series = pd.Series(test_set_evaluation['predictions'], index=X_test.index)
        plot_actual_vs_predicted(y_test, predictions_series, os.path.join(REPORTS_DIR, 'actual_vs_predicted_test.png'))
        plot_feature_importance(final_model, FEATURE_COLUMNS, os.path.join(REPORTS_DIR, 'feature_importance.png'))
        plot_optuna_diagnostics(study, REPORTS_DIR)
        plot_training_loss_curve(final_model_history, os.path.join(REPORTS_DIR, 'price_predictor_training_loss.png'))
        
        print(f"\nPrice predictor tuning, training, and evaluation complete.")
        print(f"Final Test Set Metrics: MAE={test_set_evaluation['mae']:.4f}, RMSE={test_set_evaluation['rmse']:.4f}")
    except optuna.exceptions.TrialPruned as e:
        print(f"Optuna study interrupted by pruning or error: {e}")
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
    finally:
        print(f"Plots saved in: {REPORTS_DIR}")
        print(f"Tuned model and artifacts (if successful) saved in: {MODEL_OUTPUT_DIR}")

if __name__ == '__main__':
    main() 