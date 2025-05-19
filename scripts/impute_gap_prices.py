import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
INPUT_DATA_FILE_PATH = 'data/processed/featured_dataset_phase2.csv'
MODEL_ARTIFACTS_PATH = 'models/price_predictor_xgb_tuned.joblib'
OUTPUT_DATA_FILE_PATH = 'data/processed/featured_dataset_phase3_imputed.csv'
TARGET_COLUMN = 'cost_p_per_kwh'

# Date range for the Gap Period
GAP_START_DATE = pd.Timestamp('2023-12-12 00:00:00+00:00')
GAP_END_DATE = pd.Timestamp('2024-04-02 23:59:59+00:00') # Inclusive end for filtering

def load_data_and_model(data_path, model_artifacts_path):
    """Loads the dataset and the trained price prediction model with its features."""
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    print(f"Dataset loaded. Shape: {df.shape}")

    print(f"Loading model artifacts from {model_artifacts_path}...")
    model_artifacts = joblib.load(model_artifacts_path)
    model = model_artifacts['model']
    feature_columns = model_artifacts['feature_columns']
    print("Model and feature list loaded.")
    
    return df, model, feature_columns

def impute_prices(df, model, feature_columns, target_column, gap_start, gap_end):
    """Imputes missing prices in the target_column for the specified gap period using the model."""
    df_imputed = df.copy()

    # Identify rows in the gap period where the target is NaN
    gap_mask = (df_imputed.index >= gap_start) & (df_imputed.index <= gap_end) & (df_imputed[target_column].isna())
    
    rows_to_impute_idx = df_imputed[gap_mask].index
    
    if rows_to_impute_idx.empty:
        print("No rows found in the specified gap period with NaN target values. No imputation needed.")
        return df_imputed

    print(f"Found {len(rows_to_impute_idx)} rows in the gap period ({gap_start} to {gap_end}) with NaN '{target_column}' to impute.")

    # Prepare features for prediction for these rows
    X_to_predict = df_imputed.loc[rows_to_impute_idx, feature_columns]
    
    # XGBoost can handle NaNs in input features if it was trained to expect them or has default handling.
    # Our previous training script did X.dropna() on features, so the model might not be explicitly trained for NaNs.
    # However, XGBoost prediction typically assigns NaNs to a default split direction.
    # For robustness, one might consider filling NaNs in X_to_predict if there's a clear strategy,
    # but for now, we rely on XGBoost's default NaN handling during prediction.
    print(f"Predicting prices for {X_to_predict.shape[0]} rows...")
    predicted_prices = model.predict(X_to_predict)
    
    # Fill NaN target values and update 'is_price_imputed' column
    df_imputed.loc[rows_to_impute_idx, target_column] = predicted_prices
    df_imputed.loc[rows_to_impute_idx, 'is_price_imputed'] = True # Ensure this column exists and is False by default
    
    # Verify imputation
    print(f"Imputed {df_imputed[target_column].loc[rows_to_impute_idx].notna().sum()} values for '{target_column}'.")
    print(f"Number of rows where '{target_column}' is still NaN after imputation in gap: {df_imputed.loc[rows_to_impute_idx, target_column].isna().sum()}")
    print(f"Number of rows where 'is_price_imputed' is True: {df_imputed['is_price_imputed'].sum()}")
    
    return df_imputed

def main():
    df, model, feature_columns = load_data_and_model(INPUT_DATA_FILE_PATH, MODEL_ARTIFACTS_PATH)
    
    # Ensure 'is_price_imputed' column exists and is False by default if not already present
    if 'is_price_imputed' not in df.columns:
        print("'is_price_imputed' column not found, adding and initializing to False.")
        df['is_price_imputed'] = False
    else:
        # Ensure it's boolean and reset if necessary for a clean run (optional, depends on desired behavior)
        # For this script, we assume it was correctly initialized to False in featured_dataset_phase2
        pass 

    df_with_imputed_prices = impute_prices(df, model, feature_columns, TARGET_COLUMN, GAP_START_DATE, GAP_END_DATE)
    
    print(f"\nSaving dataset with imputed prices to {OUTPUT_DATA_FILE_PATH}...")
    df_with_imputed_prices.to_csv(OUTPUT_DATA_FILE_PATH)
    print("Dataset saved.")
    
    # Brief summary of changes
    original_nan_count = df[TARGET_COLUMN].isna().sum()
    imputed_nan_count = df_with_imputed_prices[TARGET_COLUMN].isna().sum()
    print(f"\nOriginal NaN count in '{TARGET_COLUMN}': {original_nan_count}")
    print(f"NaN count in '{TARGET_COLUMN}' after imputation: {imputed_nan_count}")
    print(f"Number of values imputed: {original_nan_count - imputed_nan_count}")
    print(f"Total rows marked as imputed: {df_with_imputed_prices['is_price_imputed'].sum()}")

if __name__ == '__main__':
    main() 