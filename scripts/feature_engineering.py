import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    return df

def encode_carbon_intensity_index(df):
    """Encodes the 'carbon_intensity_index' column."""
    if 'carbon_intensity_index' in df.columns:
        # Define the order of categories
        intensity_categories = ['very low', 'low', 'moderate', 'high', 'very high']
        df['carbon_intensity_index_encoded'] = pd.Categorical(
            df['carbon_intensity_index'].str.lower(), # Ensure lowercase matching
            categories=intensity_categories,
            ordered=True
        ).codes # .codes returns -1 for NaN/missing categories
        # Replace -1 with NaN if preferred, or handle as needed
        df['carbon_intensity_index_encoded'] = df['carbon_intensity_index_encoded'].replace(-1, np.nan)
    return df

def create_cyclical_features(df):
    """Creates cyclical features for time components."""
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        # Optional: Day of year
        # df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365) # Adjust for leap year if critical
        # df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    return df

def create_lag_features(df, columns_to_lag, lag_periods):
    """Creates lag features for specified columns and lag periods."""
    for col in columns_to_lag:
        if col in df.columns:
            for lag in lag_periods:
                df[f'{col}_lag{lag}h'] = df[col].shift(lag)
    return df

def add_price_imputed_column(df):
    """Adds an 'is_price_imputed' column, initialized to False."""
    df['is_price_imputed'] = False
    return df

def main():
    # Phase 1 output, Phase 2 input
    phase1_output_path = 'data/processed/featured_dataset_phase1.csv'
    # Phase 2 output
    phase2_output_path = 'data/processed/featured_dataset_phase2.csv'

    # --- This part is for re-running Phase 1 if needed, or for direct execution of Phase 2 ---
    # Check if phase 1 file exists, if not, run phase 1 steps
    try:
        pd.read_csv(phase1_output_path, index_col='time', nrows=1)
        print(f"Loading data from {phase1_output_path} for Phase 2...")
        df = load_data(phase1_output_path)
    except FileNotFoundError:
        print(f"Warning: {phase1_output_path} not found. Attempting to generate it by running Phase 1 steps first.")
        raw_data_path = 'data/processed/final_dataset_all_periods.csv'
        print(f"Loading raw data from {raw_data_path}...")
        df = load_data(raw_data_path)
        print("Data loaded successfully.")
        print("Original columns:", df.columns.tolist())
        print("Original shape:", df.shape)

        print("\nEncoding carbon_intensity_index (Phase 1 fallback)...")
        df = encode_carbon_intensity_index(df)
        print("\nCreating cyclical features (Phase 1 fallback)...")
        df = create_cyclical_features(df)
        print("Phase 1 fallback steps completed.")
        print("Columns after fallback Phase 1:", df.columns.tolist())
        print("Shape after fallback Phase 1:", df.shape)
        # Save intermediate Phase 1 result if it was generated via fallback
        print(f"Saving intermediate Phase 1 result to {phase1_output_path}")
        df.to_csv(phase1_output_path)

    print("Data loaded successfully for Phase 2.")
    print("Columns before Phase 2:", df.columns.tolist())
    print("Shape before Phase 2:", df.shape)

    # --- Phase 2: Lag Features and Imputation Column ---
    print("\nCreating lag features...")
    columns_to_lag = ['cost_p_per_kwh', 'carbon_intensity_actual', 'carbon_intensity_forecast']
    # Define lag periods (e.g., 1 hour, 2 hours, 24 hours)
    lag_periods = [1, 2, 3, 24, 48, 168] # 1h, 2h, 3h, 1-day, 2-day, 1-week
    df = create_lag_features(df, columns_to_lag, lag_periods)
    print(f"Lag features created for {columns_to_lag} with lags {lag_periods}.")

    print("\nAdding 'is_price_imputed' column...")
    df = add_price_imputed_column(df)
    print("'is_price_imputed' column added.")

    print("\nColumns after Phase 2:", df.columns.tolist())
    print("Shape after Phase 2:", df.shape)
    if not df.empty:
        print("Sample of data with new features (Phase 2):\n", df.tail())

    print(f"\nSaving dataset with Phase 2 features to {phase2_output_path}...")
    df.to_csv(phase2_output_path)
    print("Dataset saved.")

if __name__ == '__main__':
    main() 