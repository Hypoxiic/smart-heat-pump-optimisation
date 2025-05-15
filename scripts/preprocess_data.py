import pandas as pd
from pathlib import Path
import numpy as np
import datetime # Added for datetime.timezone.utc

# Define constants for file names and paths
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "data" / "raw"
PROCESSED_DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"

# Expected raw filenames for Gap Period (2023-12-12 to 2024-04-02)
WEATHER_RAW_FILENAME = "open_meteo_51.5074_0.1278_2023-12-12_2024-04-02.csv"
CARBON_RAW_FILENAME = "uk_carbon_intensity_2023-12-12_2024-04-02.csv"
# No specific Agile tariff for this full period, so this file won't be found by preprocess script.
ELECTRICITY_RAW_FILENAME = "electricity_costs_NO_AGILE_TARIFF_FOR_GAP_2023-12-12_2024-04-02.csv"

# Add a suffix for period-specific merged files
# This will be set dynamically in the main block based on the raw file dates for now
MERGED_DATA_SUFFIX = ""

def preprocess_weather_data(raw_file_path: Path, processed_file_path: Path) -> pd.DataFrame | None:
    """Loads, preprocesses, and saves weather data."""
    try:
        df = pd.read_csv(raw_file_path)
        df['time'] = pd.to_datetime(df['time'])

        # Data from Open-Meteo was fetched with timezone="UTC". 
        # If pd.to_datetime results in naive, localize to UTC.
        # If it correctly infers tz from string (if present), then convert to UTC if it's not already.
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        elif df['time'].dt.tz != datetime.timezone.utc: # If it was parsed with a different TZ
            df['time'] = df['time'].dt.tz_convert('UTC')
            
        # Check if any NaT values were introduced by tz_localize (should not happen with 'UTC')
        if df['time'].isnull().any():
            print("Warning: NaT values introduced in weather data time column during timezone localization.")

        df = df.set_index('time')
        # Data is already hourly, select relevant columns with correct names
        df = df[['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'apparent_temperature']]
        df.columns = ['temperature', 'humidity', 'windspeed', 'apparent_temperature']
        
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_file_path)
        print(f"Processed weather data saved to {processed_file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Raw weather data file not found at {raw_file_path}")
    except Exception as e:
        print(f"Error processing weather data: {e}")
    return None

def preprocess_carbon_intensity_data(raw_file_path: Path, processed_file_path: Path) -> pd.DataFrame | None:
    """Loads, preprocesses, resamples to hourly, and saves carbon intensity data."""
    try:
        df = pd.read_csv(raw_file_path)
        df['time'] = pd.to_datetime(df['time']) # Should be UTC as per fetch script
        
        if df['time'].dt.tz is None: # Ensure UTC
             df['time'] = df['time'].dt.tz_localize('UTC')
        # Corrected timezone check
        elif df['time'].dt.tz != datetime.timezone.utc:
             df['time'] = df['time'].dt.tz_convert('UTC')

        df = df.set_index('time')
        
        # Resample from 30-min to hourly
        resampling_aggregations = {
            'carbon_intensity_forecast': 'mean',
            'carbon_intensity_actual': 'mean',
            'carbon_intensity_index': 'first' # Or 'last', or custom mode function
        }
        df_hourly = df.resample('h').agg(resampling_aggregations)
        
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_hourly.to_csv(processed_file_path)
        print(f"Processed and hourly-resampled carbon intensity data saved to {processed_file_path}")
        return df_hourly
    except FileNotFoundError:
        print(f"Error: Raw carbon intensity data file not found at {raw_file_path}")
    except Exception as e:
        print(f"Error processing carbon intensity data: {e}")
    return None

def preprocess_electricity_costs_data(raw_file_path: Path, processed_file_path: Path) -> pd.DataFrame | None:
    """Loads, preprocesses, resamples to hourly, and saves electricity cost data."""
    try:
        df = pd.read_csv(raw_file_path)
        
        # Logic from fetch_electricity_costs.py to handle column names and create 'time'
        # The Octopus API script already saves the CSV with 'time' and 'cost_p_per_kwh' columns
        # and 'time' is already in UTC. So, we simplify this part.
        if 'time' in df.columns and 'cost_p_per_kwh' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
            # Ensure cost_p_per_kwh is numeric, coerce errors to NaN
            df['cost_p_per_kwh'] = pd.to_numeric(df['cost_p_per_kwh'], errors='coerce')
            # Drop rows where cost could not be parsed to a number
            df.dropna(subset=['cost_p_per_kwh'], inplace=True)
        elif 'settlement_date' in df.columns and 'settlement_period' in df.columns: # Old format, keep for backward compatibility if needed
            # Timestamps are UTC. Settlement period is 1-48 for HH data.
            df['time'] = pd.to_datetime(df['settlement_date'], utc=True) + \
                         pd.to_timedelta(df['settlement_period'].astype(int) * 30 - 30, unit='m')
            df = df.rename(columns={'net_effective_rate_p_kwh': 'cost_p_per_kwh'})
            df = df[['time', 'cost_p_per_kwh']]
        elif 'DATE' in df.columns and 'PERIOD' in df.columns and 'PRICE' in df.columns: # A generic alternative
            df['time'] = pd.to_datetime(df['DATE'], utc=True) + \
                         pd.to_timedelta(df['PERIOD'].astype(int) * 30 - 30, unit='m')
            df = df.rename(columns={'PRICE': 'cost_p_per_kwh'})
            df = df[['time', 'cost_p_per_kwh']]
        else:
            raise ValueError("Electricity cost CSV columns not in an expected format. "
                             "Requires ('time' and 'cost_p_per_kwh') or other known legacy formats.")

        if df['time'].dt.tz is None: # Ensure UTC
             df['time'] = df['time'].dt.tz_localize('UTC')
        elif df['time'].dt.tz != datetime.timezone.utc:
             df['time'] = df['time'].dt.tz_convert('UTC')
             
        df = df.set_index('time')
        
        # Resample from 30-min to hourly
        df_hourly = df.resample('h').agg({'cost_p_per_kwh': 'mean'})
        
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_hourly.to_csv(processed_file_path)
        print(f"Processed and hourly-resampled electricity cost data saved to {processed_file_path}")
        return df_hourly
    except FileNotFoundError:
        print(f"Error: Raw electricity cost data file not found at {raw_file_path}. "
              "Please ensure the file exists and the filename matches in preprocess_data.py.") # Updated error message
    except Exception as e:
        print(f"Error processing electricity cost data: {e}")
    return None

def merge_and_save_data(weather_df, carbon_df, electricity_df, merged_file_path: Path):
    """Merges the processed dataframes and saves the result."""
    dfs_to_merge = []
    if weather_df is not None:
        dfs_to_merge.append(weather_df)
    if carbon_df is not None:
        dfs_to_merge.append(carbon_df)
    if electricity_df is not None:
        dfs_to_merge.append(electricity_df)

    if not dfs_to_merge:
        print("No dataframes to merge.")
        return

    # Merge DataFrames
    # Start with the first available df, then merge others to it.
    final_df = dfs_to_merge[0]
    for df_to_add in dfs_to_merge[1:]:
        final_df = pd.merge(final_df, df_to_add, on='time', how='outer')
    
    final_df = final_df.sort_index()
    
    # Handle missing values - forward fill
    final_df = final_df.ffill()
    
    # Optional: backward fill any remaining NaNs at the beginning
    final_df = final_df.bfill()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(merged_file_path)
    print(f"Final merged and preprocessed data saved to {merged_file_path}")
    print("\\n--- Merged Data Sample ---")
    print(final_df.head())
    print("\\n--- Merged Data Info ---")
    final_df.info()

def concatenate_period_data(processed_dir: Path, output_filename: str):
    """Concatenates merged data from different periods into a single dataset."""
    period1_file = processed_dir / "merged_project_data_2022-11-25_2023-12-11.csv"
    gap_period_file = processed_dir / "merged_project_data_2023-12-12_2024-04-02.csv"
    period2_file = processed_dir / "merged_project_data_2024-04-03_2024-09-30.csv"

    df_list = []
    files_loaded = []

    for f_path in [period1_file, gap_period_file, period2_file]:
        if f_path.exists():
            print(f"Loading {f_path.name}...")
            try:
                df = pd.read_csv(f_path, index_col='time', parse_dates=True)
                df_list.append(df)
                files_loaded.append(f_path.name)
            except Exception as e:
                print(f"Error loading {f_path.name}: {e}. Skipping this file.")
        else:
            print(f"Warning: {f_path.name} not found. Skipping.")

    if not df_list:
        print("No dataframes to concatenate. Exiting concatenation.")
        return

    print(f"\nConcatenating data from: {', '.join(files_loaded)}")
    # Concatenate. Pandas will align columns and fill missing ones with NaN.
    final_df = pd.concat(df_list, axis=0, sort=False) # sort=False initially, will sort by index later
    
    # Ensure timezone consistency (all should be UTC from previous steps)
    # If index is already DatetimeIndex and tz-aware, this should be fine.
    # If parse_dates made it naive, localize to UTC.
    if final_df.index.tz is None:
        try:
            final_df = final_df.tz_localize('UTC')
        except TypeError as te:
            # This can happen if NaT or non-datetime values are in index
            print(f"Warning: Could not localize index to UTC due to TypeError: {te}. Check for NaT/non-datetime in index.")
            # Attempt to clean NaT from index before trying again or proceeding
            final_df = final_df[final_df.index.notna()]
            if final_df.index.tz is None and not final_df.empty:
                final_df = final_df.tz_localize('UTC')
    elif final_df.index.tz != datetime.timezone.utc:
        final_df = final_df.tz_convert('UTC')
        
    final_df = final_df.sort_index()
    
    # Handle potential duplicate indices from overlaps (e.g. if period boundaries were inclusive on both sides)
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    
    # Drop rows where the index is NaT (if any slipped through)
    final_df = final_df[final_df.index.notna()]

    # Separate cost column, fill other columns, then reattach cost column to preserve its NaNs for the gap
    if 'cost_p_per_kwh' in final_df.columns:
        cost_column = final_df[['cost_p_per_kwh']].copy()
        other_columns_df = final_df.drop(columns=['cost_p_per_kwh'])
        other_columns_df = other_columns_df.ffill().bfill() # Fill NaNs in weather/carbon data
        final_df_recombined = pd.concat([other_columns_df, cost_column], axis=1)
        # Ensure column order is maintained or sensible if cost_column was not originally last
        # Reindex to original columns if necessary, though concat should handle order if names are same.
    else: # Should not happen if Period 1 and 2 files are correct
        print("Warning: 'cost_p_per_kwh' column not found in concatenated data. Skipping selective fill.")
        final_df_recombined = final_df.ffill().bfill() # Fallback to filling all if cost col is missing

    output_path = processed_dir / output_filename
    final_df_recombined.to_csv(output_path)
    print(f"\nFinal concatenated dataset saved to {output_path}")
    print("\n--- Final Concatenated Data Sample ---")
    print(final_df_recombined.head())
    print(final_df_recombined.tail())
    print("\n--- Final Concatenated Data Info ---")
    final_df_recombined.info()

if __name__ == "__main__":
    print("Starting data preprocessing...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine period from one of the raw filenames (e.g., weather)
    # This is a bit heuristic; ideally, pass dates explicitly or use a config
    # For this iteration, we assume filenames are updated before running.
    period_str = "_" + WEATHER_RAW_FILENAME.split('_')[-2] + "_" + WEATHER_RAW_FILENAME.split('_')[-1].replace(".csv","")
    
    # Define file paths
    current_weather_raw_filename = f"open_meteo_51.5074_0.1278{period_str}.csv"
    current_carbon_raw_filename = f"uk_carbon_intensity{period_str}.csv"
    # For electricity, the product code is also in the name
    # We need to extract product code and dates separately or rely on exact match.
    # Let's assume the global constants are updated before each run for now.
    # This part will need to be more robust if we fully automate multi-period processing.

    weather_raw_path = RAW_DATA_DIR / WEATHER_RAW_FILENAME
    weather_processed_path = PROCESSED_DATA_DIR / f"weather_hourly{period_str}.csv"
    
    carbon_raw_path = RAW_DATA_DIR / CARBON_RAW_FILENAME
    carbon_processed_path = PROCESSED_DATA_DIR / f"carbon_intensity_hourly{period_str}.csv"
    
    electricity_raw_path = RAW_DATA_DIR / ELECTRICITY_RAW_FILENAME
    electricity_processed_path = PROCESSED_DATA_DIR / f"electricity_costs_hourly{period_str}.csv"
    
    merged_data_path = PROCESSED_DATA_DIR / f"merged_project_data{period_str}.csv"

    # Process each dataset
    df_weather = preprocess_weather_data(weather_raw_path, weather_processed_path)
    df_carbon = preprocess_carbon_intensity_data(carbon_raw_path, carbon_processed_path)
    df_electricity = preprocess_electricity_costs_data(electricity_raw_path, electricity_processed_path)

    # Merge and save if at least weather and carbon are available
    if df_weather is not None and df_carbon is not None:
        print("\\nMerging datasets for the current period...")
        merge_and_save_data(df_weather, df_carbon, df_electricity, merged_data_path) # df_electricity can be None
    else:
        print("\\nSkipping merge for current period due to missing processed data for weather or carbon intensity.")
        print(f"Weather data processed: {df_weather is not None}")
        print(f"Carbon intensity data processed: {df_carbon is not None}")
        print(f"Electricity costs data processed: {df_electricity is not None}")

    print("\\nData preprocessing for current period finished.") 
    
    # --- Concatenate all period data --- 
    # This section assumes that the individual period processing for ALL periods has been done in previous runs
    # or that this script is run sequentially for each period and then this final step is desired.
    # For a fully automated multi-period pipeline, the main logic would need to loop through period definitions.
    print("\\nStarting concatenation of all period datasets...")
    concatenate_period_data(PROCESSED_DATA_DIR, "final_dataset_all_periods.csv")
    print("\\nConcatenation finished.") 