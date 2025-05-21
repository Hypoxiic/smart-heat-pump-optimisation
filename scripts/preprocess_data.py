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

# Historical forecast data (covers the full project range)
HISTORICAL_FORECAST_RAW_FILENAME = "open_meteo_historical_forecasts_51.5074_0.1278_2022-11-25_2024-09-30.csv"

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
        target_columns_actual_weather = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'apparent_temperature']
        
        # Filter for columns that actually exist in the dataframe to avoid KeyErrors
        available_columns = [col for col in target_columns_actual_weather if col in df.columns]
        missing_columns = [col for col in target_columns_actual_weather if col not in df.columns]
        if missing_columns:
            print(f"Warning: The following expected weather columns were not found in {raw_file_path.name}: {missing_columns}")

        df = df[available_columns]
        
        # Rename available columns to standard names
        rename_map = {
            'temperature_2m': 'temperature',
            'relative_humidity_2m': 'humidity',
            'wind_speed_10m': 'windspeed',
            'apparent_temperature': 'apparent_temperature'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in available_columns})
        
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

def preprocess_historical_forecast_data(raw_file_path: Path, processed_file_path: Path) -> pd.DataFrame | None:
    """Loads, preprocesses, and saves historical FORECAST weather data."""
    try:
        df = pd.read_csv(raw_file_path)
        df['time'] = pd.to_datetime(df['time'])

        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        elif df['time'].dt.tz != datetime.timezone.utc:
            df['time'] = df['time'].dt.tz_convert('UTC')
            
        if df['time'].isnull().any():
            print("Warning: NaT values introduced in historical forecast data time column during timezone localization.")

        df = df.set_index('time')
        
        # Select relevant forecast columns and rename them
        # Original names from fetch script: "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,cloud_cover,precipitation,apparent_temperature"
        forecast_columns = {
            'temperature_2m': 'temperature_forecast',
            'relative_humidity_2m': 'humidity_forecast',
            'wind_speed_10m': 'windspeed_forecast',
            'shortwave_radiation': 'sw_radiation_forecast',
            'cloud_cover': 'cloud_cover_forecast',
            'precipitation': 'precipitation_forecast',
            'apparent_temperature': 'apparent_temperature_forecast' # Keep this for reference/use
        }
        # Filter for columns that actually exist in the dataframe to avoid KeyErrors
        cols_to_select = {k: v for k, v in forecast_columns.items() if k in df.columns}
        df = df[list(cols_to_select.keys())]
        df = df.rename(columns=cols_to_select)
        
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_file_path)
        print(f"Processed historical forecast data saved to {processed_file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Raw historical forecast data file not found at {raw_file_path}")
    except Exception as e:
        print(f"Error processing historical forecast data: {e}")
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

def concatenate_period_data(processed_dir: Path, output_filename: str) -> pd.DataFrame | None:
    """
    Concatenates merged data from different periods into a single dataset.
    Returns the concatenated dataframe.
    """
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
    return final_df_recombined

if __name__ == "__main__":
    print("Starting data preprocessing...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine period from one of the raw filenames (e.g., weather)
    # This defines the three main periods for processing historical ACTUALS
    RAW_DATA_FILES_PERIODS = [
        {
            "weather": "open_meteo_51.5074_0.1278_2022-11-25_2023-12-11.csv",
            "carbon": "uk_carbon_intensity_2022-11-25_2023-12-11.csv",
            "electricity": "electricity_costs_octopus_AGILE-FLEX-22-11-25_C_2022-11-25_2023-12-11.csv",
            "suffix": "_2022-11-25_2023-12-11"
        },
        {
            "weather": WEATHER_RAW_FILENAME, # Uses constant for Gap period
            "carbon": CARBON_RAW_FILENAME,   # Uses constant for Gap period
            "electricity": None, # No electricity file for the full Gap period
            "suffix": "_2023-12-12_2024-04-02"
        },
        {
            "weather": "open_meteo_51.5074_0.1278_2024-04-03_2024-09-30.csv",
            "carbon": "uk_carbon_intensity_2024-04-03_2024-09-30.csv",
            "electricity": "electricity_costs_octopus_AGILE-24-04-03_C_2024-04-03_2024-09-30.csv",
            "suffix": "_2024-04-03_2024-09-30"
        }
    ]

    for period_files in RAW_DATA_FILES_PERIODS:
        print(f"\\n--- Processing Period: {period_files['suffix'].strip('_')} ---")
        
        # Preprocess Weather Data (Actuals)
        weather_df = None
        if period_files["weather"]:
            raw_weather_path = RAW_DATA_DIR / period_files["weather"]
            processed_weather_path = PROCESSED_DATA_DIR / f"weather_hourly{period_files['suffix']}.csv"
            weather_df = preprocess_weather_data(raw_weather_path, processed_weather_path)

        # Preprocess Carbon Intensity Data
        carbon_df = None
        if period_files["carbon"]:
            raw_carbon_path = RAW_DATA_DIR / period_files["carbon"]
            processed_carbon_path = PROCESSED_DATA_DIR / f"carbon_intensity_hourly{period_files['suffix']}.csv"
            carbon_df = preprocess_carbon_intensity_data(raw_carbon_path, processed_carbon_path)

        # Preprocess Electricity Costs Data
        electricity_df = None
        if period_files["electricity"]: # Check if electricity file key is present and not None
            raw_electricity_path = RAW_DATA_DIR / period_files["electricity"]
            processed_electricity_path = PROCESSED_DATA_DIR / f"electricity_costs_hourly{period_files['suffix']}.csv"
            electricity_df = preprocess_electricity_costs_data(raw_electricity_path, processed_electricity_path)
        else:
            print(f"No electricity cost file specified for period {period_files['suffix']}. Skipping electricity preprocessing.")

        # Merge and Save Data for the current period
        if weather_df is not None or carbon_df is not None or electricity_df is not None:
            merged_data_path = PROCESSED_DATA_DIR / f"merged_project_data{period_files['suffix']}.csv"
            merge_and_save_data(weather_df, carbon_df, electricity_df, merged_data_path)
        else:
            print(f"Skipping merge for period {period_files['suffix']} as no data was processed.")

    # Concatenate all period data
    final_concatenated_df = concatenate_period_data(PROCESSED_DATA_DIR, "final_dataset_all_periods.csv")

    # --- Integrate Historical Forecast Data ---
    if final_concatenated_df is not None:
        print("\\n--- Processing and Integrating Historical Forecast Data ---")

        print("\\nDEBUG: Info for final_concatenated_df (from concatenate_period_data) BEFORE MERGE:")
        final_concatenated_df.info()
        print(f"DEBUG: Columns in final_concatenated_df: {final_concatenated_df.columns.tolist()}")
        print(f"DEBUG: Index name of final_concatenated_df: {final_concatenated_df.index.name}")

        raw_historical_forecast_path = RAW_DATA_DIR / HISTORICAL_FORECAST_RAW_FILENAME
        processed_historical_forecast_path = PROCESSED_DATA_DIR / f"historical_forecasts_processed{RAW_DATA_FILES_PERIODS[0]['suffix']}_{RAW_DATA_FILES_PERIODS[-1]['suffix']}.csv" # A bit clunky filename gen, but ok for now
        
        historical_forecast_df = preprocess_historical_forecast_data(raw_historical_forecast_path, processed_historical_forecast_path)

        if historical_forecast_df is not None:
            print("Merging concatenated data with historical forecast data...")

            print("\\nDEBUG: Info for historical_forecast_df BEFORE MERGE:")
            historical_forecast_df.info()
            print(f"DEBUG: Columns in historical_forecast_df: {historical_forecast_df.columns.tolist()}")
            print(f"DEBUG: Index name of historical_forecast_df: {historical_forecast_df.index.name}")
            
            # Ensure both DataFrames have UTC timezone-aware DatetimeIndex
            if final_concatenated_df.index.tz is None:
                final_concatenated_df = final_concatenated_df.tz_localize('UTC')
            elif final_concatenated_df.index.tz != datetime.timezone.utc:
                final_concatenated_df = final_concatenated_df.tz_convert('UTC')

            if historical_forecast_df.index.tz is None:
                historical_forecast_df = historical_forecast_df.tz_localize('UTC')
            elif historical_forecast_df.index.tz != datetime.timezone.utc:
                historical_forecast_df = historical_forecast_df.tz_convert('UTC')

            # Using left merge to keep all rows from final_concatenated_df and add forecast data where available
            final_df_with_forecasts = pd.merge(final_concatenated_df, historical_forecast_df, on='time', how='left')
            
            print("\\nDEBUG: Info for final_df_with_forecasts AFTER MERGE (before ffill/bfill):")
            final_df_with_forecasts.info()
            print(f"DEBUG: Columns in final_df_with_forecasts after merge: {final_df_with_forecasts.columns.tolist()}")

            final_df_with_forecasts = final_df_with_forecasts.sort_index()
            
            # Forward-fill and backward-fill the newly added forecast columns ONLY.
            # Do not re-fill 'cost_p_per_kwh' if it has intentional NaNs from the gap period.
            
            # Identify the actual names of the forecast columns as they appear in historical_forecast_df
            # These are the columns we want to ffill/bfill after they've been merged.
            forecast_column_names_to_fill = [col for col in historical_forecast_df.columns if col in final_df_with_forecasts.columns]

            if forecast_column_names_to_fill:
                # Create a temporary DataFrame containing ONLY these forecast columns from the merged data
                temp_forecast_part = final_df_with_forecasts[forecast_column_names_to_fill].copy()
                
                # Fill NaNs ONLY in this temporary DataFrame of forecast columns
                temp_forecast_part = temp_forecast_part.ffill().bfill()
                
                # Update the main merged DataFrame (final_df_with_forecasts) with these filled forecast columns
                # This replaces the original forecast columns (which might have NaNs from the merge)
                # with their filled versions, leaving all other columns (like cost_p_per_kwh) untouched.
                for col in forecast_column_names_to_fill:
                    final_df_with_forecasts[col] = temp_forecast_part[col]
            
            output_path_with_forecasts = PROCESSED_DATA_DIR / "final_dataset_with_forecasts.csv"
            final_df_with_forecasts.to_csv(output_path_with_forecasts)
            
            print(f"\\nFinal dataset with historical forecasts saved to {output_path_with_forecasts}")
            print("\\n--- Final Data with Forecasts Sample (Head) ---")
            print(final_df_with_forecasts.head())
            print("\\n--- Final Data with Forecasts Sample (Tail) ---")
            print(final_df_with_forecasts.tail())
            print("\\n--- Final Data with Forecasts Info ---")
            final_df_with_forecasts.info()
        else:
            print("Historical forecast data could not be processed. Skipping merge.")
    else:
        print("Final concatenated data is not available. Skipping forecast integration.")

    print("\\nPreprocessing script finished.") 