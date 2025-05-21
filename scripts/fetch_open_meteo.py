import requests
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta

# Define constants for the script execution
DEFAULT_LATITUDE = 51.5074
DEFAULT_LONGITUDE = 0.1278
DEFAULT_START_DATE = "2022-11-25"
DEFAULT_END_DATE = "2024-09-30"
MAX_DAYS_PER_REQUEST = 90 # Open-Meteo historical forecast API might have limits on range for many variables

def fetch_historical_forecast_data(
    latitude: float, 
    longitude: float, 
    start_date_str: str, 
    end_date_str: str, 
    output_dir: Path,
    variables: str = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,cloud_cover,precipitation,apparent_temperature"
) -> None:
    """
    Fetches hourly historical FORECAST weather data from the Open-Meteo Historical Forecast API 
    for a given location and date range, handling pagination for long periods.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date_str (str): Start date for the data (YYYY-MM-DD).
        end_date_str (str): End date for the data (YYYY-MM-DD).
        output_dir (Path): Directory to save the fetched data.
        variables (str): Comma-separated string of hourly variables to fetch.
    """
    API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"open_meteo_historical_forecasts_{latitude}_{longitude}"
    output_csv_file = output_dir / f"{base_filename}_{start_date_str}_{end_date_str}.csv"
    output_json_file = output_dir / f"{base_filename}_{start_date_str}_{end_date_str}_api_details.json"

    all_data_dfs = []
    all_api_details = []

    current_start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    final_end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    while current_start_date <= final_end_date:
        # Determine the end date for the current chunk
        current_end_date_chunk = current_start_date + timedelta(days=MAX_DAYS_PER_REQUEST -1)
        if current_end_date_chunk > final_end_date:
            current_end_date_chunk = final_end_date
        
        current_start_date_chunk_str = current_start_date.strftime("%Y-%m-%d")
        current_end_date_chunk_str = current_end_date_chunk.strftime("%Y-%m-%d")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": current_start_date_chunk_str,
            "end_date": current_end_date_chunk_str,
            "hourly": variables,
            "timezone": "UTC",
            # forecast_hours=1 can be added if needed, but API behavior for historical range needs checking.
            # Default behavior might already provide the T+1 from each run for the specified hourly variables.
        }

        try:
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url_to_be_called = f"{API_URL}?{param_string}"
            print(f"Fetching data for period: {current_start_date_chunk_str} to {current_end_date_chunk_str}")
            print(f"Attempting to call URL: {full_url_to_be_called}")

            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if 'hourly' not in data or not data['hourly']['time']:
                print(f"Warning: No hourly data returned for {current_start_date_chunk_str} to {current_end_date_chunk_str}. API response: {data}")
                current_start_date = current_end_date_chunk + timedelta(days=1)
                continue

            df_chunk = pd.DataFrame(data['hourly'])
            df_chunk['time'] = pd.to_datetime(df_chunk['time'])
            all_data_dfs.append(df_chunk)

            api_details_chunk = {
                "url_called": response.url,
                "params_used": params,
                "data_snippet_head": df_chunk.head().to_dict(),
                "data_snippet_tail": df_chunk.tail().to_dict(),
                "row_count": len(df_chunk)
            }
            all_api_details.append(api_details_chunk)
            print(f"Successfully fetched {len(df_chunk)} records for {current_start_date_chunk_str} to {current_end_date_chunk_str}.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Open-Meteo for {current_start_date_chunk_str} to {current_end_date_chunk_str}: {e}")
        except KeyError as e:
            print(f"Error processing data from Open-Meteo for {current_start_date_chunk_str} to {current_end_date_chunk_str} - unexpected data structure: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {current_start_date_chunk_str} to {current_end_date_chunk_str}: {e}")
        
        # Move to the next period
        current_start_date = current_end_date_chunk + timedelta(days=1)
        # Small delay to be polite to the API
        # time.sleep(1) # Consider adding if making many rapid calls, though pagination handles this somewhat

    if all_data_dfs:
        final_df = pd.concat(all_data_dfs).drop_duplicates(subset=['time']).sort_values(by='time').reset_index(drop=True)
        final_df.to_csv(output_csv_file, index=False)
        print(f"All data successfully fetched and saved to {output_csv_file} ({len(final_df)} total records)")
        
        with open(output_json_file, 'w') as f:
            json.dump(all_api_details, f, indent=4, default=str)
        print(f"Saved API call details for all chunks to {output_json_file}")
    else:
        print("No data was fetched successfully for the entire period.")

if __name__ == "__main__":
    # Define parameters for data fetching - FULL RANGE
    # Users can modify these dates to fetch smaller chunks if needed for testing or due to API limits.
    start_fetch_date = DEFAULT_START_DATE 
    end_fetch_date = DEFAULT_END_DATE   
    
    # Hourly variables we want for the demand predictor (and potentially price predictor)
    # `apparent_temperature` is also included here from the forecast API for comparison/reference if needed.
    hourly_variables_to_fetch = (
        "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,snowfall,"
        "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
        "shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,"
        "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
        "et0_fao_evapotranspiration,vapor_pressure_deficit,soil_temperature_0cm,soil_moisture_0_to_1cm"
        # Add/remove variables as needed. Check Open-Meteo docs for full list and availability.
    )
    # A more focused list based on our immediate needs:
    focused_hourly_variables = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,cloud_cover,precipitation,apparent_temperature"

    SCRIPT_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "raw"

    print(f"--- Starting Historical FORECAST Data Fetching --- ")
    print(f"Target Period: {start_fetch_date} to {end_fetch_date}")
    print(f"Location: Lat={DEFAULT_LATITUDE}, Lon={DEFAULT_LONGITUDE}")
    print(f"Saving to: {OUTPUT_DIR}")
    print(f"Hourly Variables: {focused_hourly_variables}")

    fetch_historical_forecast_data(
        DEFAULT_LATITUDE, 
        DEFAULT_LONGITUDE, 
        start_fetch_date, 
        end_fetch_date, 
        OUTPUT_DIR,
        variables=focused_hourly_variables
    )
    print("--- Historical FORECAST Data Fetching Script Finished ---") 