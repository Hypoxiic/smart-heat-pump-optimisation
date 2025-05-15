import requests
import pandas as pd
from pathlib import Path
import json

def fetch_weather_data(latitude: float, longitude: float, start_date: str, end_date: str, output_dir: Path) -> None:
    """
    Fetches hourly weather data from the Open-Meteo API for a given location and date range.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date for the data (YYYY-MM-DD).
        end_date (str): End date for the data (YYYY-MM-DD).
        output_dir (Path): Directory to save the fetched data.
    """
    API_URL = "https://archive-api.open-meteo.com/v1/archive" # Corrected API URL for historical data
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,apparent_temperature",
        "timezone": "UTC", # Changed to UTC to match example
    }

    try:
        # Manually prepare the full URL for printing, to see what would be called
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url_to_be_called = f"{API_URL}?{param_string}"
        print(f"Attempting to call URL: {full_url_to_be_called}")

        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Convert to pandas DataFrame
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        output_file = output_dir / f"open_meteo_{latitude}_{longitude}_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        print(f"Successfully fetched and saved weather data to {output_file}")

        # Save API call details
        api_details = {
            "url": response.url, # Save the exact URL called
            "params": params,
            "data_snippet": df.head().to_dict()
        }
        details_file = output_dir / f"open_meteo_{latitude}_{longitude}_{start_date}_{end_date}_api_details.json"
        with open(details_file, 'w') as f:
            json.dump(api_details, f, indent=4, default=str)
        print(f"Saved API call details to {details_file}")


    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Open-Meteo: {e}")
    except KeyError as e:
        print(f"Error processing data from Open-Meteo - unexpected data structure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define parameters for data fetching
    # London, UK coordinates
    LATITUDE = 51.5074
    LONGITUDE = 0.1278 
    # Gap Period
    START_DATE = "2023-12-12" 
    END_DATE = "2024-04-02"   
    
    # Define the output directory relative to this script's location
    # Assuming this script is in 'scripts/' and data should go to 'data/raw/'
    # So, ../data/raw/
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "raw"

    fetch_weather_data(LATITUDE, LONGITUDE, START_DATE, END_DATE, OUTPUT_DIR) 