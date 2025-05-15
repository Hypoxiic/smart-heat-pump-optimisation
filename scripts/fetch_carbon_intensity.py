import requests
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta

def fetch_carbon_intensity_data(start_date_str: str, end_date_str: str, output_dir: Path) -> None:
    """
    Fetches carbon intensity data from the UK Carbon Intensity API for a given date range.
    The API provides data in 30-minute intervals. This function fetches data day by day.

    Args:
        start_date_str (str): Start date for the data (YYYY-MM-DD).
        end_date_str (str): End date for the data (YYYY-MM-DD).
        output_dir (Path): Directory to save the fetched data.
    """
    API_BASE_URL = "https://api.carbonintensity.org.uk/intensity"
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    all_intensity_data = []
    current_date = start_date
    request_urls = []

    print(f"Fetching carbon intensity data from {start_date_str} to {end_date_str}...")

    try:
        while current_date <= end_date:
            # API format for date is /YYYY-MM-DD
            # The API provides data for the 24 hours FOLLOWING the specified ISO8601 datetime.
            # To get data FOR a specific day, we can query for that day.
            # For ranges, it's often /from/{from}/to/{to} but here we'll do day by day for simplicity and to match hourly weather.
            # The API endpoint for a specific date seems to be /date/{YYYY-MM-DD}
            # However, the most granular for a period is /intensity/{from}/{to}
            # Let's fetch it for the full day to ensure we cover all half-hour slots.
            
            # The API returns data in 30-min intervals. We will query for the whole day.
            # For a specific day, the API expects /intensity/date/{YYYY-MM-DD}
            # For a range, it is /intensity/{from_iso_timestamp}/to/{iso_timestamp}
            # We will fetch for the entire specified period in one go if possible, or iterate.
            # The API documentation states: "Get intensity data between the {from} and {to} ISO8601 date times.
            # Returns a maximum of 48h of data."
            # So we must iterate in chunks if the period is longer than 2 days.

            # Let's try fetching the entire range first, then adapt if it's too long.
            # The API docs show max 48h data for /from/{from}/to/{to} endpoint.
            # We will iterate day by day to be safe and collect all 30-min slots.
            
            from_datetime_iso = current_date.isoformat() + "Z" # API expects ISO8601 format, 'Z' denotes UTC
            # Fetch for the whole day, so end is start of next day
            to_datetime_iso = (current_date + timedelta(days=1) - timedelta(minutes=30)).isoformat() + "Z" 
            
            # Corrected approach: use /intensity/date/{YYYY-MM-DD} and iterate
            # Or better, use /from/to and handle 48h limit.
            # For simplicity, let's iterate day by day using the /date endpoint as it returns all slots for that day.
            
            api_url_for_day = f"{API_BASE_URL}/date/{current_date.strftime('%Y-%m-%d')}"
            request_urls.append(api_url_for_day)

            headers = {
                'Accept': 'application/json'
            }
            response = requests.get(api_url_for_day, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and isinstance(data['data'], list):
                for entry in data['data']: # entry is a dict for a 30-min slot
                    # entry has 'from', 'to', 'intensity' (which is another dict)
                    intensity_metrics = entry.get('intensity', {})
                    record = {
                        'from_time': entry.get('from'),
                        'to_time': entry.get('to'),
                        'carbon_intensity_forecast': intensity_metrics.get('forecast'),
                        'carbon_intensity_actual': intensity_metrics.get('actual'),
                        'carbon_intensity_index': intensity_metrics.get('index')
                    }
                    all_intensity_data.append(record)
            else:
                print(f"Warning: No 'data' list found for {current_date.strftime('%Y-%m-%d')}. Response: {data}")

            current_date += timedelta(days=1)

        if not all_intensity_data:
            print("No carbon intensity data fetched. Please check the date range and API status.")
            return

        # Create a DataFrame
        df = pd.DataFrame(all_intensity_data)
        
        # Timestamps are now 'from_time'
        df['time'] = pd.to_datetime(df['from_time'])
        # We can drop 'from_time' and 'to_time' if 'time' (derived from 'from_time') is sufficient
        df = df.drop(columns=['from_time', 'to_time'], errors='ignore')
        
        # Ensure column order and select relevant columns if needed, though all are relevant here.
        # df = df[['time', 'carbon_intensity_forecast', 'carbon_intensity_actual', 'carbon_intensity_index']]
        df = df.sort_values(by='time').reset_index(drop=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"uk_carbon_intensity_{start_date_str}_{end_date_str}.csv"
        df.to_csv(output_file, index=False)
        print(f"Successfully fetched and saved carbon intensity data to {output_file}")

        api_details = {
            "urls_called": request_urls, # Could be multiple if spanning many days
            "date_range": {"start": start_date_str, "end": end_date_str},
            "data_snippet": df.head().to_dict()
        }
        details_file = output_dir / f"uk_carbon_intensity_{start_date_str}_{end_date_str}_api_details.json"
        with open(details_file, 'w') as f:
            json.dump(api_details, f, indent=4, default=str)
        print(f"Saved API call details to {details_file}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from UK Carbon Intensity API: {e}")
    except KeyError as e:
        print(f"Error processing data from UK Carbon Intensity API - unexpected data structure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Gap Period
    START_DATE = "2023-12-12" 
    END_DATE = "2024-04-02"   
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "raw"

    fetch_carbon_intensity_data(START_DATE, END_DATE, OUTPUT_DIR) 