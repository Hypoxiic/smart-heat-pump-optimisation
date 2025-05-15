import requests
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
# import io # No longer needed for CSV parsing from string

# Agile product and tariff codes (example for London region C)
# These can be discovered via the API:
# 1. Products: https://api.octopus.energy/v1/products/?is_variable=true&is_green=true&brand=OCTOPUS_ENERGY
# 2. Tariffs for a product: https://api.octopus.energy/v1/products/AGILE-FLEX-22-11-25/ (example)
#    Look for E-1R-{PRODUCT_CODE}-{REGION_LETTER}
# For this example, we'll use a known recent Agile tariff.
# It's good practice to allow these to be configurable or discovered dynamically.
DEFAULT_PRODUCT_CODE = "AGILE-24-04-03" # Period 2
DEFAULT_TARIFF_CODE_PREFIX = "E-1R-" # Prefix for electricity single rate
DEFAULT_REGION_LETTER = "C" # London DNO region for Octopus

def get_octopus_tariff_code(product_code: str, region_letter: str) -> str:
    """Constructs the full tariff code for Octopus API."""
    # Example: E-1R-AGILE-FLEX-22-11-25-C
    return f"{DEFAULT_TARIFF_CODE_PREFIX}{product_code}-{region_letter}"

def fetch_electricity_costs(
    product_code: str,
    tariff_code: str,
    start_date_str: str,
    end_date_str: str,
    output_dir: Path
) -> None:
    """
    Fetches half-hourly electricity cost data from the Octopus Energy API
    for a specified Agile tariff product and region.
    """
    # Ensure end_date_obj includes the whole day by setting time to end of day
    # The API is exclusive for the end date, so to get data for 2023-01-07,
    # period_to should be 2023-01-08T00:00:00Z or 2023-01-07T23:59:59Z
    start_date_obj = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)

    # Format dates for API (ISO 8601 UTC)
    period_from = start_date_obj.strftime("%Y-%m-%dT00:00:00Z")
    # To include the entire end_date, period_to should be the start of the next day
    # or end of the specified day. Octopus API uses exclusive end date, so next day start.
    period_to = end_date_obj.strftime("%Y-%m-%dT00:00:00Z")

    base_url = f"https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
    
    params = {
        "period_from": period_from,
        "period_to": period_to,
        "page_size": 1500  # Max results per page (covers ~31 days of HH data)
    }

    headers = {
        'Accept': 'application/json'
    }

    print(f"Fetching Octopus Agile electricity costs for tariff {tariff_code} from {start_date_str} to {end_date_str}...")
    
    all_results = []
    current_url = base_url

    try:
        while current_url:
            response = requests.get(current_url, params=params if current_url == base_url else {}, headers=headers)
            response.raise_for_status()
            data = response.json()
            full_request_url = response.url # For logging

            all_results.extend(data.get("results", []))
            
            current_url = data.get("next") # For pagination
            if current_url:
                print(f"Fetching next page: {current_url}")
            # Params are only needed for the first request when using pagination link
            params = {}

        if not all_results:
            print("No data returned from Octopus API.")
            # Optionally, call inform_manual_download or raise an error
            inform_api_issue(output_dir, product_code, tariff_code, start_date_str, end_date_str, "No data returned")
            return

        df = pd.DataFrame(all_results)

        # API returns prices in pence, including VAT, in 'value_inc_vat'
        # 'valid_from' is the start of the half-hour slot, UTC
        df['time'] = pd.to_datetime(df['valid_from'], utc=True)
        df = df.rename(columns={'value_inc_vat': 'cost_p_per_kwh'})
        
        # Ensure 'cost_p_per_kwh' is numeric
        df['cost_p_per_kwh'] = pd.to_numeric(df['cost_p_per_kwh'], errors='coerce')
        df.dropna(subset=['cost_p_per_kwh'], inplace=True)

        df = df[['time', 'cost_p_per_kwh']]
        df = df.sort_values(by='time').reset_index(drop=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        # Adjust filename to reflect new source and parameters
        output_file_name = f"electricity_costs_octopus_{product_code}_{tariff_code.split('-')[-1]}_{start_date_str}_{end_date_str}.csv"
        output_file = output_dir / output_file_name
        df.to_csv(output_file, index=False)
        print(f"Successfully fetched and saved electricity cost data to {output_file}")

        api_details = {
            "source_api": "Octopus Energy",
            "product_code": product_code,
            "tariff_code": tariff_code,
            "requested_period_from": period_from,
            "requested_period_to": (datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(minutes=30)).strftime("%Y-%m-%dT23:30:00Z"), # show actual end of data
            "urls_called": [base_url], # Could extend if paginated, but full_request_url is already the final one
            "final_url_with_params": full_request_url,
            "data_snippet": df.head().to_dict() if not df.empty else "empty_dataframe"
        }
        details_file_name = f"electricity_costs_octopus_{product_code}_{tariff_code.split('-')[-1]}_{start_date_str}_{end_date_str}_api_details.json"
        details_file = output_dir / details_file_name
        with open(details_file, 'w') as f:
            json.dump(api_details, f, indent=4, default=str) # Add default=str for datetime
        print(f"Saved API call details to {details_file}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Octopus Energy API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            try:
                print(f"Response JSON: {e.response.json()}") # Octopus usually returns JSON errors
            except json.JSONDecodeError:
                print(f"Response text: {e.response.text[:500]}")
        inform_api_issue(output_dir, product_code, tariff_code, start_date_str, end_date_str, str(e))
    except KeyError as e:
        print(f"Error processing JSON data from Octopus API - unexpected structure: {e}")
        inform_api_issue(output_dir, product_code, tariff_code, start_date_str, end_date_str, f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        inform_api_issue(output_dir, product_code, tariff_code, start_date_str, end_date_str, f"Unexpected error: {e}")

def inform_api_issue(output_dir: Path, product:str, tariff:str, start:str, end:str, error_msg: str) -> None:
    """Provides information if automatic fetching from Octopus API fails."""
    instructions = f"""
    ================================================================================
    IMPORTANT: Automatic fetching of electricity costs from Octopus Energy API
    failed for product '{product}', tariff '{tariff}', period {start} to {end}.
    Error: {error_msg}
    
    Please check:
    1. Your internet connection.
    2. The Octopus Energy API status (https://developer.octopus.energy/status).
    3. The product code ('{product}') and tariff code ('{tariff}') are still valid
       and available for the specified region and dates.
       You can check available products at: https://api.octopus.energy/v1/products/
       And specific tariff details, e.g., for AGILE-FLEX-22-11-25:
       https://api.octopus.energy/v1/products/AGILE-FLEX-22-11-25/
    4. The date range is valid for the selected tariff.
    ================================================================================
    """
    print(instructions)
    placeholder_file = output_dir / f"OCTOPUS_API_FETCH_FAILED_{product}_{tariff.split('-')[-1]}_{start}_{end}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(placeholder_file, 'w') as f:
        f.write(instructions)
    print(f"Information about API fetch issue saved to: {placeholder_file}")

if __name__ == "__main__":
    # Example: Fetch data for London (Region C) for a specific Agile product
    # The DNO_REGION and VOLTAGE_LEVEL are no longer directly used by Octopus API in this way.
    # We use product code and tariff code (which includes region).
    
    # User might need to find the current Agile product code for their region.
    # This can be done by browsing https://api.octopus.energy/v1/products/
    # and then finding the appropriate tariff code for their region within that product.
    # For London (region C), the tariff code would end in "-C".
    # Example for Agile Octopus (product AGILE-FLEX-22-11-25) in region C:
    # Product Code: AGILE-FLEX-22-11-25
    # Tariff Code: E-1R-AGILE-FLEX-22-11-25-C
    
    octopus_product_code = DEFAULT_PRODUCT_CODE
    octopus_region_letter = DEFAULT_REGION_LETTER # For London
    octopus_tariff_code = get_octopus_tariff_code(octopus_product_code, octopus_region_letter)

    START_DATE = "2024-04-03" # Period 2
    END_DATE = "2024-09-30"   # Period 2
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "raw"

    fetch_electricity_costs(
        product_code=octopus_product_code,
        tariff_code=octopus_tariff_code,
        start_date_str=START_DATE,
        end_date_str=END_DATE,
        output_dir=OUTPUT_DIR
    ) 