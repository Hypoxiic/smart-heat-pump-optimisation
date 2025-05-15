# Smart Heat Pump Optimisation Using Open Data

## 🎯 Objective
Develop a machine learning model to optimise residential air-to-water heat pump operation. The goal is to minimise energy costs and carbon emissions by adjusting usage schedules based on:
- Weather forecasts (Open-Meteo)
- Carbon intensity (UK Carbon Intensity API)
- Electricity tariffs (ElectricityCosts.org.uk)

## 🧩 Development Stages

### 1. 🏁 Initial Setup
- Create project structure with directories:
  - `data/` – raw and processed data
  - `scripts/` – data fetching and preprocessing scripts
  - `models/` – ML and optimisation code
  - `notebooks/` – EDA and exploratory modelling
- Create a `README.md` file that includes:
  - Project overview
  - Data source descriptions
  - Setup and usage instructions
  - Licensing info
- Continuously update `README.md` as project evolves

### 2. 📥 Data Acquisition
- Write Python scripts to fetch data from:
  - Open-Meteo (historical weather data using the `/v1/archive` endpoint)
  - UK Carbon Intensity API (30-minute interval carbon intensity)
  - Octopus Energy API (half-hourly Agile tariff electricity prices)
- Implement a multi-period data fetching strategy due to changes in available Octopus Agile tariffs, covering:
    - Period 1: `AGILE-FLEX-22-11-25` (2022-11-25 to 2023-12-11)
    - Gap Period: (2023-12-12 to 2024-04-02) - Weather and Carbon Intensity data fetched; no corresponding single Agile tariff identified for electricity costs for this entire span.
    - Period 2: `AGILE-24-04-03` (2024-04-03 to 2024-09-30)
- Save raw datasets to `data/raw/` with corresponding API call detail JSON files.
- Document API usage, endpoints, and specific product codes/periods in `README.md`.

### 3. 📊 Exploratory Data Analysis
- Perform EDA in Jupyter Notebooks:
  - Temporal patterns in temperature, emissions, tariffs
  - Correlations across variables
- Generate visuals (matplotlib/plotly/seaborn)
- Summarise findings in `README.md`

### 4. 🤖 Model Development
- Predict short-term heating demand:
  - XGBoost for tabular modelling
  - LSTM for sequence modelling
- Optimisation engine:
  - Pyomo or simple reinforcement learning
  - Schedule heat pump operation to reduce cost/emissions
- Document model pipeline and results in `README.md`

### 5. 🔁 Simulation and Testing
- Simulate control decisions under varying scenarios
- Compare against naive or fixed-schedule baseline
- Quantify savings in cost and carbon
- Describe methods and outcomes in `README.md`

### 6. 🚀 Deployment Prep (optional)
- Modularise code
- Add configuration via `.env` or YAML
- Write CLI entry or script to run full pipeline
- Update `README.md` with usage examples

## 📘 Continuous Documentation
- `README.md` must stay updated through all phases
- Include:
  - Installation
  - Data schemas
  - Results
  - Limitations and future work

## Data Sources
- **Open-Meteo:** Provides historical hourly weather data.
  - **Endpoint Used:** `https://archive-api.open-meteo.com/v1/archive`
  - **Parameters for Fetching (example):**
    - `latitude`: 51.5074 (London)
    - `longitude`: 0.1278 (London)
    - `hourly`: "temperature_2m,relative_humidity_2m,wind_speed_10m,apparent_temperature"
    - `timezone`: "UTC" (Data is fetched in UTC)
    - `start_date`: (e.g., "2022-11-25")
    - `end_date`: (e.g., "2023-12-11")
  - **Script:** `scripts/fetch_open_meteo.py` fetches data and stores it in `data/raw/` as a CSV file. It also saves API call details (URL, parameters, data snippet) in a corresponding JSON file.
- **UK Carbon Intensity API:** Offers data on carbon intensity of electricity generation in Great Britain.
  - **Endpoint Used:** `https://api.carbonintensity.org.uk/intensity/date/{YYYY-MM-DD}` (iterated for date ranges)
  - **Data Fetched:** 30-minute interval data including forecast intensity, actual intensity, and intensity index (e.g., low, moderate, high).
  - **Script:** `scripts/fetch_carbon_intensity.py` fetches data for a specified range and stores it in `data/raw/` as a CSV file. It also saves API call details (URLs called, date range, data snippet) in a corresponding JSON file.
- **ElectricityCosts.org.uk:** (Superseded by Octopus Energy API for this project)
  - Previously investigated for non-domestic tariffs. Found to be unreliable for programmatic access and not suitable for Agile residential pricing.
- **Octopus Energy API:** Provides access to various electricity tariffs, including Agile Octopus which offers half-hourly pricing.
  - **API Documentation:** [https://developer.octopus.energy/rest/](https://developer.octopus.energy/rest/)
  - **Endpoint Used (Example for Agile tariff):** `https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/`
  - **Specific Tariffs and Periods Used:**
    - **Period 1:**
      - Product Code: `AGILE-FLEX-22-11-25`
      - Tariff Code Example: `E-1R-AGILE-FLEX-22-11-25-C` (Region C - London)
      - Data Fetched For: `2022-11-25` to `2023-12-11`
    - **Gap Period (`2023-12-12` to `2024-04-02`):** No single continuous Agile tariff was identified to cover this entire period. Therefore, electricity cost data for this specific span is not included from Octopus Agile tariffs in the final merged dataset.
    - **Period 2:**
      - Product Code: `AGILE-24-04-03`
      - Tariff Code Example: `E-1R-AGILE-24-04-03-C` (Region C - London)
      - Data Fetched For: `2024-04-03` to `2024-09-30`
  - **Parameters for Fetching (example within script):**
    - `period_from`: (e.g., "2022-11-25T00:00:00Z")
    - `period_to`: (e.g., "2023-12-12T00:00:00Z" - API `period_to` is exclusive)
    - `page_size`: Default is 100, script uses a larger value (e.g. 1500, actual value may vary in script for max) to handle pagination for long periods.
  - **Script:** `scripts/fetch_electricity_costs.py` fetches half-hourly price data (pence per kWh) for the specified Agile tariffs and periods. It saves the data to `data/raw/` as CSV files and API call details to corresponding JSON files.
  - **Data Format:** Half-hourly cost data (`cost_p_per_kwh`) and `time` (UTC).

## Data Preprocessing and Final Dataset

The script `scripts/preprocess_data.py` is responsible for cleaning the raw data files and preparing a unified dataset for analysis and modeling. The key steps for each data type and period are:

1.  **Loading Data:** Raw CSV files for weather, carbon intensity, and electricity costs are loaded from `data/raw/`.
2.  **Time Standardization:** All time-related columns are converted to `datetime` objects and localized or converted to UTC to ensure consistency.
3.  **Resampling (Carbon Intensity & Electricity Costs):** Data originally in 30-minute intervals (carbon intensity, electricity costs) is resampled to hourly frequency. For electricity costs, the mean of the two half-hourly `cost_p_per_kwh` values within the hour is taken. For carbon intensity, `mean` is used for forecast/actual, and `first` for the index.
4.  **Column Renaming and Selection:** Columns are renamed for consistency (e.g., `temperature_2m` to `temperature`) and relevant columns are selected.
5.  **Individual Processed Files:** Processed hourly data for each source and period is saved to `data/processed/` (e.g., `weather_hourly_{period}.csv`).

**Merging Datasets (per period):**
- For each defined period (Period 1, Gap Period, Period 2), the corresponding processed hourly weather, carbon intensity, and (if available) electricity cost dataframes are merged using an outer merge on the hourly `time` index.
- Missing values after the merge (e.g., if one dataset has a slightly different time range start/end for an hour) are handled using forward fill (`ffill`) and then backward fill (`bfill`).
- For the **Gap Period (2023-12-12 to 2024-04-02)**, no electricity cost data is included in its merged file, as no continuous Agile tariff was used for this span.
- These period-specific merged files are saved to `data/processed/` (e.g., `merged_project_data_{period_start}_{period_end}.csv`).

**Final Concatenation:**
- The three period-specific merged CSV files:
    - `merged_project_data_2022-11-25_2023-12-11.csv`
    - `merged_project_data_2023-12-12_2024-04-02.csv`
    - `merged_project_data_2024-04-03_2024-09-30.csv`
- ...are loaded and concatenated in chronological order.
- Duplicate indices (if any from boundary overlaps, though unlikely with current setup) are removed, keeping the first occurrence.
- Any rows with `NaT` (Not a Time) in the index are dropped.
- **Handling Missing Electricity Costs in Gap:** To preserve the known gap in electricity cost data, the `cost_p_per_kwh` column is separated before `ffill`/`bfill` is applied to the other columns (weather, carbon intensity). The `cost_p_per_kwh` column (which contains NaNs for the Gap Period) is then rejoined. This ensures that weather and carbon data are continuous where available, while electricity costs correctly show NaNs for the period they were not fetched.
- The final, comprehensive dataset is saved as `data/processed/final_dataset_all_periods.csv`.

**Final Dataset Overview:**
- **Timespan:** November 25, 2022, to September 30, 2024 (hourly data).
- **Columns:** `time` (UTC index), `temperature`, `humidity`, `windspeed`, `apparent_temperature`, `carbon_intensity_forecast`, `carbon_intensity_actual`, `carbon_intensity_index`, `cost_p_per_kwh`.
- The `cost_p_per_kwh` column contains `NaN` values for the period `2023-12-12` to `2024-04-02`.

## Next Steps: Feature Engineering, Gap Handling, and Baselines

Based on the initial EDA and the project goals, the following steps are planned for feature engineering and establishing baseline performance:

### Feature Engineering
- **Categorical Encoding:** Convert `carbon_intensity_index` (which is categorical: low, moderate, high, very high) into a numerical format. Options include ordered integer mapping (e.g., Low=0, Moderate=1, High=2, Very High=3) or one-hot encoding.
- **Lag Features:** Create lagged versions of key variables, particularly for `cost_p_per_kwh` and carbon intensity signals (e.g., values from the previous hour, same hour previous day, or 24 hours prior) to capture auto-correlations and daily patterns.
- **Cyclical Time Features:** Encode time-based features such as hour of day, day of week, and month using cyclical encoding methods (e.g., sine/cosine transformations) to represent their circular nature effectively for machine learning models.

### Gap Period Handling (for `cost_p_per_kwh`)
Strategies for managing the NaN values in the `cost_p_per_kwh` column during the Gap Period (`2023-12-12` to `2024-04-02`) for model training tasks:
- **For Price Prediction Models:** Rows corresponding to the Gap Period might be dropped if the primary goal is to predict actual observed prices.
- **Imputation Strategies:** Alternatively, consider imputing the missing price data. This could involve:
    - Simpler methods (e.g., mean/median of surrounding periods, though less ideal for volatile prices).
    - Time-series models (e.g., SARIMAX), potentially using `carbon_intensity_actual` or `carbon_intensity_forecast` as exogenous variables if a relationship is confirmed and stable. If imputation is used, an additional binary indicator column (`is_price_imputed`) should be created to flag these rows, allowing models to treat these data points differently if necessary.
- **For the Optimiser:** The optimiser itself will need a defined strategy for these NaN periods, such as switching to a different objective (e.g., pure carbon minimization if cost is unknown) or using a pre-defined surrogate/fallback tariff.

### Baseline Benchmarks
Before developing complex optimisation models, establish baseline performance using simple rule-based approaches. This will provide a reference point to quantify the benefits of the advanced optimiser.
- **Example Rules:**
    - "Run heat pump if `cost_p_per_kwh` < X (e.g., 20 p/kWh)"
    - "Run heat pump if `carbon_intensity_actual` < Y (e.g., 120 gCO2/kWh)"
    - Combinations of these, or time-of-day based rules.
- **Outcome:** Compute the total operational cost and carbon emissions over the entire dataset (or representative periods) for these baseline rules. This will generate baseline cost and carbon curves against which the ML-driven optimiser can be compared.

## Setup and Usage
(Instructions to be added)

## Licensing
(Licensing information to be added) 