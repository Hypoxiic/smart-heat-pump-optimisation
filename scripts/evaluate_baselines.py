import pandas as pd
import numpy as np
import argparse # For command-line arguments
import os

# --- Configuration ---
DEFAULT_DATA_FILE_PATH = 'data/processed/featured_dataset_phase2.csv'
IMPUTED_DATA_FILE_PATH = 'data/processed/featured_dataset_phase3_imputed.csv' # Path to the new imputed dataset
HEAT_PUMP_POWER_KW = 1.0
DEFAULT_HIGH_COST_P_PER_KWH = 999 # Still used if prices are NaN AND not part of a known imputed set

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    print(f"Data loaded. Shape: {df.shape}")
    # Ensure 'is_price_imputed' column exists if not present, default to False
    if 'is_price_imputed' not in df.columns:
        df['is_price_imputed'] = False
    return df

def calculate_metrics(df_run_schedule, heat_pump_power_kw, data_has_imputed_prices):
    results = {}
    df_schedule = df_run_schedule.copy()
    
    results['total_consumption_kwh'] = (df_schedule['run_heat_pump'] * heat_pump_power_kw).sum()

    # Cost Calculation Strategy:
    # If data_has_imputed_prices is True, we assume NaNs in original cost_p_per_kwh might have been filled.
    # We still need to handle original NaNs if the rule operates outside imputed regions OR if imputation wasn't perfect.
    # The `is_price_imputed` column marks where imputation happened.
    
    # Create a working cost column. Start with actual prices.
    df_schedule['cost_p_per_kwh_for_calc'] = df_schedule['cost_p_per_kwh']

    # If a price is NaN AND it was NOT imputed (meaning it's a genuine NaN outside the imputed gap, or imputation failed for some reason)
    # then apply the penalty. This path should ideally not be hit much if using the fully imputed dataset for the gap.
    # If using non-imputed data (phase2), this handles the gap penalty.
    # If using imputed data (phase3), this handles any *other* NaNs outside the imputed gap.
    condition_for_penalty = df_schedule['cost_p_per_kwh_for_calc'].isna() & ~df_schedule['is_price_imputed']
    if not data_has_imputed_prices: # Simpler logic for old dataset: penalize all NaNs
         condition_for_penalty = df_schedule['cost_p_per_kwh_for_calc'].isna()

    df_schedule.loc[condition_for_penalty, 'cost_p_per_kwh_for_calc'] = DEFAULT_HIGH_COST_P_PER_KWH
    
    # If any costs are still NaN after this (e.g., if imputation failed and we didn't penalize), make them 0 to avoid sum issues, but this is an error condition.
    if df_schedule['cost_p_per_kwh_for_calc'].isna().any():
        print("Warning: cost_p_per_kwh_for_calc still contains NaNs after processing. Filling with 0 for sum, but check data!")
        df_schedule['cost_p_per_kwh_for_calc'] = df_schedule['cost_p_per_kwh_for_calc'].fillna(0)
        
    df_schedule['cost_gbp_per_kwh_filled'] = df_schedule['cost_p_per_kwh_for_calc'] / 100.0
    results['total_cost_gbp'] = (df_schedule['run_heat_pump'] * heat_pump_power_kw * df_schedule['cost_gbp_per_kwh_filled']).sum()

    df_schedule['carbon_kgco2_per_kwh'] = df_schedule['carbon_intensity_actual'] / 1000.0
    results['total_carbon_kgco2'] = (df_schedule['run_heat_pump'] * heat_pump_power_kw * df_schedule['carbon_kgco2_per_kwh']).sum()
    
    results['hours_run_with_nan_cost_data_points'] = df_schedule[df_schedule['run_heat_pump'] & df_schedule['cost_p_per_kwh'].isna()].shape[0] * heat_pump_power_kw
    results['hours_run_with_imputed_price'] = df_schedule[df_schedule['run_heat_pump'] & df_schedule['is_price_imputed']].shape[0] * heat_pump_power_kw
    results['hours_run_with_nan_carbon'] = df_schedule[df_schedule['run_heat_pump'] & df_schedule['carbon_intensity_actual'].isna()].shape[0] * heat_pump_power_kw
    return results

# --- Baseline Rule Implementations ---

def baseline_always_on(df):
    """Baseline: Heat pump is always on."""
    df_schedule = df.copy()
    df_schedule['run_heat_pump'] = True
    return df_schedule

def baseline_cost_threshold(df, cost_threshold_p_per_kwh):
    """Baseline: Run if cost < threshold.
       If cost is NaN, do not run (conservative approach)."""
    df_schedule = df.copy()
    # Ensure 'cost_p_per_kwh' is available
    if 'cost_p_per_kwh' not in df_schedule.columns:
        print(f"Warning: 'cost_p_per_kwh' not in DataFrame for cost_threshold rule.")
        df_schedule['run_heat_pump'] = False # Cannot apply rule
        return df_schedule
        
    df_schedule['run_heat_pump'] = df_schedule['cost_p_per_kwh'] < cost_threshold_p_per_kwh
    # Handle NaN costs explicitly: if cost is NaN, rule condition is False, so don't run.
    df_schedule.loc[df_schedule['cost_p_per_kwh'].isna(), 'run_heat_pump'] = False
    return df_schedule

def baseline_carbon_threshold(df, carbon_threshold_gco2_per_kwh):
    """Baseline: Run if carbon intensity < threshold."""
    df_schedule = df.copy()
    if 'carbon_intensity_actual' not in df_schedule.columns:
        print(f"Warning: 'carbon_intensity_actual' not in DataFrame for carbon_threshold rule.")
        df_schedule['run_heat_pump'] = False
        return df_schedule

    df_schedule['run_heat_pump'] = df_schedule['carbon_intensity_actual'] < carbon_threshold_gco2_per_kwh
    # Handle NaN carbon intensity if necessary, though less likely for 'actual' based on processing.
    df_schedule.loc[df_schedule['carbon_intensity_actual'].isna(), 'run_heat_pump'] = False
    return df_schedule

def baseline_fixed_schedule(df, start_hour, end_hour):
    """Baseline: Run during fixed daily hours (inclusive start, exclusive end)."""
    df_schedule = df.copy()
    df_schedule['run_heat_pump'] = (df_schedule.index.hour >= start_hour) & (df_schedule.index.hour < end_hour)
    return df_schedule
    
def baseline_combined_threshold(df, cost_threshold_p_per_kwh, carbon_threshold_gco2_per_kwh):
    """Baseline: Run if cost < X AND carbon < Y.
    If cost is NaN, do not run."""
    df_schedule = df.copy()
    if 'cost_p_per_kwh' not in df_schedule.columns or 'carbon_intensity_actual' not in df_schedule.columns:
        print(f"Warning: Missing columns for combined_threshold rule.")
        df_schedule['run_heat_pump'] = False
        return df_schedule
        
    run_on_cost = df_schedule['cost_p_per_kwh'] < cost_threshold_p_per_kwh
    run_on_carbon = df_schedule['carbon_intensity_actual'] < carbon_threshold_gco2_per_kwh
    
    df_schedule['run_heat_pump'] = run_on_cost & run_on_carbon
    # If cost is NaN, condition 'run_on_cost' becomes False, so 'run_heat_pump' becomes False.
    df_schedule.loc[df_schedule['cost_p_per_kwh'].isna(), 'run_heat_pump'] = False
    df_schedule.loc[df_schedule['carbon_intensity_actual'].isna(), 'run_heat_pump'] = False
    return df_schedule

def baseline_temperature_driven(df, temp_threshold_c, temperature_col='apparent_temperature'):
    """Baseline: Run if temperature < threshold."""
    df_schedule = df.copy()
    if temperature_col not in df_schedule.columns:
        print(f"Warning: '{temperature_col}' not in DataFrame for temperature_driven rule.")
        df_schedule['run_heat_pump'] = False # Or some default behaviour, like always run if temp unavailable
        return df_schedule
    df_schedule['run_heat_pump'] = df_schedule[temperature_col] < temp_threshold_c
    # Handle NaN temperatures: if temperature is NaN, rule condition is False, so don't run.
    df_schedule.loc[df_schedule[temperature_col].isna(), 'run_heat_pump'] = False
    return df_schedule

def main(input_file_path):
    df_featured = load_data(input_file_path)
    data_has_imputed_prices = df_featured['is_price_imputed'].any()
    
    if data_has_imputed_prices:
        print("\nINFO: Running with dataset containing imputed prices. Cost calculation will use these.")
    else:
        print("\nINFO: Running with original dataset (or no imputed prices detected). NaN costs will be penalized.")

    all_baseline_results = {}
    print("\n--- Evaluating Baselines --- Kerr Sering")

    # 1. Always-On
    print("\nEvaluating: Always-On")
    schedule_always_on = baseline_always_on(df_featured)
    metrics_always_on = calculate_metrics(schedule_always_on, HEAT_PUMP_POWER_KW, data_has_imputed_prices)
    all_baseline_results['Always-On'] = metrics_always_on
    print(f"Results: {metrics_always_on}")

    # 2. Cost-Threshold
    cost_thresholds = [15, 20, 25] # p/kWh
    for ct in cost_thresholds:
        rule_name = f'Cost-Threshold (<{ct}p/kWh)'
        print(f"\nEvaluating: {rule_name}")
        schedule_cost_thresh = baseline_cost_threshold(df_featured, ct)
        metrics_cost_thresh = calculate_metrics(schedule_cost_thresh, HEAT_PUMP_POWER_KW, data_has_imputed_prices)
        all_baseline_results[rule_name] = metrics_cost_thresh
        print(f"Results: {metrics_cost_thresh}")
        if metrics_cost_thresh['hours_run_with_nan_cost_data_points'] > 0:
            print(f"Warning: This rule operated for {metrics_cost_thresh['hours_run_with_nan_cost_data_points']} kWh equivalent hours where cost was NaN. Cost calculation might be underestimated if not handled by rule.")

    # 3. Carbon-Threshold
    carbon_thresholds = [100, 150, 200] # gCO2/kWh
    for ct in carbon_thresholds:
        rule_name = f'Carbon-Threshold (<{ct}gCO2/kWh)'
        print(f"\nEvaluating: {rule_name}")
        schedule_carbon_thresh = baseline_carbon_threshold(df_featured, ct)
        metrics_carbon_thresh = calculate_metrics(schedule_carbon_thresh, HEAT_PUMP_POWER_KW, data_has_imputed_prices)
        all_baseline_results[rule_name] = metrics_carbon_thresh
        print(f"Results: {metrics_carbon_thresh}")

    # 4. Fixed Schedule (e.g., off-peak 00:00-05:00)
    print("\nEvaluating: Fixed Schedule (00:00-05:00)")
    schedule_fixed = baseline_fixed_schedule(df_featured, 0, 5) # Hours 0, 1, 2, 3, 4
    metrics_fixed = calculate_metrics(schedule_fixed, HEAT_PUMP_POWER_KW, data_has_imputed_prices)
    all_baseline_results['Fixed Schedule (00:00-05:00)'] = metrics_fixed
    print(f"Results: {metrics_fixed}")

    # 5. Combined Threshold
    cost_thresh_combined = 20 # p/kWh
    carbon_thresh_combined = 150 # gCO2/kWh
    rule_name_combined = f'Combined (<{cost_thresh_combined}p/kWh AND <{carbon_thresh_combined}gCO2/kWh)'
    print(f"\nEvaluating: {rule_name_combined}")
    schedule_combined = baseline_combined_threshold(df_featured, cost_thresh_combined, carbon_thresh_combined)
    metrics_combined = calculate_metrics(schedule_combined, HEAT_PUMP_POWER_KW, data_has_imputed_prices)
    all_baseline_results[rule_name_combined] = metrics_combined
    print(f"Results: {metrics_combined}")

    # 6. Temperature-Driven Baseline
    temp_thresh_c = 18.0
    temp_col = 'apparent_temperature' # or 'temperature'
    rule_name_temp = f'Temperature-Driven (<{temp_thresh_c}°C on {temp_col})'
    print(f"\nEvaluating: {rule_name_temp}")
    schedule_temp_driven = baseline_temperature_driven(df_featured, temp_thresh_c, temp_col)
    metrics_temp_driven = calculate_metrics(schedule_temp_driven, HEAT_PUMP_POWER_KW, data_has_imputed_prices)
    all_baseline_results[rule_name_temp] = metrics_temp_driven
    print(f"Results: {metrics_temp_driven}")

    # --- Summarise All Results ---
    print("\n\n--- All Baseline Results Summary ---")
    results_df = pd.DataFrame.from_dict(all_baseline_results, orient='index')
    # Add a column for efficiency or rank if desired later
    print(results_df)
    
    output_csv_suffix = "imputed" if data_has_imputed_prices else "original_with_penalty"
    output_csv_filename = f'data/processed/baseline_evaluation_results_{output_csv_suffix}.csv'
    results_df.to_csv(output_csv_filename)
    print(f"\nBaseline results saved to {output_csv_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate baseline heat pump control strategies.')
    parser.add_argument('--input_file', type=str, default=None, 
                        help=f'Path to the input dataset. Defaults to {DEFAULT_DATA_FILE_PATH} (original) or {IMPUTED_DATA_FILE_PATH} (if --use_imputed_data is set).')
    parser.add_argument('--use_imputed_data', action='store_true',
                        help=f'If set, uses the imputed dataset ({IMPUTED_DATA_FILE_PATH}) instead of the default.')
    
    args = parser.parse_args()
    
    file_to_process = DEFAULT_DATA_FILE_PATH
    if args.use_imputed_data:
        file_to_process = IMPUTED_DATA_FILE_PATH
    if args.input_file:
        file_to_process = args.input_file # Override if specific path given

    if not os.path.exists(file_to_process):
        print(f"Error: Input data file not found at {file_to_process}")
        if file_to_process == IMPUTED_DATA_FILE_PATH and args.use_imputed_data:
            print(f"Please ensure you have run scripts/impute_gap_prices.py first to generate {IMPUTED_DATA_FILE_PATH}")
    else:
        main(file_to_process) 