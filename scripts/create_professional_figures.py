import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for professional figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
MODEL_PATH = "models/heating_demand_predictor_xgb_tuned.joblib"
DATA_PATH = "data/processed/final_dataset_with_forecasts.csv"
FIGURES_OUTPUT_DIR = "reports/figures"

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'muted': '#6C757D',
    'light': '#F8F9FA'
}

def calculate_heating_demand(row, base_temp=18.0):
    """Calculate heating demand in degree-hours."""
    apparent_temp = row['apparent_temperature']
    if pd.isna(apparent_temp):
        return np.nan
    return max(0, base_temp - apparent_temp)

def load_and_prepare_data():
    """Load data and recreate the exact same preprocessing as training."""
    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    df = df.set_index('time').sort_index()
    
    # Create heating demand target
    df['heating_demand_current'] = df.apply(calculate_heating_demand, axis=1)
    df['target_heating_demand'] = df['heating_demand_current'].shift(-1)
    df['target_time'] = df.index.to_series().shift(-1)
    
    return df

def apply_feature_engineering(df):
    """Apply the same feature engineering as in training."""
    # Time features
    if not pd.api.types.is_datetime64_any_dtype(df['target_time']):
        df['target_time'] = pd.to_datetime(df['target_time'])
    
    df['hour'] = df['target_time'].dt.hour
    df['dayofweek'] = df['target_time'].dt.dayofweek
    df['month'] = df['target_time'].dt.month
    df['dayofyear'] = df['target_time'].dt.dayofyear
    df['quarter'] = df['target_time'].dt.quarter
    
    # Enhanced time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    df['dayofyear_sin'] = np.sin(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    df['dayofyear_cos'] = np.cos(2 * np.pi * (df['dayofyear'] - 1) / 365.0)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Season
    def get_season(month_col):
        conditions = [
            month_col.isin([12, 1, 2]), # Winter
            month_col.isin([3, 4, 5]),   # Spring
            month_col.isin([6, 7, 8]),   # Summer
            month_col.isin([9, 10, 11]) # Autumn
        ]
        choices = [0, 1, 2, 3]
        return np.select(conditions, choices, default=np.nan)
    df['season'] = get_season(df['month'])
    
    # Create lag features
    weather_cols = ['apparent_temperature', 'temperature', 'humidity', 'windspeed', 'heating_demand_current']
    for col in weather_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Create derived heating demand from forecast
    if 'apparent_temperature_forecast' in df.columns:
        df['heating_demand_forecast'] = df['apparent_temperature_forecast'].apply(
            lambda x: max(0, 18.0 - x) if pd.notna(x) else np.nan
        )
    
    # Basic interactions
    if 'temperature_forecast' in df.columns and 'windspeed_forecast' in df.columns:
        df['wind_chill_factor'] = df['temperature_forecast'] - (df['windspeed_forecast'] * 0.5)
    
    if 'temperature_forecast' in df.columns and 'humidity_forecast' in df.columns:
        df['temp_humidity_interaction'] = df['temperature_forecast'] * (df['humidity_forecast'] / 100.0)
    
    if 'sw_radiation_forecast' in df.columns and 'cloud_cover_forecast' in df.columns:
        cloud_fraction = (df['cloud_cover_forecast'] / 100.0).clip(0, 1)
        df['solar_heating_effect'] = df['sw_radiation_forecast'] * (1 - cloud_fraction)
    
    return df

def create_loss_curve_figure():
    """Create a professional loss curve figure by retraining with validation tracking."""
    print("🎨 Creating loss curve figure...")
    
    # Load data and prepare for training
    df = load_and_prepare_data()
    df = apply_feature_engineering(df)
    
    # Load model to get feature list
    model_data = joblib.load(MODEL_PATH)
    feature_columns = model_data['features']
    
    # Filter data
    df_clean = df.dropna(subset=feature_columns + ['target_heating_demand'])
    X = df_clean[feature_columns]
    y = df_clean['target_heating_demand']
    
    # Create train/test split (same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create validation split from training data
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)
    
    # Import XGBoost and train model with eval tracking
    import xgboost as xgb
    
    # Use the best hyperparameters from saved model
    best_params = model_data['best_hyperparameters']
    
    # Create model with validation tracking
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        **best_params,
        random_state=42,
        early_stopping_rounds=50,
        verbose=0
    )
    
    # Train with evaluation sets
    model.fit(
        X_train_part, y_train_part,
        eval_set=[(X_train_part, y_train_part), (X_val, y_val)],
        verbose=False
    )
    
    # Extract training history
    results = model.evals_result()
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']
    epochs = range(1, len(train_rmse) + 1)
    
    # Create professional loss curve figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curve
    ax1.plot(epochs, train_rmse, label='Training RMSE', color=COLORS['primary'], linewidth=2)
    ax1.plot(epochs, val_rmse, label='Validation RMSE', color=COLORS['secondary'], linewidth=2)
    ax1.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (degree-hours)', fontsize=12, fontweight='bold')
    ax1.set_title('🔥 Heating Demand Model Training Progress', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add performance annotations
    min_val_idx = np.argmin(val_rmse)
    ax1.annotate(f'Best Val RMSE: {val_rmse[min_val_idx]:.3f}', 
                xy=(min_val_idx + 1, val_rmse[min_val_idx]), 
                xytext=(min_val_idx + 20, val_rmse[min_val_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5),
                fontsize=10, fontweight='bold', color=COLORS['accent'])
    
    # Performance metrics comparison
    final_train_rmse = train_rmse[-1]
    final_val_rmse = val_rmse[-1]
    
    # Test set performance
    y_pred_test = model.predict(X_test)
    y_pred_test = np.maximum(y_pred_test, 0)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Performance summary bar chart
    metrics = ['Training\nRMSE', 'Validation\nRMSE', 'Test\nRMSE', 'Test\nMAE']
    values = [final_train_rmse, final_val_rmse, test_rmse, test_mae]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Error (degree-hours)', fontsize=12, fontweight='bold')
    ax2.set_title('📊 Final Model Performance', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_training_performance.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return test_rmse, test_mae

def create_comprehensive_analysis_figure():
    """Create a comprehensive 2x2 analysis figure."""
    print("🎨 Creating comprehensive analysis figure...")
    
    # Load model and data
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_columns = model_data['features']
    
    df = load_and_prepare_data()
    df = apply_feature_engineering(df)
    df_clean = df.dropna(subset=feature_columns + ['target_heating_demand'])
    
    X = df_clean[feature_columns]
    y_true = df_clean['target_heating_demand']
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, shuffle=False)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    
    # Create the comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Actual vs Predicted Scatter (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(y_test, y_pred, alpha=0.6, s=15, c=COLORS['primary'], edgecolors='white', linewidth=0.5)
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], '--', color=COLORS['accent'], linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Heating Demand (degree-hours)', fontweight='bold')
    ax1.set_ylabel('Predicted Heating Demand (degree-hours)', fontweight='bold')
    ax1.set_title('🎯 Prediction Accuracy', fontweight='bold', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² annotation
    from scipy.stats import pearsonr
    r, _ = pearsonr(y_test, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r**2:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8),
             fontweight='bold', fontsize=11)
    
    # 2. Time series (top-right) 
    ax2 = fig.add_subplot(gs[0, 1])
    time_subset = slice(0, min(300, len(y_test)))
    ax2.plot(range(len(y_test[time_subset])), y_test.iloc[time_subset], 
             label='Actual', color=COLORS['primary'], linewidth=2, alpha=0.8)
    ax2.plot(range(len(y_pred[time_subset])), y_pred[time_subset], 
             label='Predicted', color=COLORS['secondary'], linewidth=2, alpha=0.8, linestyle='--')
    ax2.set_xlabel('Test Sample Index', fontweight='bold')
    ax2.set_ylabel('Heating Demand (degree-hours)', fontweight='bold')
    ax2.set_title('⏱️ Time Series Predictions', fontweight='bold', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    residuals = y_test - y_pred
    ax3.hist(residuals, bins=40, color=COLORS['muted'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axvline(0, color=COLORS['accent'], linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Prediction Error (degree-hours)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('📊 Error Distribution', fontweight='bold', fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add error statistics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ax3.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}', 
             transform=ax3.transAxes, 
             bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8),
             fontweight='bold', fontsize=10, verticalalignment='top')
    
    # 4. Performance by demand level (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Define demand ranges and calculate MAE for each
    demand_ranges = [
        ("No heating\n(0)", y_test == 0),
        ("Low\n(0-5)", (y_test > 0) & (y_test <= 5)),
        ("Medium\n(5-10)", (y_test > 5) & (y_test <= 10)),
        ("High\n(10-15)", (y_test > 10) & (y_test <= 15)),
        ("Very High\n(>15)", y_test > 15)
    ]
    
    labels = []
    maes = []
    counts = []
    colors_demand = []
    
    for label, mask in demand_ranges:
        if mask.sum() > 0:
            mae_subset = mean_absolute_error(y_test[mask], y_pred[mask])
            labels.append(label)
            maes.append(mae_subset)
            counts.append(mask.sum())
            colors_demand.append(COLORS['primary'])
    
    bars = ax4.bar(labels, maes, color=colors_demand, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('MAE (degree-hours)', fontweight='bold')
    ax4.set_title('🔥 Performance by Demand Level', fontweight='bold', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'n={count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add overall title
    fig.suptitle('🔥 Heating Demand Prediction Model - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_feature_importance_figure():
    """Create a beautiful feature importance figure."""
    print("🎨 Creating feature importance figure...")
    
    # Load model
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_columns = model_data['features']
    
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature importances!")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Take top 20 features
    top_n = min(20, len(feature_columns))
    top_indices = indices[:top_n]
    top_features = [feature_columns[i] for i in top_indices]
    top_importances = [importances[i] for i in top_indices]
    
    # Clean up feature names for display
    display_names = []
    for feature in top_features:
        if 'forecast' in feature:
            display_names.append(feature.replace('_forecast', ' (forecast)').replace('_', ' ').title())
        elif 'lag' in feature:
            parts = feature.split('_lag')
            base_name = parts[0].replace('_', ' ').title()
            lag_num = parts[1]
            display_names.append(f"{base_name} (lag {lag_num}h)")
        else:
            display_names.append(feature.replace('_', ' ').title())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(display_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(display_names)))
    
    bars = ax.barh(y_pos, top_importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize appearance
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Feature Importance', fontweight='bold', fontsize=12)
    ax.set_title('🔥 Heating Demand Prediction - Top 20 Feature Importances', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_feature_importance_professional.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_seasonal_performance_figure():
    """Create a seasonal performance analysis figure."""
    print("🎨 Creating seasonal performance figure...")
    
    # Load model and data
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_columns = model_data['features']
    
    df = load_and_prepare_data()
    df = apply_feature_engineering(df)
    df_clean = df.dropna(subset=feature_columns + ['target_heating_demand'])
    
    X = df_clean[feature_columns]
    y_true = df_clean['target_heating_demand']
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, shuffle=False)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    
    # Get test data with predictions
    test_data = df_clean.loc[y_test.index].copy()
    test_data['y_pred'] = y_pred
    test_data['residuals'] = y_test - y_pred
    test_data['abs_residuals'] = np.abs(test_data['residuals'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. MAE by hour of day
    hourly_mae = test_data.groupby('hour')['abs_residuals'].mean()
    ax1.plot(hourly_mae.index, hourly_mae.values, marker='o', linewidth=2, 
             markersize=6, color=COLORS['primary'])
    ax1.fill_between(hourly_mae.index, hourly_mae.values, alpha=0.3, color=COLORS['primary'])
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_ylabel('MAE (degree-hours)', fontweight='bold')
    ax1.set_title('🕐 Performance by Hour of Day', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 3))
    
    # 2. MAE by month - Fixed handling of float indices
    monthly_mae = test_data.groupby('month')['abs_residuals'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Handle potential float indices
    valid_months = []
    monthly_values = []
    colors_month = []
    months_in_data = []
    
    for month_idx in monthly_mae.index:
        if pd.notna(month_idx):
            month_int = int(month_idx)
            valid_months.append(month_int)
            monthly_values.append(monthly_mae[month_idx])
            months_in_data.append(month_names[month_int - 1])
            colors_month.append(COLORS['primary'] if month_int in [5, 6, 7, 8, 9] else COLORS['secondary'])
    
    bars = ax2.bar(months_in_data, monthly_values, color=colors_month, 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Month', fontweight='bold')
    ax2.set_ylabel('MAE (degree-hours)', fontweight='bold')
    ax2.set_title('📅 Performance by Month', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. MAE by season
    seasonal_mae = test_data.groupby('season')['abs_residuals'].mean()
    season_names = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    valid_seasons = []
    seasonal_values = []
    seasons_in_data = []
    season_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    season_colors_selected = []
    
    for season_idx in seasonal_mae.index:
        if pd.notna(season_idx):
            season_int = int(season_idx)
            valid_seasons.append(season_int)
            seasonal_values.append(seasonal_mae[season_idx])
            seasons_in_data.append(season_names[season_int])
            season_colors_selected.append(season_colors[season_int])
    
    bars = ax3.bar(seasons_in_data, seasonal_values, color=season_colors_selected, 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Season', fontweight='bold')
    ax3.set_ylabel('MAE (degree-hours)', fontweight='bold')
    ax3.set_title('🍂 Performance by Season', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on seasonal bars
    for bar, value in zip(bars, seasonal_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Heating demand distribution by season
    season_demand = []
    season_labels_box = []
    season_colors_dist = []
    
    for season_idx in sorted([s for s in test_data['season'].unique() if pd.notna(s)]):
        season_int = int(season_idx)
        season_data = test_data[test_data['season'] == season_idx]['target_heating_demand']
        season_demand.append(season_data.values)
        season_labels_box.append(season_names[season_int])
        season_colors_dist.append(season_colors[season_int])
    
    if season_demand:  # Only create box plot if we have data
        box_plot = ax4.boxplot(season_demand, labels=season_labels_box, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], season_colors_dist):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax4.set_xlabel('Season', fontweight='bold')
    ax4.set_ylabel('Heating Demand (degree-hours)', fontweight='bold')
    ax4.set_title('🔥 Heating Demand Distribution by Season', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🔥 Heating Demand Model - Seasonal Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'heating_demand_seasonal_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Generate all professional figures."""
    print("🎨 CREATING PROFESSIONAL FIGURES FOR HEATING DEMAND MODEL")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
    
    # Generate all figures
    test_rmse, test_mae = create_loss_curve_figure()
    create_comprehensive_analysis_figure()
    create_feature_importance_figure() 
    create_seasonal_performance_figure()
    
    print(f"\n✅ ALL PROFESSIONAL FIGURES CREATED!")
    print(f"📁 Saved to: {FIGURES_OUTPUT_DIR}/")
    print(f"📊 Final Model Performance:")
    print(f"   • Test RMSE: {test_rmse:.4f} degree-hours")
    print(f"   • Test MAE:  {test_mae:.4f} degree-hours")
    print("\n🎨 Generated Figures:")
    print("   • heating_demand_training_performance.png - Loss curves & metrics")
    print("   • heating_demand_comprehensive_analysis.png - 4-panel analysis")
    print("   • heating_demand_feature_importance_professional.png - Top features")
    print("   • heating_demand_seasonal_analysis.png - Temporal patterns")

if __name__ == '__main__':
    main() 