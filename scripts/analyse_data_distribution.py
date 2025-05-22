import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/processed/final_dataset_with_forecasts.csv')
df['target'] = df['apparent_temperature'].shift(-1)
df_clean = df.dropna(subset=['target'])

print('TARGET VARIABLE DISTRIBUTION ANALYSIS')
print('=' * 50)

print(f'Total samples: {len(df_clean):,}')
print(f'Min: {df_clean.target.min():.2f}°C')
print(f'Max: {df_clean.target.max():.2f}°C')
print(f'Mean: {df_clean.target.mean():.2f}°C')
print(f'Median: {df_clean.target.median():.2f}°C')
print(f'Std: {df_clean.target.std():.2f}°C')
print()

print('DISTRIBUTION BY TEMPERATURE RANGES:')
print('-' * 30)
ranges = [
    ('< 0°C', df_clean.target < 0),
    ('0-10°C', (df_clean.target >= 0) & (df_clean.target < 10)),
    ('10-15°C', (df_clean.target >= 10) & (df_clean.target < 15)),
    ('15-20°C', (df_clean.target >= 15) & (df_clean.target < 20)),
    ('20-25°C', (df_clean.target >= 20) & (df_clean.target < 25)),
    ('25-30°C', (df_clean.target >= 25) & (df_clean.target < 30)),
    ('>= 30°C', df_clean.target >= 30)
]

for label, condition in ranges:
    count = condition.sum()
    percentage = condition.mean() * 100
    print(f'{label:8}: {count:6,} ({percentage:5.1f}%)')

print()
print('TRAINING vs TEST SPLIT ANALYSIS:')
print('-' * 35)

# Recreate exact split from training script
X_dummy = df_clean[['apparent_temperature']].values
y = df_clean['target'].values
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.2, shuffle=False)

print(f'Train samples: {len(y_train):,}')
print(f'Test samples: {len(y_test):,}')
print()
print('TRAIN SET:')
print(f'  Max temp: {y_train.max():.2f}°C')
print(f'  Mean temp: {y_train.mean():.2f}°C')
print(f'  >15°C: {(y_train > 15).sum():,} ({(y_train > 15).mean()*100:.1f}%)')
print(f'  >20°C: {(y_train > 20).sum():,} ({(y_train > 20).mean()*100:.1f}%)')
print(f'  >25°C: {(y_train > 25).sum():,} ({(y_train > 25).mean()*100:.1f}%)')
print(f'  >30°C: {(y_train > 30).sum():,} ({(y_train > 30).mean()*100:.1f}%)')

print()
print('TEST SET:')
print(f'  Max temp: {y_test.max():.2f}°C')
print(f'  Mean temp: {y_test.mean():.2f}°C')
print(f'  >15°C: {(y_test > 15).sum():,} ({(y_test > 15).mean()*100:.1f}%)')
print(f'  >20°C: {(y_test > 20).sum():,} ({(y_test > 20).mean()*100:.1f}%)')
print(f'  >25°C: {(y_test > 25).sum():,} ({(y_test > 25).mean()*100:.1f}%)')
print(f'  >30°C: {(y_test > 30).sum():,} ({(y_test > 30).mean()*100:.1f}%)')

print()
print('TEMPORAL DISTRIBUTION:')
print('-' * 20)
df_with_time = pd.read_csv('data/processed/final_dataset_with_forecasts.csv', parse_dates=['time'])
df_with_time['target'] = df_with_time['apparent_temperature'].shift(-1)
df_with_time = df_with_time.dropna(subset=['target'])
df_with_time['month'] = df_with_time['time'].dt.month
df_with_time['year'] = df_with_time['time'].dt.year

monthly_stats = df_with_time.groupby('month')['target'].agg(['mean', 'max', 'count'])
print('Monthly target temperature stats:')
for month in range(1, 13):
    if month in monthly_stats.index:
        mean_temp = monthly_stats.loc[month, 'mean']
        max_temp = monthly_stats.loc[month, 'max']
        count = monthly_stats.loc[month, 'count']
        print(f'  Month {month:2d}: Mean={mean_temp:5.1f}°C, Max={max_temp:5.1f}°C, Count={count:4,}')

print()
print('POTENTIAL ISSUES IDENTIFIED:')
print('-' * 30)
high_temp_train = (y_train > 25).sum()
high_temp_test = (y_test > 25).sum()
print(f'• High temp (>25°C) training examples: {high_temp_train:,}')
print(f'• High temp (>25°C) test examples: {high_temp_test:,}')

if high_temp_train < 1000:
    print('⚠️  WARNING: Very few high-temperature training examples!')
if y_test.max() > y_train.max():
    print(f'⚠️  WARNING: Test set has higher max temp ({y_test.max():.1f}°C) than training ({y_train.max():.1f}°C)!')

# Check seasonal split
train_end_idx = len(y_train)
df_split = df_with_time.iloc[:len(df_clean)]  # Match our clean dataset
train_months = df_split.iloc[:int(len(df_split) * 0.8)]['month'].unique()
test_months = df_split.iloc[int(len(df_split) * 0.8):]['month'].unique()
print(f'• Training months: {sorted(train_months)}')
print(f'• Test months: {sorted(test_months)}') 