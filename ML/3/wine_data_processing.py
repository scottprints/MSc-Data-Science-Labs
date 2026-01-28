import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

path = os.path.dirname(os.path.abspath(__file__))
df_a = pd.read_csv(os.path.join(path, 'Wine_a.csv'))
df_b = pd.read_csv(os.path.join(path, 'Wine_b.csv'))

print("Dataset A shape:", df_a.shape)
print("Dataset B shape:", df_b.shape)
print("\n" + "="*80)

# Q1: Merge datasets on ID, then drop ID column
print("\nQ1: Merging datasets using ID")
print("-" * 80)

df = pd.merge(df_a, df_b, on='ID', how='outer')
print(f"Merged shape: {df.shape}")
print(f"Columns before: {df.columns.tolist()}")

df = df.drop('ID', axis=1)
print(f"Columns after: {df.columns.tolist()}")
print(f"Final shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\n" + "="*80)

# Q2: Remove duplicates
print("\nQ2: Removing duplicates")
print("-" * 80)

dups = df.duplicated().sum()
print(f"Duplicate rows found: {dups}")

df = df.drop_duplicates()
print(f"Shape after deduplication: {df.shape}")

print("\n" + "="*80)

# Q3: Handle missing values - drop rows with >2 NaNs, then drop remaining NaNs
print("\nQ3: Handling missing values")
print("-" * 80)

nan_count = df.isna().sum(axis=1)
print(f"NaN distribution:")
print(f"  Min: {nan_count.min()}, Max: {nan_count.max()}, Mean: {nan_count.mean():.2f}")
print(nan_count.value_counts().sort_index())

# Remove rows with more than 2 missing values
threshold = nan_count > 2
removed = threshold.sum()
print(f"\nRows with >2 NaNs: {removed}")

df = df[~threshold]
print(f"Shape after removing >2 NaN rows: {df.shape}")

# Remove any remaining rows with missing values
missing = df.isna().any(axis=1).sum()
print(f"Rows with any missing values: {missing}")

df = df.dropna()
print(f"Final shape after removing all NaNs: {df.shape}")

print("\n" + "="*80)

# Q4: Remove outliers using IQR method
print("\nQ4: Outlier detection and removal (IQR method)")
print("-" * 80)

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {numeric_cols.tolist()}")

start_rows = df.shape[0]

outlier_rows = set()
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    outliers = (df[col] < lower) | (df[col] > upper)
    outlier_idx = df[outliers].index.tolist()
    
    print(f"  {col}: [{lower:.2f}, {upper:.2f}] - {outliers.sum()} outliers")
    outlier_rows.update(outlier_idx)

df = df.drop(index=outlier_rows)
removed = start_rows - df.shape[0]
print(f"\nTotal rows removed: {removed}")
print(f"Final shape: {df.shape}")

print("\n" + "="*80)
print("\nFinal Summary")
print("="*80)
print(f"Cleaned dataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nSummary statistics:")
print(df.describe())

# Save cleaned data
output_file = os.path.join(path, 'Wine_cleaned.csv')
df.to_csv(output_file, index=False)
print(f"\nSaved to: Wine_cleaned.csv")
print("="*80)
