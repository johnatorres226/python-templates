# Replace 'your_data.csv' with your dataset
# Outlier Detection Template - Z-score, IQR, and Isolation Forest Methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== OUTLIER DETECTION ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Get numerical columns only
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns for outlier detection: {numerical_cols}")

# 1. Z-SCORE METHOD
print("\n=== Z-SCORE OUTLIER DETECTION ===")
# Outliers defined as |z-score| > 3 (or 2.5 for more conservative)

z_threshold = 3  # Adjust as needed (2.5 for more conservative, 3.5 for less)
outliers_z = {}

df_z_outliers = df.copy()
df_z_outliers['is_outlier_zscore'] = False

for col in numerical_cols:
    if df[col].isnull().sum() == 0:  # Only process columns without missing values
        # Calculate z-scores
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outlier_mask = z_scores > z_threshold
        
        outliers_z[col] = {
            'count': outlier_mask.sum(),
            'percentage': (outlier_mask.sum() / len(df)) * 100,
            'threshold_value': z_threshold,
            'outlier_indices': df[outlier_mask].index.tolist()
        }
        
        # Mark outliers in dataframe
        df_z_outliers.loc[outlier_mask, 'is_outlier_zscore'] = True
        df_z_outliers[f'{col}_zscore'] = z_scores
        
        print(f"{col}: {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(df)*100:.2f}%)")
        if outlier_mask.sum() > 0:
            print(f"  Outlier range: {df[outlier_mask][col].min():.2f} to {df[outlier_mask][col].max():.2f}")
            print(f"  Normal range: {df[~outlier_mask][col].min():.2f} to {df[~outlier_mask][col].max():.2f}")

# 2. INTERQUARTILE RANGE (IQR) METHOD
print("\n=== IQR OUTLIER DETECTION ===")
# Outliers defined as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

outliers_iqr = {}
df_iqr_outliers = df.copy()
df_iqr_outliers['is_outlier_iqr'] = False

for col in numerical_cols:
    if df[col].isnull().sum() == 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        outliers_iqr[col] = {
            'count': outlier_mask.sum(),
            'percentage': (outlier_mask.sum() / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'outlier_indices': df[outlier_mask].index.tolist()
        }
        
        # Mark outliers in dataframe
        df_iqr_outliers.loc[outlier_mask, 'is_outlier_iqr'] = True
        
        print(f"{col}: {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(df)*100:.2f}%)")
        print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        if outlier_mask.sum() > 0:
            print(f"  Outlier range: {df[outlier_mask][col].min():.2f} to {df[outlier_mask][col].max():.2f}")

# 3. ISOLATION FOREST METHOD (for multivariate outliers)
print("\n=== ISOLATION FOREST OUTLIER DETECTION ===")

# Prepare data (remove missing values and scale)
df_clean = df[numerical_cols].dropna()

if len(df_clean) > 0:
    # Scale the data for better isolation forest performance
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    # Apply Isolation Forest
    # contamination = expected proportion of outliers (adjust based on domain knowledge)
    contamination_rate = 0.1  # Assume 10% outliers
    
    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
    outlier_labels = iso_forest.fit_predict(df_scaled)
    
    # -1 indicates outlier, 1 indicates normal
    outlier_mask_iso = outlier_labels == -1
    
    print(f"Isolation Forest detected {outlier_mask_iso.sum()} outliers ({outlier_mask_iso.sum()/len(df_clean)*100:.2f}%)")
    
    # Add results back to original dataframe
    df_iso_outliers = df.copy()
    df_iso_outliers['is_outlier_isolation'] = False
    df_iso_outliers.loc[df_clean.index[outlier_mask_iso], 'is_outlier_isolation'] = True
    
    # Get outlier scores (lower scores indicate more anomalous)
    outlier_scores = iso_forest.decision_function(df_scaled)
    df_temp = df_clean.copy()
    df_temp['isolation_score'] = outlier_scores
    
    print(f"Outlier score range: {outlier_scores.min():.3f} to {outlier_scores.max():.3f}")
    print(f"Outlier threshold: {iso_forest.offset_:.3f}")

# 4. COMBINED OUTLIER ANALYSIS
print("\n=== COMBINED OUTLIER ANALYSIS ===")

# Find rows that are outliers by multiple methods
if len(numerical_cols) > 0:
    combined_analysis = df.copy()
    combined_analysis['outlier_count'] = 0
    combined_analysis['outlier_methods'] = ''
    
    # Z-score outliers
    z_outlier_rows = df_z_outliers['is_outlier_zscore']
    combined_analysis.loc[z_outlier_rows, 'outlier_count'] += 1
    combined_analysis.loc[z_outlier_rows, 'outlier_methods'] += 'Z-score,'
    
    # IQR outliers
    iqr_outlier_rows = df_iqr_outliers['is_outlier_iqr']
    combined_analysis.loc[iqr_outlier_rows, 'outlier_count'] += 1
    combined_analysis.loc[iqr_outlier_rows, 'outlier_methods'] += 'IQR,'
    
    # Isolation Forest outliers (if available)
    if 'df_iso_outliers' in locals():
        iso_outlier_rows = df_iso_outliers['is_outlier_isolation']
        combined_analysis.loc[iso_outlier_rows, 'outlier_count'] += 1
        combined_analysis.loc[iso_outlier_rows, 'outlier_methods'] += 'Isolation,'
    
    # Clean up methods string
    combined_analysis['outlier_methods'] = combined_analysis['outlier_methods'].str.rstrip(',')
    
    # Summary of combined results
    outlier_summary = combined_analysis['outlier_count'].value_counts().sort_index()
    print("Outliers detected by number of methods:")
    for count, freq in outlier_summary.items():
        if count > 0:
            print(f"  {count} method(s): {freq} rows ({freq/len(df)*100:.2f}%)")
    
    # High-confidence outliers (detected by 2+ methods)
    high_confidence_outliers = combined_analysis[combined_analysis['outlier_count'] >= 2]
    print(f"\nHigh-confidence outliers (2+ methods): {len(high_confidence_outliers)} rows")
    
    if len(high_confidence_outliers) > 0:
        print("Sample high-confidence outliers:")
        print(high_confidence_outliers[numerical_cols + ['outlier_count', 'outlier_methods']].head())

# 5. VISUALIZATION
print("\n=== OUTLIER VISUALIZATIONS ===")

# Create visualizations for first few numerical columns
n_plots = min(4, len(numerical_cols))
if n_plots > 0:
    fig, axes = plt.subplots(n_plots, 3, figsize=(15, 4*n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numerical_cols[:n_plots]):
        # Box plot
        axes[i, 0].boxplot(df[col].dropna())
        axes[i, 0].set_title(f'{col} - Box Plot')
        axes[i, 0].set_ylabel('Value')
        
        # Histogram
        axes[i, 1].hist(df[col].dropna(), bins=30, alpha=0.7)
        axes[i, 1].set_title(f'{col} - Histogram')
        axes[i, 1].set_xlabel('Value')
        axes[i, 1].set_ylabel('Frequency')
        
        # Scatter plot with outlier highlighting (using z-score)
        if col in outliers_z:
            normal_mask = ~df_z_outliers['is_outlier_zscore']
            outlier_mask = df_z_outliers['is_outlier_zscore']
            
            axes[i, 2].scatter(df[normal_mask].index, df[normal_mask][col], 
                             alpha=0.6, label='Normal', s=20)
            if outlier_mask.sum() > 0:
                axes[i, 2].scatter(df[outlier_mask].index, df[outlier_mask][col], 
                                 color='red', label='Z-score Outlier', s=30)
            axes[i, 2].set_title(f'{col} - Outliers Highlighted')
            axes[i, 2].set_xlabel('Index')
            axes[i, 2].set_ylabel('Value')
            axes[i, 2].legend()
    
    plt.tight_layout()
    plt.show()

# 6. OUTLIER TREATMENT RECOMMENDATIONS
print("\n=== OUTLIER TREATMENT RECOMMENDATIONS ===")

for col in numerical_cols:
    if col in outliers_z and col in outliers_iqr:
        z_pct = outliers_z[col]['percentage']
        iqr_pct = outliers_iqr[col]['percentage']
        
        print(f"\n{col}:")
        print(f"  Z-score outliers: {z_pct:.2f}%")
        print(f"  IQR outliers: {iqr_pct:.2f}%")
        
        # Recommendations based on outlier percentage
        if max(z_pct, iqr_pct) < 1:
            print(f"  Recommendation: FEW outliers - Consider keeping or light trimming")
        elif max(z_pct, iqr_pct) < 5:
            print(f"  Recommendation: MODERATE outliers - Investigate and consider capping")
        else:
            print(f"  Recommendation: MANY outliers - Check data quality, consider transformation")

print("\n=== TREATMENT OPTIONS ===")
print("1. REMOVE: Delete outlier rows (use for clear data errors)")
print("2. CAP/WINSORIZE: Replace with percentile values (e.g., 95th/5th percentile)")
print("3. TRANSFORM: Apply log, sqrt, or Box-Cox transformation")
print("4. SEPARATE: Analyze outliers separately as special cases")
print("5. ROBUST METHODS: Use median, MAD instead of mean, std")

# Example treatment code (uncomment and modify as needed):

# Remove outliers (be careful with this!)
# df_no_outliers = df[~df_z_outliers['is_outlier_zscore']]

# Cap outliers at 95th/5th percentiles
# for col in numerical_cols:
#     p95 = df[col].quantile(0.95)
#     p05 = df[col].quantile(0.05)
#     df[col] = df[col].clip(lower=p05, upper=p95)

# Log transformation for right-skewed data with outliers
# for col in numerical_cols:
#     if df[col].min() > 0:  # Only for positive values
#         df[f'{col}_log'] = np.log(df[col])

print("\nOutlier detection complete. Review results and choose appropriate treatment.")
