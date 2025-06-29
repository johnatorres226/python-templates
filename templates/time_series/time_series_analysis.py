# ==============================================================================
# TIME SERIES ANALYSIS TEMPLATE
# ==============================================================================
# Purpose: Comprehensive time series analysis with trend, seasonality, and stationarity
# Replace 'your_data.csv' with your dataset
# Update 'date_column' and 'value_column' with your actual column names
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

print("Time Series Analysis")
print("="*50)

# ================================
# 1. DATA PREPARATION
# ================================
print("1. Data Preparation")
print("-" * 20)

# Replace 'date_column' and 'value_column' with your actual column names
date_column = 'date_column'
value_column = 'value_column'

# Convert date column to datetime
df[date_column] = pd.to_datetime(df[date_column])

# Sort by date
df = df.sort_values(date_column).reset_index(drop=True)

# Set date as index
df.set_index(date_column, inplace=True)

# Remove any missing values
df = df.dropna()

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Frequency: {pd.infer_freq(df.index)}")

# Basic statistics
print(f"\nBasic Statistics for {value_column}:")
print(df[value_column].describe())

# ================================
# 2. TIME SERIES VISUALIZATION
# ================================
print("\n2. Time Series Visualization")
print("-" * 20)

# Create comprehensive time series plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Time Series Overview', fontsize=16, fontweight='bold')

# Plot 1: Full time series
axes[0, 0].plot(df.index, df[value_column], linewidth=1)
axes[0, 0].set_title('Complete Time Series')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution
axes[0, 1].hist(df[value_column], bins=30, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Value Distribution')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Box plot by year (if data spans multiple years)
df['year'] = df.index.year
if df['year'].nunique() > 1:
    df.boxplot(column=value_column, by='year', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution by Year')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Value')
else:
    axes[1, 0].plot(df.index, df[value_column])
    axes[1, 0].set_title('Time Series (Single Year)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Value')

# Plot 4: Monthly patterns (if applicable)
df['month'] = df.index.month
monthly_avg = df.groupby('month')[value_column].mean()
axes[1, 1].plot(monthly_avg.index, monthly_avg.values, 'o-')
axes[1, 1].set_title('Average by Month')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Average Value')
axes[1, 1].set_xticks(range(1, 13))
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# 3. STATIONARITY TESTING
# ================================
print("\n3. Stationarity Testing")
print("-" * 20)

def check_stationarity(timeseries, title):
    """Perform stationarity tests"""
    print(f"\n{title}")
    print("-" * len(title))
    
    # Augmented Dickey-Fuller test
    print("Augmented Dickey-Fuller Test:")
    adf_result = adfuller(timeseries)
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.3f}")
    
    if adf_result[1] <= 0.05:
        print("✓ Series is stationary (ADF test)")
    else:
        print("✗ Series is non-stationary (ADF test)")
    
    # KPSS test
    print("\nKPSS Test:")
    kpss_result = kpss(timeseries, regression='c')
    print(f"KPSS Statistic: {kpss_result[0]:.6f}")
    print(f"p-value: {kpss_result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"\t{key}: {value:.3f}")
    
    if kpss_result[1] >= 0.05:
        print("✓ Series is stationary (KPSS test)")
    else:
        print("✗ Series is non-stationary (KPSS test)")
    
    return adf_result[1] <= 0.05 and kpss_result[1] >= 0.05

# Test original series
is_stationary = check_stationarity(df[value_column], "Original Series Stationarity")

# ================================
# 4. SEASONAL DECOMPOSITION
# ================================
print("\n4. Seasonal Decomposition")
print("-" * 20)

# Determine appropriate period for decomposition
freq = pd.infer_freq(df.index)
if freq:
    if 'D' in freq:  # Daily data
        period = 365  # Annual seasonality
    elif 'M' in freq:  # Monthly data
        period = 12   # Annual seasonality
    elif 'Q' in freq:  # Quarterly data
        period = 4    # Annual seasonality
    else:
        period = min(len(df) // 2, 12)  # Default
else:
    period = min(len(df) // 2, 12)

print(f"Using decomposition period: {period}")

# Perform seasonal decomposition
try:
    decomposition = seasonal_decompose(df[value_column], model='additive', period=period)
    
    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Seasonal Decomposition (Additive)', fontsize=16, fontweight='bold')
    
    decomposition.observed.plot(ax=axes[0], title='Original')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate strength of trend and seasonality
    trend_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend.dropna() + decomposition.resid.dropna())
    seasonal_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())
    
    print(f"Trend strength: {trend_strength:.3f}")
    print(f"Seasonal strength: {seasonal_strength:.3f}")
    
    if trend_strength > 0.6:
        print("✓ Strong trend component detected")
    if seasonal_strength > 0.6:
        print("✓ Strong seasonal component detected")
        
except Exception as e:
    print(f"Decomposition failed: {e}")
    decomposition = None

# ================================
# 5. AUTOCORRELATION ANALYSIS
# ================================
print("\n5. Autocorrelation Analysis")
print("-" * 20)

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Autocorrelation Analysis', fontsize=16, fontweight='bold')

# ACF plot
plot_acf(df[value_column], lags=min(40, len(df)//4), ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
axes[0].grid(True, alpha=0.3)

# PACF plot
plot_pacf(df[value_column], lags=min(40, len(df)//4), ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Ljung-Box test for autocorrelation
lb_test = acorr_ljungbox(df[value_column], lags=min(10, len(df)//10), return_df=True)
print("Ljung-Box Test for Autocorrelation:")
print(lb_test)

significant_lags = lb_test[lb_test['lb_pvalue'] < 0.05]
if len(significant_lags) > 0:
    print(f"✗ Significant autocorrelation detected at lags: {list(significant_lags.index)}")
else:
    print("✓ No significant autocorrelation detected")

# ================================
# 6. TREND ANALYSIS
# ================================
print("\n6. Trend Analysis")
print("-" * 20)

# Linear trend
from scipy.stats import linregress

# Create numeric time index for trend analysis
time_index = np.arange(len(df))
slope, intercept, r_value, p_value, std_err = linregress(time_index, df[value_column])

print(f"Linear Trend Analysis:")
print(f"Slope: {slope:.6f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    if slope > 0:
        print("✓ Significant positive trend detected")
    else:
        print("✓ Significant negative trend detected")
else:
    print("✗ No significant linear trend")

# Plot with trend line
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[value_column], label='Original Data', alpha=0.7)

# Add trend line
trend_line = intercept + slope * time_index
plt.plot(df.index, trend_line, color='red', linestyle='--', 
         label=f'Linear Trend (slope={slope:.4f})')

plt.title('Time Series with Linear Trend')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================
# 7. ROLLING STATISTICS
# ================================
print("\n7. Rolling Statistics Analysis")
print("-" * 20)

# Calculate rolling statistics
window_size = min(30, len(df) // 4)  # Adaptive window size
rolling_mean = df[value_column].rolling(window=window_size).mean()
rolling_std = df[value_column].rolling(window=window_size).std()

# Plot rolling statistics
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'Rolling Statistics (Window = {window_size})', fontsize=16, fontweight='bold')

# Rolling mean
axes[0].plot(df.index, df[value_column], label='Original', alpha=0.7)
axes[0].plot(df.index, rolling_mean, color='red', label='Rolling Mean')
axes[0].set_title('Rolling Mean')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Rolling standard deviation
axes[1].plot(df.index, rolling_std, color='orange', label='Rolling Std')
axes[1].set_title('Rolling Standard Deviation')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Standard Deviation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test for changing variance
first_half_var = df[value_column][:len(df)//2].var()
second_half_var = df[value_column][len(df)//2:].var()
variance_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)

print(f"Variance Analysis:")
print(f"First half variance: {first_half_var:.4f}")
print(f"Second half variance: {second_half_var:.4f}")
print(f"Variance ratio: {variance_ratio:.4f}")

if variance_ratio > 2:
    print("⚠ Significant change in variance detected (heteroscedasticity)")
else:
    print("✓ Variance appears stable")

# ================================
# 8. ANOMALY DETECTION
# ================================
print("\n8. Anomaly Detection")
print("-" * 20)

# Statistical outlier detection
Q1 = df[value_column].quantile(0.25)
Q3 = df[value_column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[value_column] < lower_bound) | (df[value_column] > upper_bound)]
print(f"IQR-based outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# Z-score based outliers
z_scores = np.abs(stats.zscore(df[value_column]))
z_outliers = df[z_scores > 3]
print(f"Z-score outliers (|z| > 3): {len(z_outliers)} ({len(z_outliers)/len(df)*100:.1f}%)")

# Plot outliers
plt.figure(figsize=(14, 6))
plt.plot(df.index, df[value_column], label='Time Series', alpha=0.7)
if len(outliers) > 0:
    plt.scatter(outliers.index, outliers[value_column], 
               color='red', s=50, label=f'Outliers ({len(outliers)})')
plt.axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.7, label='IQR Bounds')
plt.axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7)
plt.title('Time Series with Outlier Detection')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================
# 9. PERIODICITY ANALYSIS
# ================================
print("\n9. Periodicity Analysis")
print("-" * 20)

# Fast Fourier Transform for frequency analysis
from scipy.fft import fft, fftfreq

# Perform FFT
fft_values = fft(df[value_column].values)
fft_freq = fftfreq(len(df), d=1)

# Find dominant frequencies
power_spectrum = np.abs(fft_values)**2
dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies

print("Dominant Frequencies:")
for i, idx in enumerate(dominant_freq_idx[::-1]):
    if fft_freq[idx] > 0:  # Positive frequencies only
        period = 1 / fft_freq[idx] if fft_freq[idx] != 0 else np.inf
        print(f"{i+1}. Frequency: {fft_freq[idx]:.6f}, Period: {period:.2f} observations")

# Plot power spectrum
plt.figure(figsize=(12, 6))
plt.plot(fft_freq[:len(fft_freq)//2], power_spectrum[:len(power_spectrum)//2])
plt.title('Power Spectrum (Frequency Analysis)')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================
# 10. SUMMARY REPORT
# ================================
print("\n10. Summary Report")
print("-" * 20)

print("TIME SERIES CHARACTERISTICS:")
print(f"• Data points: {len(df)}")
print(f"• Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"• Frequency: {freq or 'Unknown'}")
print(f"• Missing values: {df[value_column].isnull().sum()}")

print(f"\nSTATISTICAL PROPERTIES:")
print(f"• Mean: {df[value_column].mean():.4f}")
print(f"• Standard deviation: {df[value_column].std():.4f}")
print(f"• Skewness: {stats.skew(df[value_column]):.4f}")
print(f"• Kurtosis: {stats.kurtosis(df[value_column]):.4f}")

print(f"\nTIME SERIES PROPERTIES:")
print(f"• Stationary: {'Yes' if is_stationary else 'No'}")
print(f"• Linear trend: {'Significant' if p_value < 0.05 else 'Not significant'}")
if decomposition:
    print(f"• Trend strength: {trend_strength:.3f}")
    print(f"• Seasonal strength: {seasonal_strength:.3f}")
print(f"• Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

print(f"\nRECOMMendations:")
if not is_stationary:
    print("• Consider differencing to achieve stationarity")
if seasonal_strength > 0.6:
    print("• Strong seasonality detected - consider seasonal models")
if trend_strength > 0.6:
    print("• Strong trend detected - detrending may be necessary")
if len(outliers) > len(df) * 0.05:
    print("• High number of outliers - investigate data quality")

print("\nTime series analysis completed!")
print("Remember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update 'date_column' and 'value_column' names")
print("3. Adjust decomposition period based on your data frequency")
print("4. Consider the recommendations for modeling approach")
