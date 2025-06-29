"""
===============================================================================
SEASONALITY DECOMPOSITION TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Advanced seasonality analysis and decomposition techniques

This template covers:
- Classical decomposition (additive/multiplicative)
- STL (Seasonal and Trend decomposition using Loess)
- X-13ARIMA-SEATS decomposition
- Fourier analysis for seasonality
- Multiple seasonal patterns detection
- Seasonal strength measurement

Prerequisites:
- pandas, numpy, matplotlib, seaborn, statsmodels, scipy
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.x13 import x13_arima_analysis
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# LOAD AND PREPARE TIME SERIES DATA
# ===============================================================================

# Load your time series dataset
# df = pd.read_csv('your_timeseries_data.csv', parse_dates=['date'], index_col='date')

# Sample time series creation with multiple seasonal patterns
np.random.seed(42)
dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
n_points = len(dates)

# Create synthetic time series with multiple seasonal components
trend = np.linspace(100, 300, n_points)  # Linear trend
annual_seasonal = 30 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)  # Annual
weekly_seasonal = 15 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Weekly
monthly_seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 30.44)  # Monthly
noise = np.random.normal(0, 8, n_points)

# Combine components
ts_values = trend + annual_seasonal + weekly_seasonal + monthly_seasonal + noise

# Create DataFrame
df = pd.DataFrame({
    'value': ts_values,
    'date': dates
}).set_index('date')

# Store true components for comparison
df['true_trend'] = trend
df['true_annual'] = annual_seasonal
df['true_weekly'] = weekly_seasonal
df['true_monthly'] = monthly_seasonal
df['true_noise'] = noise

print("Time Series Dataset for Seasonality Analysis:")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Sample data:")
print(df[['value']].head(10))

# Plot original time series
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(df.index, df['value'])
plt.title('Original Time Series')
plt.ylabel('Value')

plt.subplot(3, 1, 2)
plt.plot(df.index[-365:], df['value'][-365:])  # Last year
plt.title('Last Year Detail')
plt.ylabel('Value')

plt.subplot(3, 1, 3)
plt.plot(df.index[-30:], df['value'][-30:])  # Last month
plt.title('Last Month Detail')
plt.ylabel('Value')
plt.xlabel('Date')

plt.tight_layout()
plt.show()

# ===============================================================================
# 1. CLASSICAL SEASONAL DECOMPOSITION
# ===============================================================================

print("\n" + "="*60)
print("1. CLASSICAL SEASONAL DECOMPOSITION")
print("="*60)

# Additive decomposition
print("Performing Additive Decomposition...")
decomposition_add = seasonal_decompose(df['value'], model='additive', period=365)

# Multiplicative decomposition
print("Performing Multiplicative Decomposition...")
decomposition_mult = seasonal_decompose(df['value'], model='multiplicative', period=365)

# Plot classical decompositions
fig, axes = plt.subplots(4, 2, figsize=(20, 16))

# Additive decomposition
axes[0, 0].plot(df.index, df['value'])
axes[0, 0].set_title('Original Series')
axes[0, 0].set_ylabel('Value')

axes[1, 0].plot(decomposition_add.trend.index, decomposition_add.trend)
axes[1, 0].set_title('Additive Trend')
axes[1, 0].set_ylabel('Trend')

axes[2, 0].plot(decomposition_add.seasonal.index, decomposition_add.seasonal)
axes[2, 0].set_title('Additive Seasonal')
axes[2, 0].set_ylabel('Seasonal')

axes[3, 0].plot(decomposition_add.resid.index, decomposition_add.resid)
axes[3, 0].set_title('Additive Residual')
axes[3, 0].set_ylabel('Residual')
axes[3, 0].set_xlabel('Date')

# Multiplicative decomposition
axes[0, 1].plot(df.index, df['value'])
axes[0, 1].set_title('Original Series')
axes[0, 1].set_ylabel('Value')

axes[1, 1].plot(decomposition_mult.trend.index, decomposition_mult.trend)
axes[1, 1].set_title('Multiplicative Trend')
axes[1, 1].set_ylabel('Trend')

axes[2, 1].plot(decomposition_mult.seasonal.index, decomposition_mult.seasonal)
axes[2, 1].set_title('Multiplicative Seasonal')
axes[2, 1].set_ylabel('Seasonal')

axes[3, 1].plot(decomposition_mult.resid.index, decomposition_mult.resid)
axes[3, 1].set_title('Multiplicative Residual')
axes[3, 1].set_ylabel('Residual')
axes[3, 1].set_xlabel('Date')

plt.tight_layout()
plt.show()

# Calculate seasonal strength
def seasonal_strength(seasonal_component, residual_component):
    """Calculate seasonal strength as variance ratio"""
    var_seasonal = np.var(seasonal_component.dropna())
    var_residual = np.var(residual_component.dropna())
    return var_seasonal / (var_seasonal + var_residual)

seasonal_strength_add = seasonal_strength(decomposition_add.seasonal, decomposition_add.resid)
seasonal_strength_mult = seasonal_strength(decomposition_mult.seasonal, decomposition_mult.resid)

print(f"Seasonal Strength (Additive): {seasonal_strength_add:.4f}")
print(f"Seasonal Strength (Multiplicative): {seasonal_strength_mult:.4f}")

# ===============================================================================
# 2. STL DECOMPOSITION
# ===============================================================================

print("\n" + "="*60)
print("2. STL (SEASONAL AND TREND DECOMPOSITION USING LOESS)")
print("="*60)

# STL decomposition with different parameters
stl_models = {
    'STL_Default': STL(df['value'], seasonal=13, trend=None, low_pass=None),
    'STL_Robust': STL(df['value'], seasonal=13, robust=True),
    'STL_Short_Season': STL(df['value'], seasonal=7, trend=21),
    'STL_Long_Season': STL(df['value'], seasonal=25, trend=51)
}

stl_results = {}
for name, model in stl_models.items():
    print(f"Fitting {name}...")
    result = model.fit()
    stl_results[name] = result

# Plot STL decompositions
fig, axes = plt.subplots(4, 2, figsize=(20, 16))

# Plot two different STL decompositions
stl_names = list(stl_results.keys())[:2]

for col, stl_name in enumerate(stl_names):
    result = stl_results[stl_name]
    
    axes[0, col].plot(df.index, df['value'])
    axes[0, col].set_title(f'{stl_name} - Original')
    axes[0, col].set_ylabel('Value')
    
    axes[1, col].plot(df.index, result.trend)
    axes[1, col].set_title(f'{stl_name} - Trend')
    axes[1, col].set_ylabel('Trend')
    
    axes[2, col].plot(df.index, result.seasonal)
    axes[2, col].set_title(f'{stl_name} - Seasonal')
    axes[2, col].set_ylabel('Seasonal')
    
    axes[3, col].plot(df.index, result.resid)
    axes[3, col].set_title(f'{stl_name} - Residual')
    axes[3, col].set_ylabel('Residual')
    axes[3, col].set_xlabel('Date')

plt.tight_layout()
plt.show()

# Compare STL seasonal strengths
stl_seasonal_strengths = {}
for name, result in stl_results.items():
    strength = seasonal_strength(result.seasonal, result.resid)
    stl_seasonal_strengths[name] = strength
    print(f"{name} Seasonal Strength: {strength:.4f}")

# ===============================================================================
# 3. FOURIER ANALYSIS FOR SEASONALITY
# ===============================================================================

print("\n" + "="*60)
print("3. FOURIER ANALYSIS FOR SEASONALITY")
print("="*60)

def fourier_analysis(ts, sample_rate=1):
    """Perform Fourier analysis to identify seasonal patterns"""
    # Remove trend
    detrended = signal.detrend(ts.dropna())
    
    # Compute FFT
    fft_values = fft(detrended)
    freqs = fftfreq(len(detrended), d=1/sample_rate)
    
    # Compute power spectral density
    power = np.abs(fft_values) ** 2
    
    # Only positive frequencies
    positive_freq_idx = freqs > 0
    freqs_positive = freqs[positive_freq_idx]
    power_positive = power[positive_freq_idx]
    
    return freqs_positive, power_positive

# Perform Fourier analysis
freqs, power = fourier_analysis(df['value'])

# Convert frequencies to periods (in days)
periods = 1 / freqs
periods = periods[periods > 1]  # Only periods > 1 day
power_filtered = power[:len(periods)]

# Find dominant periods
dominant_indices = np.argsort(power_filtered)[-10:]  # Top 10 periods
dominant_periods = periods[dominant_indices]
dominant_powers = power_filtered[dominant_indices]

print("Top 10 Dominant Periods (days):")
for i, (period, power_val) in enumerate(zip(dominant_periods[::-1], dominant_powers[::-1])):
    print(f"{i+1}. {period:.2f} days (Power: {power_val:.0f})")

# Plot power spectrum
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.loglog(freqs, power)
plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.loglog(periods[periods < 1000], power_filtered[periods < 1000])
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.title('Power vs Period (< 1000 days)')
plt.grid(True)

# Highlight common seasonal periods
plt.subplot(2, 2, 3)
mask = (periods >= 6) & (periods <= 400)
plt.plot(periods[mask], power_filtered[mask])
plt.axvline(x=7, color='red', linestyle='--', label='Weekly (7 days)')
plt.axvline(x=30.44, color='green', linestyle='--', label='Monthly (~30 days)')
plt.axvline(x=365.25, color='blue', linestyle='--', label='Annual (365 days)')
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.title('Seasonal Periods (6-400 days)')
plt.legend()
plt.grid(True)

# Plot dominant frequencies
plt.subplot(2, 2, 4)
plt.bar(range(len(dominant_periods)), dominant_powers[::-1])
plt.xlabel('Rank')
plt.ylabel('Power')
plt.title('Top 10 Dominant Frequencies')
plt.xticks(range(len(dominant_periods)), 
           [f'{p:.1f}d' for p in dominant_periods[::-1]], rotation=45)

plt.tight_layout()
plt.show()

# ===============================================================================
# 4. MULTIPLE SEASONAL PATTERNS DETECTION
# ===============================================================================

print("\n" + "="*60)
print("4. MULTIPLE SEASONAL PATTERNS DETECTION")
print("="*60)

def detect_multiple_seasonalities(ts, min_period=2, max_period=None):
    """Detect multiple seasonal patterns using autocorrelation"""
    if max_period is None:
        max_period = len(ts) // 3
    
    # Calculate autocorrelation for different lags
    lags = range(min_period, min(max_period, len(ts)//2))
    autocorrelations = []
    
    for lag in lags:
        if lag < len(ts):
            shifted = ts.shift(lag)
            corr = ts.corr(shifted)
            autocorrelations.append(corr)
        else:
            autocorrelations.append(0)
    
    autocorrelations = np.array(autocorrelations)
    
    # Find peaks in autocorrelation
    peaks, properties = signal.find_peaks(autocorrelations, height=0.1, distance=5)
    
    seasonal_periods = []
    for peak in peaks:
        period = lags[peak]
        correlation = autocorrelations[peak]
        seasonal_periods.append({'period': period, 'correlation': correlation})
    
    # Sort by correlation strength
    seasonal_periods = sorted(seasonal_periods, key=lambda x: x['correlation'], reverse=True)
    
    return lags, autocorrelations, seasonal_periods

# Detect multiple seasonalities
lags, autocorrs, seasonal_periods = detect_multiple_seasonalities(df['value'], max_period=400)

print("Detected Seasonal Patterns (by autocorrelation):")
for i, pattern in enumerate(seasonal_periods[:10]):
    period = pattern['period']
    corr = pattern['correlation']
    print(f"{i+1}. Period: {period} days, Correlation: {corr:.4f}")

# Plot autocorrelation analysis
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(lags, autocorrs)
plt.xlabel('Lag (days)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function')
plt.grid(True)

# Highlight detected peaks
if seasonal_periods:
    peak_lags = [p['period'] for p in seasonal_periods[:5]]
    peak_corrs = [p['correlation'] for p in seasonal_periods[:5]]
    plt.scatter(peak_lags, peak_corrs, color='red', s=50, zorder=5)

plt.subplot(2, 2, 2)
# Focus on short-term patterns (weekly/monthly)
short_mask = np.array(lags) <= 50
plt.plot(np.array(lags)[short_mask], autocorrs[short_mask])
plt.axvline(x=7, color='red', linestyle='--', alpha=0.7, label='Weekly')
plt.axvline(x=30, color='green', linestyle='--', alpha=0.7, label='Monthly')
plt.xlabel('Lag (days)')
plt.ylabel('Autocorrelation')
plt.title('Short-term Seasonality (≤50 days)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
# Focus on long-term patterns (annual)
long_mask = np.array(lags) >= 300
if np.any(long_mask):
    plt.plot(np.array(lags)[long_mask], autocorrs[long_mask])
    plt.axvline(x=365, color='blue', linestyle='--', alpha=0.7, label='Annual')
    plt.xlabel('Lag (days)')
    plt.ylabel('Autocorrelation')
    plt.title('Long-term Seasonality (≥300 days)')
    plt.legend()
    plt.grid(True)

# Seasonal pattern strength over time
plt.subplot(2, 2, 4)
if len(seasonal_periods) >= 3:
    period_strengths = [p['correlation'] for p in seasonal_periods[:10]]
    period_labels = [f"{p['period']}d" for p in seasonal_periods[:10]]
    
    plt.bar(range(len(period_strengths)), period_strengths)
    plt.xlabel('Seasonal Pattern')
    plt.ylabel('Correlation Strength')
    plt.title('Top Seasonal Patterns by Strength')
    plt.xticks(range(len(period_labels)), period_labels, rotation=45)

plt.tight_layout()
plt.show()

# ===============================================================================
# 5. SEASONAL SUBSERIES PLOTS
# ===============================================================================

print("\n" + "="*60)
print("5. SEASONAL SUBSERIES ANALYSIS")
print("="*60)

def create_seasonal_subseries(ts, period):
    """Create seasonal subseries for analysis"""
    # Create seasonal groups
    seasonal_groups = []
    for i in range(period):
        indices = range(i, len(ts), period)
        subseries = ts.iloc[indices]
        seasonal_groups.append(subseries.values)
    
    return seasonal_groups

# Create seasonal subseries for different periods
periods_to_analyze = [7, 30, 365]  # Weekly, monthly, annual

plt.figure(figsize=(18, 12))

for idx, period in enumerate(periods_to_analyze):
    if period < len(df):
        seasonal_groups = create_seasonal_subseries(df['value'], period)
        
        plt.subplot(2, 3, idx + 1)
        for i, group in enumerate(seasonal_groups):
            if len(group) > 0:
                plt.plot(group, alpha=0.7, label=f'Season {i+1}')
        
        plt.title(f'Seasonal Subseries (Period={period} days)')
        plt.xlabel('Cycle Number')
        plt.ylabel('Value')
        
        # Box plot version
        plt.subplot(2, 3, idx + 4)
        valid_groups = [group for group in seasonal_groups if len(group) > 1]
        if valid_groups:
            plt.boxplot(valid_groups, labels=range(1, len(valid_groups) + 1))
            plt.title(f'Seasonal Boxplots (Period={period} days)')
            plt.xlabel('Season')
            plt.ylabel('Value')

plt.tight_layout()
plt.show()

# ===============================================================================
# 6. SEASONAL STRENGTH MEASUREMENT
# ===============================================================================

print("\n" + "="*60)
print("6. SEASONAL STRENGTH MEASUREMENT")
print("="*60)

def calculate_seasonal_strength_metrics(ts, decomposition_result):
    """Calculate various seasonal strength metrics"""
    seasonal = decomposition_result.seasonal
    residual = decomposition_result.resid
    
    # Variance-based strength
    var_seasonal = np.var(seasonal.dropna())
    var_residual = np.var(residual.dropna())
    strength_variance = var_seasonal / (var_seasonal + var_residual)
    
    # Range-based strength
    range_seasonal = np.ptp(seasonal.dropna())  # peak-to-peak
    range_residual = np.ptp(residual.dropna())
    strength_range = range_seasonal / (range_seasonal + range_residual)
    
    # Autocorrelation-based strength
    seasonal_period = len(seasonal.dropna()) // 2  # Approximate period
    if seasonal_period > 1 and seasonal_period < len(ts):
        seasonal_autocorr = ts.autocorr(lag=seasonal_period)
        strength_autocorr = max(0, seasonal_autocorr)
    else:
        strength_autocorr = 0
    
    return {
        'variance_strength': strength_variance,
        'range_strength': strength_range,
        'autocorr_strength': strength_autocorr
    }

# Calculate seasonal strength for different decompositions
decomposition_methods = {
    'Classical_Additive': decomposition_add,
    'Classical_Multiplicative': decomposition_mult,
    'STL_Default': stl_results['STL_Default'],
    'STL_Robust': stl_results['STL_Robust']
}

strength_comparison = {}
for method_name, decomp in decomposition_methods.items():
    strengths = calculate_seasonal_strength_metrics(df['value'], decomp)
    strength_comparison[method_name] = strengths
    
    print(f"\n{method_name}:")
    print(f"  Variance-based strength: {strengths['variance_strength']:.4f}")
    print(f"  Range-based strength: {strengths['range_strength']:.4f}")
    print(f"  Autocorr-based strength: {strengths['autocorr_strength']:.4f}")

# Visualize strength comparison
strength_df = pd.DataFrame(strength_comparison).T

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
strength_df['variance_strength'].plot(kind='bar')
plt.title('Variance-based Seasonal Strength')
plt.ylabel('Strength')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
strength_df['range_strength'].plot(kind='bar')
plt.title('Range-based Seasonal Strength')
plt.ylabel('Strength')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
strength_df['autocorr_strength'].plot(kind='bar')
plt.title('Autocorrelation-based Seasonal Strength')
plt.ylabel('Strength')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
# Radar chart comparison
angles = np.linspace(0, 2 * np.pi, len(strength_df.columns), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

for idx, (method, values) in enumerate(strength_df.iterrows()):
    values_plot = list(values) + [values[0]]
    plt.polar(angles, values_plot, 'o-', linewidth=2, label=method)

plt.title('Seasonal Strength Comparison (Radar Chart)')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.show()

# ===============================================================================
# 7. COMPARISON WITH TRUE COMPONENTS
# ===============================================================================

print("\n" + "="*60)
print("7. COMPARISON WITH TRUE COMPONENTS")
print("="*60)

# Compare decomposed components with true components
best_stl = stl_results['STL_Default']

# Calculate correlation with true components
trend_corr = best_stl.trend.corr(df['true_trend'])
seasonal_corr = best_stl.seasonal.corr(df['true_annual'])  # Compare with annual
residual_corr = best_stl.resid.corr(df['true_noise'])

print("Correlation with True Components (STL Default):")
print(f"Trend correlation: {trend_corr:.4f}")
print(f"Seasonal correlation (annual): {seasonal_corr:.4f}")
print(f"Residual correlation: {residual_corr:.4f}")

# Plot comparison
plt.figure(figsize=(15, 12))

plt.subplot(3, 2, 1)
plt.plot(df.index, df['true_trend'], label='True Trend', alpha=0.8)
plt.plot(df.index, best_stl.trend, label='STL Trend', alpha=0.8)
plt.title('Trend Comparison')
plt.legend()
plt.ylabel('Value')

plt.subplot(3, 2, 2)
plt.plot(df.index[:365], df['true_annual'][:365], label='True Annual', alpha=0.8)
plt.plot(df.index[:365], best_stl.seasonal[:365], label='STL Seasonal', alpha=0.8)
plt.title('Seasonal Comparison (First Year)')
plt.legend()
plt.ylabel('Value')

plt.subplot(3, 2, 3)
plt.plot(df.index, df['true_noise'], label='True Noise', alpha=0.8)
plt.plot(df.index, best_stl.resid, label='STL Residual', alpha=0.8)
plt.title('Residual Comparison')
plt.legend()
plt.ylabel('Value')

plt.subplot(3, 2, 4)
plt.scatter(df['true_trend'], best_stl.trend, alpha=0.6)
plt.xlabel('True Trend')
plt.ylabel('STL Trend')
plt.title(f'Trend Scatter (r={trend_corr:.3f})')
plt.plot([df['true_trend'].min(), df['true_trend'].max()], 
         [df['true_trend'].min(), df['true_trend'].max()], 'r--')

plt.subplot(3, 2, 5)
plt.scatter(df['true_annual'], best_stl.seasonal, alpha=0.6)
plt.xlabel('True Annual Seasonal')
plt.ylabel('STL Seasonal')
plt.title(f'Seasonal Scatter (r={seasonal_corr:.3f})')
plt.plot([df['true_annual'].min(), df['true_annual'].max()], 
         [df['true_annual'].min(), df['true_annual'].max()], 'r--')

plt.subplot(3, 2, 6)
plt.scatter(df['true_noise'], best_stl.resid, alpha=0.6)
plt.xlabel('True Noise')
plt.ylabel('STL Residual')
plt.title(f'Residual Scatter (r={residual_corr:.3f})')
plt.plot([df['true_noise'].min(), df['true_noise'].max()], 
         [df['true_noise'].min(), df['true_noise'].max()], 'r--')

plt.tight_layout()
plt.show()

# ===============================================================================
# 8. SEASONALITY SUMMARY AND RECOMMENDATIONS
# ===============================================================================

print("\n" + "="*60)
print("8. SEASONALITY SUMMARY AND RECOMMENDATIONS")
print("="*60)

print("SEASONALITY ANALYSIS SUMMARY:")
print("=" * 35)

# Identify strongest seasonal patterns
if seasonal_periods:
    strongest_pattern = seasonal_periods[0]
    print(f"Strongest Seasonal Pattern: {strongest_pattern['period']} days (correlation: {strongest_pattern['correlation']:.4f})")

# Best decomposition method
best_method = max(strength_comparison.keys(), 
                 key=lambda x: strength_comparison[x]['variance_strength'])
best_strength = strength_comparison[best_method]['variance_strength']
print(f"Best Decomposition Method: {best_method} (strength: {best_strength:.4f})")

# Dominant periods
print(f"\nTop 5 Dominant Periods:")
for i, pattern in enumerate(seasonal_periods[:5]):
    period = pattern['period']
    if period <= 10:
        period_name = f"{period} days (weekly pattern)"
    elif 25 <= period <= 35:
        period_name = f"{period} days (monthly pattern)"
    elif 360 <= period <= 370:
        period_name = f"{period} days (annual pattern)"
    else:
        period_name = f"{period} days"
    
    print(f"  {i+1}. {period_name} - correlation: {pattern['correlation']:.4f}")

print(f"\nRECOMMENDATIONS:")
print("=" * 20)
print("✓ Use STL decomposition for robust seasonal extraction")
print("✓ Consider multiple seasonal patterns in forecasting models")
print("✓ Monitor seasonal strength changes over time")
print("✓ Apply appropriate seasonal adjustments before modeling")
if best_strength > 0.7:
    print("✓ Strong seasonality detected - use seasonal models")
elif best_strength > 0.3:
    print("✓ Moderate seasonality - consider seasonal adjustments")
else:
    print("✓ Weak seasonality - focus on trend modeling")

# Export decomposition results
decomposition_df = pd.DataFrame({
    'Date': df.index,
    'Original': df['value'],
    'STL_Trend': best_stl.trend,
    'STL_Seasonal': best_stl.seasonal,
    'STL_Residual': best_stl.resid,
    'Classical_Add_Trend': decomposition_add.trend,
    'Classical_Add_Seasonal': decomposition_add.seasonal,
    'Classical_Add_Residual': decomposition_add.resid
})

# decomposition_df.to_csv('seasonality_decomposition_results.csv', index=False)
print(f"\nSeasonality analysis complete!")
print(f"Decomposition results ready for export!")
print(f"Multiple seasonal patterns identified and analyzed!")
