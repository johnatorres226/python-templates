# ==============================================================================
# LINE PLOTS TEMPLATE
# ==============================================================================
# Purpose: Create professional line plots with multiple series and trendlines
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

# ================================
# 1. BASIC LINE PLOT
# ================================
# Replace 'date_column' and 'value_column' with your actual column names
plt.figure(figsize=(12, 6))
plt.plot(df['date_column'], df['value_column'], linewidth=2, marker='o', markersize=4)
plt.title('Time Series Plot', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# 2. MULTIPLE SERIES LINE PLOT
# ================================
# For plotting multiple variables over time
# Replace column names as needed
plt.figure(figsize=(14, 8))

# Plot multiple lines
# Replace 'series1', 'series2', 'series3' with your actual column names
plt.plot(df['date_column'], df['series1'], label='Series 1', linewidth=2, marker='o')
plt.plot(df['date_column'], df['series2'], label='Series 2', linewidth=2, marker='s')
plt.plot(df['date_column'], df['series3'], label='Series 3', linewidth=2, marker='^')

plt.title('Multiple Time Series Comparison', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# 3. LINE PLOT WITH TREND LINE
# ================================
# Add linear trend line to your data
plt.figure(figsize=(12, 8))

# Create numeric x-values for trend calculation
x_numeric = range(len(df))
y_values = df['value_column']  # Replace with your column

# Plot original data
plt.plot(df['date_column'], y_values, 'o-', label='Data', linewidth=2, markersize=6)

# Calculate and plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
trend_line = slope * np.array(x_numeric) + intercept
plt.plot(df['date_column'], trend_line, '--', color='red', linewidth=2, 
         label=f'Trend (R² = {r_value**2:.3f})')

plt.title('Time Series with Trend Line', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Trend Analysis:")
print(f"Slope: {slope:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.4f}")

# ================================
# 4. SUBPLOT LINE PLOTS
# ================================
# Create multiple subplots for comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Multiple Line Plot Dashboard', fontsize=18, fontweight='bold')

# Subplot 1: Raw data
axes[0, 0].plot(df['date_column'], df['value_column'], 'b-', linewidth=2)
axes[0, 0].set_title('Raw Data', fontsize=14)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Moving average (replace 7 with your desired window)
window_size = 7
moving_avg = df['value_column'].rolling(window=window_size).mean()
axes[0, 1].plot(df['date_column'], df['value_column'], 'lightgray', alpha=0.7, label='Raw')
axes[0, 1].plot(df['date_column'], moving_avg, 'r-', linewidth=2, label=f'{window_size}-period MA')
axes[0, 1].set_title(f'{window_size}-Period Moving Average', fontsize=14)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Cumulative sum
cumulative = df['value_column'].cumsum()
axes[1, 0].plot(df['date_column'], cumulative, 'g-', linewidth=2)
axes[1, 0].set_title('Cumulative Sum', fontsize=14)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Cumulative Value')
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Percentage change (replace with your period)
pct_change = df['value_column'].pct_change() * 100
axes[1, 1].plot(df['date_column'], pct_change, 'orange', linewidth=2)
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Percentage Change', fontsize=14)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('% Change')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# 5. GROUPED LINE PLOT
# ================================
# Plot lines grouped by a categorical variable
# Replace 'category_column' with your grouping column
plt.figure(figsize=(14, 8))

# Get unique categories
categories = df['category_column'].unique()

# Plot line for each category
for category in categories:
    subset = df[df['category_column'] == category]
    plt.plot(subset['date_column'], subset['value_column'], 
             label=category, linewidth=2, marker='o', markersize=4)

plt.title('Line Plots by Category', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Category', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# CUSTOMIZATION TIPS
# ================================
# Line styles: '-', '--', '-.', ':'
# Markers: 'o', 's', '^', 'v', 'D', 'p', '*'
# Colors: 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'
# Or use hex codes: '#FF5733', '#33FF57'

print("Line plot templates completed!")
print("Remember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update column names to match your data")
print("3. Adjust date parsing if needed (pd.to_datetime)")
print("4. Customize colors, markers, and styles as desired")
