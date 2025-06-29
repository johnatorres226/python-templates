# ==============================================================================
# BOXPLOTS & VIOLIN PLOTS TEMPLATE
# ==============================================================================
# Purpose: Create professional box plots and violin plots for distribution analysis
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# ================================
# 1. BASIC BOX PLOT
# ================================
# Single variable box plot
# Replace 'numeric_column' with your column name
plt.figure(figsize=(8, 6))
plt.boxplot(df['numeric_column'].dropna(), patch_artist=True, 
           boxprops=dict(facecolor='lightblue', alpha=0.7))
plt.title('Distribution of Numeric Variable', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Value', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ================================
# 2. GROUPED BOX PLOTS
# ================================
# Box plots by categorical variable
# Replace 'category_column' and 'numeric_column' with your column names
plt.figure(figsize=(12, 8))
df.boxplot(column='numeric_column', by='category_column', figsize=(12, 8))
plt.title('Distribution by Category', fontsize=16, fontweight='bold')
plt.suptitle('')  # Remove default title
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# 3. SEABORN BOX PLOTS (More Advanced)
# ================================
# Professional box plots with seaborn
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='category_column', y='numeric_column', 
           palette='Set2', linewidth=1.5)
plt.title('Distribution Analysis by Category', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ================================
# 4. MULTIPLE GROUPED BOX PLOTS
# ================================
# Box plots with multiple grouping variables
# Replace column names as needed
plt.figure(figsize=(16, 8))
sns.boxplot(data=df, x='category_column', y='numeric_column', 
           hue='subcategory_column', palette='viridis')
plt.title('Distribution by Multiple Categories', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Main Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Subcategory', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# 5. VIOLIN PLOTS
# ================================
# Violin plots show distribution shape better than box plots
plt.figure(figsize=(14, 8))
sns.violinplot(data=df, x='category_column', y='numeric_column', 
              palette='muted', linewidth=1.5)
plt.title('Distribution Shape Analysis (Violin Plot)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ================================
# 6. COMBINED BOX AND VIOLIN PLOTS
# ================================
# Combine violin and box plots for comprehensive view
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Box plot
sns.boxplot(data=df, x='category_column', y='numeric_column', ax=ax1, palette='Set1')
ax1.set_title('Box Plot View', fontsize=14, fontweight='bold')
ax1.set_xlabel('Category', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# Violin plot
sns.violinplot(data=df, x='category_column', y='numeric_column', ax=ax2, palette='Set1')
ax2.set_title('Violin Plot View', fontsize=14, fontweight='bold')
ax2.set_xlabel('Category', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Distribution Comparison: Box vs Violin Plots', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ================================
# 7. HORIZONTAL BOX PLOTS
# ================================
# Horizontal orientation for better label readability
plt.figure(figsize=(10, 12))
sns.boxplot(data=df, y='category_column', x='numeric_column', 
           palette='coolwarm', orient='h')
plt.title('Horizontal Distribution Analysis', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ================================
# 8. MULTIPLE VARIABLES BOX PLOTS
# ================================
# Compare distributions of multiple numeric variables
# Replace with your numeric column names
numeric_columns = ['numeric_col1', 'numeric_col2', 'numeric_col3', 'numeric_col4']

# Prepare data for plotting
plot_data = []
for col in numeric_columns:
    for value in df[col].dropna():
        plot_data.append({'Variable': col, 'Value': value})

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(14, 8))
sns.boxplot(data=plot_df, x='Variable', y='Value', palette='bright')
plt.title('Comparison of Multiple Variables', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ================================
# 9. STATISTICAL ANNOTATIONS
# ================================
# Add statistical information to box plots
plt.figure(figsize=(14, 8))
box_plot = sns.boxplot(data=df, x='category_column', y='numeric_column', palette='pastel')

# Calculate and display statistics
categories = df['category_column'].unique()
for i, category in enumerate(categories):
    subset = df[df['category_column'] == category]['numeric_column']
    
    # Calculate statistics
    median = subset.median()
    q75 = subset.quantile(0.75)
    
    # Add text annotation
    plt.text(i, q75 + (q75 * 0.1), f'n={len(subset)}\nμ={subset.mean():.1f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Box Plot with Statistical Annotations', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ================================
# 10. OUTLIER ANALYSIS
# ================================
# Identify and highlight outliers
plt.figure(figsize=(14, 8))

# Create box plot and capture outlier points
box_plot = plt.boxplot([df[df['category_column'] == cat]['numeric_column'].dropna() 
                       for cat in df['category_column'].unique()], 
                      labels=df['category_column'].unique(),
                      patch_artist=True, showfliers=True)

# Customize outlier appearance
for flier in box_plot['fliers']:
    flier.set(marker='o', color='red', alpha=0.7, markersize=8)

plt.title('Outlier Detection in Box Plots', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Print outlier statistics
print("Outlier Analysis:")
for category in df['category_column'].unique():
    subset = df[df['category_column'] == category]['numeric_column']
    Q1 = subset.quantile(0.25)
    Q3 = subset.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
    print(f"{category}: {len(outliers)} outliers ({len(outliers)/len(subset)*100:.1f}%)")

print("\nBox and violin plot templates completed!")
print("Remember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update column names to match your data")
print("3. Adjust figure sizes based on your data")
print("4. Consider log transformation for highly skewed data")
