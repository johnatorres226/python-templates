# ==============================================================================
# HEATMAPS & PAIRPLOTS TEMPLATE
# ==============================================================================
# Purpose: Create professional heatmaps and pairplots for correlation and relationship analysis
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

# ================================
# 1. CORRELATION HEATMAP
# ================================
# Basic correlation matrix heatmap
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
           annot=True,  # Show correlation values
           cmap='coolwarm',  # Color scheme
           center=0,  # Center colormap at 0
           square=True,  # Square cells
           fmt='.2f',  # Format for annotations
           cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# ================================
# 2. ADVANCED CORRELATION HEATMAP
# ================================
# More sophisticated heatmap with custom styling
plt.figure(figsize=(14, 12))

# Create mask for upper triangle (optional)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Generate heatmap
sns.heatmap(correlation_matrix,
           mask=mask,  # Hide upper triangle
           annot=True,
           cmap='RdYlBu_r',
           center=0,
           square=True,
           linewidths=0.5,
           cbar_kws={"shrink": 0.5, "label": "Correlation Coefficient"})
           
plt.title('Lower Triangle Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# ================================
# 3. CLUSTERMAP (HIERARCHICAL CLUSTERING)
# ================================
# Automatically cluster similar variables
plt.figure(figsize=(12, 10))
cluster_map = sns.clustermap(correlation_matrix,
                           annot=True,
                           cmap='viridis',
                           center=0,
                           figsize=(12, 10),
                           cbar_kws={'label': 'Correlation'})
cluster_map.fig.suptitle('Clustered Correlation Matrix', 
                        fontsize=16, fontweight='bold', y=0.95)
plt.show()

# ================================
# 4. CATEGORICAL DATA HEATMAP
# ================================
# Heatmap for categorical data (e.g., survey responses, ratings)
# Replace with your categorical columns
categorical_cols = ['category1', 'category2']  # Update with your columns

if len(categorical_cols) >= 2:
    # Create pivot table for categorical data
    pivot_table = pd.crosstab(df[categorical_cols[0]], df[categorical_cols[1]])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table,
               annot=True,
               fmt='d',  # Integer format
               cmap='Blues',
               cbar_kws={'label': 'Count'})
    plt.title('Categorical Data Cross-tabulation', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(categorical_cols[1], fontsize=12)
    plt.ylabel(categorical_cols[0], fontsize=12)
    plt.tight_layout()
    plt.show()

# ================================
# 5. TIME-BASED HEATMAP
# ================================
# Heatmap showing patterns over time (e.g., by month/day)
# Assuming you have a date column - replace 'date_column' and 'value_column'
if 'date_column' in df.columns:
    # Convert to datetime if not already
    df['date_column'] = pd.to_datetime(df['date_column'])
    
    # Extract time components
    df['month'] = df['date_column'].dt.month
    df['day_of_week'] = df['date_column'].dt.day_name()
    
    # Create pivot table
    time_pivot = df.pivot_table(values='value_column', 
                               index='day_of_week', 
                               columns='month', 
                               aggfunc='mean')
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(time_pivot,
               annot=True,
               fmt='.1f',
               cmap='YlOrRd',
               cbar_kws={'label': 'Average Value'})
    plt.title('Time-based Pattern Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    plt.show()

# ================================
# 6. PAIRPLOT (SCATTERPLOT MATRIX)
# ================================
# Pairwise relationships between all numeric variables
# Select subset of columns for better readability (max 6-8 variables)
plot_columns = numeric_df.columns[:6]  # Adjust as needed
subset_df = df[plot_columns]

# Basic pairplot
sns.pairplot(subset_df, diag_kind='hist', height=2.5)
plt.suptitle('Pairwise Relationships Matrix', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ================================
# 7. ENHANCED PAIRPLOT WITH CATEGORIES
# ================================
# Pairplot colored by categorical variable
# Replace 'category_column' with your categorical column
if 'category_column' in df.columns:
    sns.pairplot(df[plot_columns + ['category_column']], 
                hue='category_column',
                diag_kind='kde',  # Kernel density for diagonal
                height=2.5,
                aspect=1.2)
    plt.suptitle('Pairwise Relationships by Category', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# ================================
# 8. CUSTOM PAIRPLOT WITH REGRESSION LINES
# ================================
# Pairplot with regression lines
g = sns.PairGrid(subset_df, height=2.5)
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(sns.regplot, scatter_kws={'alpha': 0.6})
g.map_diag(sns.histplot, kde=True)
g.fig.suptitle('Pairplot with Regression Lines', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ================================
# 9. MISSING DATA HEATMAP
# ================================
# Visualize missing data patterns
plt.figure(figsize=(12, 8))
missing_data = df.isnull()
sns.heatmap(missing_data,
           cbar=True,
           cmap='viridis',
           yticklabels=False,  # Don't show row labels
           cbar_kws={'label': 'Missing Data'})
plt.title('Missing Data Pattern', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Observations', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print missing data summary
print("Missing Data Summary:")
missing_summary = df.isnull().sum()
missing_percentage = (missing_summary / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_summary,
    'Percentage': missing_percentage
}).sort_values('Percentage', ascending=False)
print(missing_df[missing_df['Missing Count'] > 0])

# ================================
# 10. ANNOTATED HEATMAP WITH SIGNIFICANCE
# ================================
# Correlation heatmap with significance indicators
from scipy.stats import pearsonr

def correlation_significance(df):
    """Calculate correlation matrix with p-values"""
    cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = np.zeros((len(cols), len(cols)))
    p_values = np.zeros((len(cols), len(cols)))
    
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i == j:
                correlation_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                corr, p_val = pearsonr(df[col1].dropna(), df[col2].dropna())
                correlation_matrix[i, j] = corr
                p_values[i, j] = p_val
    
    return pd.DataFrame(correlation_matrix, columns=cols, index=cols), \
           pd.DataFrame(p_values, columns=cols, index=cols)

# Calculate correlations and p-values
try:
    from scipy.stats import pearsonr
    corr_matrix, p_matrix = correlation_significance(df)
    
    # Create significance mask (p < 0.05)
    sig_mask = p_matrix > 0.05
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix,
               annot=True,
               mask=sig_mask,  # Mask non-significant correlations
               cmap='RdBu_r',
               center=0,
               square=True,
               fmt='.2f',
               cbar_kws={'label': 'Correlation (p < 0.05)'})
    plt.title('Significant Correlations Only (p < 0.05)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("Scipy not available for significance testing")

# ================================
# INTERPRETATION GUIDE
# ================================
print("\nHeatmap & Pairplot Interpretation Guide:")
print("="*50)
print("CORRELATION VALUES:")
print("• +1.0 to +0.7: Strong positive correlation")
print("• +0.7 to +0.3: Moderate positive correlation")
print("• +0.3 to -0.3: Weak correlation")
print("• -0.3 to -0.7: Moderate negative correlation")
print("• -0.7 to -1.0: Strong negative correlation")
print("\nPAIRPLOT PATTERNS:")
print("• Linear patterns: Linear relationships")
print("• Curved patterns: Non-linear relationships")
print("• Scattered patterns: Weak relationships")
print("• Clustered patterns: Distinct groups")

print("\nTemplates completed!")
print("Remember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update column names to match your data")
print("3. Adjust figure sizes based on number of variables")
print("4. Consider correlation strength when interpreting results")
