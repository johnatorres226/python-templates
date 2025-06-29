# Replace 'your_data.csv' with your dataset
# Variable Distribution Analysis Template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== VARIABLE DISTRIBUTION ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Separate numerical and categorical variables
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical variables: {len(numerical_cols)}")
print(f"Categorical variables: {len(categorical_cols)}")

# 1. NUMERICAL VARIABLE DISTRIBUTIONS
print("\n=== NUMERICAL DISTRIBUTIONS ===")

if numerical_cols:
    # Calculate distribution statistics
    dist_stats = pd.DataFrame(index=numerical_cols)
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        if len(data) > 0:
            dist_stats.loc[col, 'count'] = len(data)
            dist_stats.loc[col, 'mean'] = data.mean()
            dist_stats.loc[col, 'median'] = data.median()
            dist_stats.loc[col, 'std'] = data.std()
            dist_stats.loc[col, 'min'] = data.min()
            dist_stats.loc[col, 'max'] = data.max()
            dist_stats.loc[col, 'skewness'] = stats.skew(data)
            dist_stats.loc[col, 'kurtosis'] = stats.kurtosis(data)
            dist_stats.loc[col, 'missing_pct'] = (df[col].isnull().sum() / len(df)) * 100
            
            # Normality tests
            if len(data) >= 8:  # Minimum sample size for Shapiro-Wilk
                shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit sample size for large datasets
                dist_stats.loc[col, 'shapiro_p'] = shapiro_p
                dist_stats.loc[col, 'is_normal'] = shapiro_p > 0.05
            
            # Distribution shape classification
            if abs(dist_stats.loc[col, 'skewness']) < 0.5:
                shape = 'Approximately Normal'
            elif dist_stats.loc[col, 'skewness'] > 0.5:
                shape = 'Right Skewed'
            else:
                shape = 'Left Skewed'
            
            if abs(dist_stats.loc[col, 'kurtosis']) > 3:
                shape += ' (Heavy Tails)' if dist_stats.loc[col, 'kurtosis'] > 3 else ' (Light Tails)'
            
            dist_stats.loc[col, 'distribution_shape'] = shape
    
    # Display summary
    print("Distribution Statistics:")
    print(dist_stats.round(3))
    
    # Create comprehensive plots for numerical variables
    n_cols = min(4, len(numerical_cols))  # Max 4 columns per row
    n_vars = min(8, len(numerical_cols))   # Max 8 variables to plot
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    if n_vars > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(numerical_cols[:n_vars]):
            row = i // n_cols
            col_idx = i % n_cols
            
            data = df[col].dropna()
            
            # Histogram with KDE
            axes[row, col_idx].hist(data, bins=30, alpha=0.7, density=True, color='skyblue')
            
            # Add KDE curve
            if len(data) > 1:
                kde_x = np.linspace(data.min(), data.max(), 100)
                kde = stats.gaussian_kde(data)
                axes[row, col_idx].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
            
            # Add normal distribution overlay for comparison
            mean, std = data.mean(), data.std()
            x_norm = np.linspace(data.min(), data.max(), 100)
            y_norm = stats.norm.pdf(x_norm, mean, std)
            axes[row, col_idx].plot(x_norm, y_norm, 'g--', alpha=0.7, label='Normal')
            
            axes[row, col_idx].set_title(f'{col}\nSkew: {stats.skew(data):.2f}, Kurt: {stats.kurtosis(data):.2f}')
            axes[row, col_idx].set_xlabel('Value')
            axes[row, col_idx].set_ylabel('Density')
            axes[row, col_idx].legend()
            axes[row, col_idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Numerical Variable Distributions', y=1.02, fontsize=16)
        plt.show()
    
    # Q-Q plots for normality assessment
    if len(numerical_cols) > 0:
        n_qq_plots = min(6, len(numerical_cols))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols[:n_qq_plots]):
            data = df[col].dropna()
            stats.probplot(data, dist="norm", plot=axes[i])
            axes[i].set_title(f'{col} - Q-Q Plot')
            axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_qq_plots, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Q-Q Plots for Normality Assessment', y=1.02, fontsize=16)
        plt.show()

# 2. CATEGORICAL VARIABLE DISTRIBUTIONS
print("\n=== CATEGORICAL DISTRIBUTIONS ===")

if categorical_cols:
    # Calculate categorical statistics
    cat_stats = pd.DataFrame(index=categorical_cols)
    
    for col in categorical_cols:
        data = df[col].dropna()
        
        cat_stats.loc[col, 'count'] = len(data)
        cat_stats.loc[col, 'unique_values'] = df[col].nunique()
        cat_stats.loc[col, 'missing_pct'] = (df[col].isnull().sum() / len(df)) * 100
        cat_stats.loc[col, 'most_frequent'] = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
        cat_stats.loc[col, 'most_frequent_pct'] = (df[col].value_counts().iloc[0] / len(data)) * 100 if len(data) > 0 else 0
        
        # Entropy (measure of diversity)
        value_counts = df[col].value_counts()
        probabilities = value_counts / value_counts.sum()
        entropy = -sum(probabilities * np.log2(probabilities))
        cat_stats.loc[col, 'entropy'] = entropy
        
        # Classification based on cardinality
        unique_ratio = cat_stats.loc[col, 'unique_values'] / len(df)
        if unique_ratio > 0.9:
            cardinality = 'Very High (ID-like)'
        elif unique_ratio > 0.5:
            cardinality = 'High'
        elif cat_stats.loc[col, 'unique_values'] > 20:
            cardinality = 'Medium-High'
        elif cat_stats.loc[col, 'unique_values'] > 10:
            cardinality = 'Medium'
        else:
            cardinality = 'Low'
        
        cat_stats.loc[col, 'cardinality'] = cardinality
    
    print("Categorical Statistics:")
    print(cat_stats.round(3))
    
    # Plot categorical distributions
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= 15]
    n_cat_plots = min(6, len(low_cardinality_cols))
    
    if n_cat_plots > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(low_cardinality_cols[:n_cat_plots]):
            value_counts = df[col].value_counts()
            
            # Bar plot
            axes[i].bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            axes[i].set_title(f'{col}\n({len(value_counts)} categories)')
            axes[i].set_xlabel('Categories')
            axes[i].set_ylabel('Count')
            
            # Set x-tick labels
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels on bars
            total = value_counts.sum()
            for j, v in enumerate(value_counts.values):
                axes[i].text(j, v + total*0.01, f'{v/total*100:.1f}%', 
                           ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for i in range(n_cat_plots, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Categorical Variable Distributions', y=1.02, fontsize=16)
        plt.show()

# 3. DISTRIBUTION INSIGHTS AND RECOMMENDATIONS
print("\n=== DISTRIBUTION INSIGHTS ===")

if numerical_cols:
    print("Numerical Variables:")
    for col in numerical_cols:
        if col in dist_stats.index:
            skewness = dist_stats.loc[col, 'skewness']
            kurtosis = dist_stats.loc[col, 'kurtosis']
            
            print(f"\n{col}:")
            print(f"  Shape: {dist_stats.loc[col, 'distribution_shape']}")
            
            # Transformation recommendations
            if abs(skewness) > 1:
                if skewness > 1:
                    print(f"  Recommendation: Highly right-skewed - consider log or sqrt transformation")
                else:
                    print(f"  Recommendation: Highly left-skewed - consider square or exponential transformation")
            elif abs(skewness) > 0.5:
                print(f"  Recommendation: Moderately skewed - transformation may help")
            else:
                print(f"  Recommendation: Approximately normal - no transformation needed")
            
            # Outlier indicators
            if abs(kurtosis) > 3:
                print(f"  Note: Heavy tails detected - check for outliers")

if categorical_cols:
    print("\nCategorical Variables:")
    for col in categorical_cols:
        if col in cat_stats.index:
            print(f"\n{col}:")
            print(f"  Cardinality: {cat_stats.loc[col, 'cardinality']}")
            print(f"  Unique values: {cat_stats.loc[col, 'unique_values']}")
            print(f"  Entropy: {cat_stats.loc[col, 'entropy']:.2f}")
            
            # Encoding recommendations
            unique_count = cat_stats.loc[col, 'unique_values']
            if unique_count <= 5:
                print(f"  Recommendation: One-hot encoding suitable")
            elif unique_count <= 20:
                print(f"  Recommendation: Label encoding or one-hot with feature selection")
            else:
                print(f"  Recommendation: Target encoding, binary encoding, or feature hashing")

# 4. CORRELATION WITH TARGET (if target variable exists)
target_column = 'target_column'  # Replace with your target column name

if target_column in df.columns:
    print(f"\n=== CORRELATION WITH TARGET: {target_column} ===")
    
    # Numerical correlations
    if numerical_cols:
        print("Numerical variable correlations with target:")
        correlations = []
        for col in numerical_cols:
            if col != target_column:
                corr = df[col].corr(df[target_column])
                correlations.append((col, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, corr in correlations:
            print(f"  {col}: {corr:.3f}")
    
    # Categorical associations (using point-biserial correlation for binary target)
    if categorical_cols and df[target_column].nunique() == 2:
        print("\nCategorical variable associations with binary target (point-biserial):")
        for col in categorical_cols:
            # Create dummy variables and calculate correlation
            dummies = pd.get_dummies(df[col], prefix=col)
            for dummy_col in dummies.columns:
                corr = dummies[dummy_col].corr(df[target_column])
                print(f"  {dummy_col}: {corr:.3f}")

print("\n=== ANALYSIS COMPLETE ===")
print("Key recommendations:")
print("1. Check skewed variables for transformation needs")
print("2. Investigate variables with high kurtosis for outliers")
print("3. Consider encoding strategies for categorical variables")
print("4. Focus on variables with strong target correlations for modeling")
print("5. Validate distribution assumptions before statistical tests")
