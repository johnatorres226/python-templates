# Replace 'your_data.csv' with your dataset
# Descriptive Statistics Template - Comprehensive Statistical Summary

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== DESCRIPTIVE STATISTICS ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Get numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical variables: {len(numerical_cols)}")
print(f"Categorical variables: {len(categorical_cols)}")

# 1. BASIC DESCRIPTIVE STATISTICS
print("\n=== BASIC DESCRIPTIVE STATISTICS ===")

if numerical_cols:
    # Standard pandas describe
    basic_stats = df[numerical_cols].describe()
    print("Standard Descriptive Statistics:")
    print(basic_stats.round(3))
    
    # Extended descriptive statistics
    extended_stats = pd.DataFrame(index=numerical_cols)
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        if len(data) > 0:
            # Central tendency
            extended_stats.loc[col, 'count'] = len(data)
            extended_stats.loc[col, 'mean'] = data.mean()
            extended_stats.loc[col, 'median'] = data.median()
            extended_stats.loc[col, 'mode'] = data.mode()[0] if not data.mode().empty else np.nan
            
            # Dispersion measures
            extended_stats.loc[col, 'std'] = data.std()
            extended_stats.loc[col, 'variance'] = data.var()
            extended_stats.loc[col, 'range'] = data.max() - data.min()
            extended_stats.loc[col, 'iqr'] = data.quantile(0.75) - data.quantile(0.25)
            extended_stats.loc[col, 'mad'] = (data - data.median()).abs().median()  # Median Absolute Deviation
            extended_stats.loc[col, 'cv'] = (data.std() / data.mean()) * 100 if data.mean() != 0 else np.nan  # Coefficient of Variation
            
            # Shape measures
            extended_stats.loc[col, 'skewness'] = skew(data)
            extended_stats.loc[col, 'kurtosis'] = kurtosis(data)
            
            # Percentiles
            extended_stats.loc[col, 'p1'] = data.quantile(0.01)
            extended_stats.loc[col, 'p5'] = data.quantile(0.05)
            extended_stats.loc[col, 'p10'] = data.quantile(0.10)
            extended_stats.loc[col, 'p90'] = data.quantile(0.90)
            extended_stats.loc[col, 'p95'] = data.quantile(0.95)
            extended_stats.loc[col, 'p99'] = data.quantile(0.99)
            
            # Missing data
            extended_stats.loc[col, 'missing_count'] = df[col].isnull().sum()
            extended_stats.loc[col, 'missing_pct'] = (df[col].isnull().sum() / len(df)) * 100
            
            # Outliers (using IQR method)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            extended_stats.loc[col, 'outliers_count'] = len(outliers)
            extended_stats.loc[col, 'outliers_pct'] = (len(outliers) / len(data)) * 100
    
    print("\nExtended Descriptive Statistics:")
    print(extended_stats.round(3))

# 2. DISTRIBUTION SHAPE ANALYSIS
print("\n=== DISTRIBUTION SHAPE ANALYSIS ===")

if numerical_cols:
    shape_analysis = pd.DataFrame(index=numerical_cols)
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        if len(data) > 0:
            skewness_val = skew(data)
            kurtosis_val = kurtosis(data)
            
            # Skewness interpretation
            if abs(skewness_val) < 0.5:
                skew_interp = "Approximately symmetric"
            elif skewness_val > 0.5:
                if skewness_val > 1:
                    skew_interp = "Highly right-skewed"
                else:
                    skew_interp = "Moderately right-skewed"
            else:
                if skewness_val < -1:
                    skew_interp = "Highly left-skewed"
                else:
                    skew_interp = "Moderately left-skewed"
            
            # Kurtosis interpretation (excess kurtosis)
            if abs(kurtosis_val) < 0.5:
                kurt_interp = "Normal-tailed (mesokurtic)"
            elif kurtosis_val > 0.5:
                if kurtosis_val > 3:
                    kurt_interp = "Very heavy-tailed (leptokurtic)"
                else:
                    kurt_interp = "Heavy-tailed (leptokurtic)"
            else:
                if kurtosis_val < -1:
                    kurt_interp = "Very light-tailed (platykurtic)"
                else:
                    kurt_interp = "Light-tailed (platykurtic)"
            
            shape_analysis.loc[col, 'skewness'] = skewness_val
            shape_analysis.loc[col, 'skew_interpretation'] = skew_interp
            shape_analysis.loc[col, 'kurtosis'] = kurtosis_val
            shape_analysis.loc[col, 'kurtosis_interpretation'] = kurt_interp
            
            # Normality tests
            if len(data) >= 8:
                # Shapiro-Wilk test (good for small samples)
                shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for large datasets
                shape_analysis.loc[col, 'shapiro_stat'] = shapiro_stat
                shape_analysis.loc[col, 'shapiro_p'] = shapiro_p
                shape_analysis.loc[col, 'shapiro_normal'] = shapiro_p > 0.05
                
                # Jarque-Bera test (for large samples)
                if len(data) >= 30:
                    jb_stat, jb_p = stats.jarque_bera(data)
                    shape_analysis.loc[col, 'jarque_bera_stat'] = jb_stat
                    shape_analysis.loc[col, 'jarque_bera_p'] = jb_p
                    shape_analysis.loc[col, 'jb_normal'] = jb_p > 0.05
    
    print("Distribution Shape Analysis:")
    print(shape_analysis.round(4))

# 3. CATEGORICAL DESCRIPTIVE STATISTICS
print("\n=== CATEGORICAL DESCRIPTIVE STATISTICS ===")

if categorical_cols:
    cat_stats = pd.DataFrame(index=categorical_cols)
    
    for col in categorical_cols:
        data = df[col].dropna()
        value_counts = df[col].value_counts()
        
        cat_stats.loc[col, 'count'] = len(data)
        cat_stats.loc[col, 'unique_values'] = df[col].nunique()
        cat_stats.loc[col, 'missing_count'] = df[col].isnull().sum()
        cat_stats.loc[col, 'missing_pct'] = (df[col].isnull().sum() / len(df)) * 100
        
        # Mode and frequency
        if not value_counts.empty:
            cat_stats.loc[col, 'mode'] = value_counts.index[0]
            cat_stats.loc[col, 'mode_frequency'] = value_counts.iloc[0]
            cat_stats.loc[col, 'mode_percentage'] = (value_counts.iloc[0] / len(data)) * 100
        
        # Diversity measures
        if len(value_counts) > 0:
            # Calculate entropy (measure of diversity)
            probabilities = value_counts / value_counts.sum()
            entropy = -sum(probabilities * np.log2(probabilities))
            cat_stats.loc[col, 'entropy'] = entropy
            cat_stats.loc[col, 'max_entropy'] = np.log2(len(value_counts))  # Maximum possible entropy
            cat_stats.loc[col, 'normalized_entropy'] = entropy / np.log2(len(value_counts)) if len(value_counts) > 1 else 0
            
            # Gini-Simpson index (probability that two randomly selected items are different)
            gini_simpson = 1 - sum(probabilities ** 2)
            cat_stats.loc[col, 'gini_simpson'] = gini_simpson
        
        # Cardinality assessment
        unique_ratio = cat_stats.loc[col, 'unique_values'] / len(df)
        if unique_ratio > 0.95:
            cardinality = "Very High (Identifier-like)"
        elif unique_ratio > 0.7:
            cardinality = "High"
        elif cat_stats.loc[col, 'unique_values'] > 50:
            cardinality = "Medium-High"
        elif cat_stats.loc[col, 'unique_values'] > 10:
            cardinality = "Medium"
        else:
            cardinality = "Low"
        
        cat_stats.loc[col, 'cardinality_level'] = cardinality
    
    print("Categorical Statistics:")
    print(cat_stats.round(3))
    
    # Display value counts for low cardinality variables
    print("\nValue Counts for Low Cardinality Variables:")
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            value_percentages = df[col].value_counts(normalize=True) * 100
            
            summary_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_percentages
            })
            print(summary_df.round(2))

# 4. ROBUST STATISTICS (less sensitive to outliers)
print("\n=== ROBUST STATISTICS ===")

if numerical_cols:
    robust_stats = pd.DataFrame(index=numerical_cols)
    
    for col in numerical_cols:
        data = df[col].dropna()
        
        if len(data) > 0:
            # Robust central tendency
            robust_stats.loc[col, 'median'] = data.median()
            robust_stats.loc[col, 'trimmed_mean_10'] = stats.trim_mean(data, 0.1)  # 10% trimmed mean
            robust_stats.loc[col, 'trimmed_mean_20'] = stats.trim_mean(data, 0.2)  # 20% trimmed mean
            
            # Robust dispersion
            robust_stats.loc[col, 'mad'] = (data - data.median()).abs().median()  # Median Absolute Deviation
            robust_stats.loc[col, 'iqr'] = data.quantile(0.75) - data.quantile(0.25)
            robust_stats.loc[col, 'robust_std'] = 1.4826 * robust_stats.loc[col, 'mad']  # Robust standard deviation
            
            # Percentile-based statistics
            robust_stats.loc[col, 'p25'] = data.quantile(0.25)
            robust_stats.loc[col, 'p75'] = data.quantile(0.75)
            robust_stats.loc[col, 'p10_p90_range'] = data.quantile(0.90) - data.quantile(0.10)
            
            # Comparison with non-robust statistics
            robust_stats.loc[col, 'mean_vs_median'] = data.mean() - data.median()
            robust_stats.loc[col, 'std_vs_robust_std'] = data.std() - robust_stats.loc[col, 'robust_std']
    
    print("Robust Statistics:")
    print(robust_stats.round(3))

# 5. VISUALIZATION OF DESCRIPTIVE STATISTICS
print("\n=== DESCRIPTIVE STATISTICS VISUALIZATIONS ===")

if numerical_cols:
    # Box plots for all numerical variables
    n_vars = min(8, len(numerical_cols))
    n_cols = min(4, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    if n_vars > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(numerical_cols[:n_vars]):
            row = i // n_cols
            col_idx = i % n_cols
            
            data = df[col].dropna()
            
            # Box plot
            axes[row][col_idx].boxplot(data)
            axes[row][col_idx].set_title(f'{col}\nMean: {data.mean():.2f}, Median: {data.median():.2f}')
            axes[row][col_idx].set_ylabel('Value')
            axes[row][col_idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row][col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Box Plots - Central Tendency and Spread', y=1.02, fontsize=16)
        plt.show()
    
    # Summary statistics comparison plot
    if len(numerical_cols) > 0:
        # Create comparison of mean vs median
        comparison_data = []
        for col in numerical_cols[:10]:  # Limit to 10 variables
            data = df[col].dropna()
            if len(data) > 0:
                comparison_data.append({
                    'Variable': col,
                    'Mean': data.mean(),
                    'Median': data.median(),
                    'Std': data.std(),
                    'MAD': (data - data.median()).abs().median() * 1.4826
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Mean vs Median comparison
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            x = range(len(comp_df))
            width = 0.35
            
            axes[0].bar([i - width/2 for i in x], comp_df['Mean'], width, label='Mean', alpha=0.7)
            axes[0].bar([i + width/2 for i in x], comp_df['Median'], width, label='Median', alpha=0.7)
            axes[0].set_xlabel('Variables')
            axes[0].set_ylabel('Value')
            axes[0].set_title('Mean vs Median Comparison')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(comp_df['Variable'], rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Std vs MAD comparison
            axes[1].bar([i - width/2 for i in x], comp_df['Std'], width, label='Standard Deviation', alpha=0.7)
            axes[1].bar([i + width/2 for i in x], comp_df['MAD'], width, label='Robust Std (MAD)', alpha=0.7)
            axes[1].set_xlabel('Variables')
            axes[1].set_ylabel('Value')
            axes[1].set_title('Standard Deviation vs Robust Standard Deviation')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(comp_df['Variable'], rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

# 6. SUMMARY REPORT
print("\n=== DESCRIPTIVE STATISTICS SUMMARY REPORT ===")

if numerical_cols:
    print("NUMERICAL VARIABLES SUMMARY:")
    print(f"Total numerical variables: {len(numerical_cols)}")
    
    # Identify interesting patterns
    if 'extended_stats' in locals():
        high_variability = extended_stats[extended_stats['cv'] > 100].index.tolist() if 'cv' in extended_stats.columns else []
        highly_skewed = extended_stats[abs(extended_stats['skewness']) > 1].index.tolist() if 'skewness' in extended_stats.columns else []
        high_outliers = extended_stats[extended_stats['outliers_pct'] > 5].index.tolist() if 'outliers_pct' in extended_stats.columns else []
        
        if high_variability:
            print(f"High variability (CV > 100%): {high_variability}")
        if highly_skewed:
            print(f"Highly skewed (|skewness| > 1): {highly_skewed}")
        if high_outliers:
            print(f"High outlier percentage (>5%): {high_outliers}")

if categorical_cols:
    print(f"\nCATEGORICAL VARIABLES SUMMARY:")
    print(f"Total categorical variables: {len(categorical_cols)}")
    
    if 'cat_stats' in locals():
        high_cardinality = cat_stats[cat_stats['cardinality_level'].str.contains('High', na=False)].index.tolist()
        high_missing = cat_stats[cat_stats['missing_pct'] > 10].index.tolist() if 'missing_pct' in cat_stats.columns else []
        
        if high_cardinality:
            print(f"High cardinality variables: {high_cardinality}")
        if high_missing:
            print(f"High missing data (>10%): {high_missing}")

print("\n=== RECOMMENDATIONS ===")
print("1. Variables with high skewness may need transformation")
print("2. High coefficient of variation indicates high relative variability")
print("3. Use robust statistics (median, MAD) for data with outliers")
print("4. High cardinality categorical variables may need special encoding")
print("5. Consider missing data patterns before analysis")
print("6. Validate unusual distributions with domain knowledge")

print("\nDescriptive statistics analysis complete.")
