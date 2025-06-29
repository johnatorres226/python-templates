# Replace 'your_data.csv' with your dataset
# Histogram and KDE Template - Distribution Analysis and Density Estimation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== HISTOGRAM AND KDE ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Get numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

if len(numerical_cols) == 0:
    print("No numerical columns found for histogram analysis.")
else:
    # 1. BASIC HISTOGRAMS
    print("\n=== 1. BASIC HISTOGRAMS ===")
    
    n_vars = min(6, len(numerical_cols))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols[:n_vars]):
        data = df[col].dropna()
        
        if len(data) > 0:
            # Calculate optimal number of bins using Freedman-Diaconis rule
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(data) ** (1/3))
            n_bins = int((data.max() - data.min()) / bin_width) if bin_width > 0 else 30
            n_bins = max(10, min(50, n_bins))  # Keep between 10 and 50 bins
            
            # Create histogram
            axes[i].hist(data, bins=n_bins, alpha=0.7, color='skyblue', 
                        edgecolor='black', linewidth=0.5, density=True)
            
            # Add statistics text
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            # Customize plot
            axes[i].set_title(f'{col} Distribution\nStd: {std_val:.2f}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_vars, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Basic Histograms with Central Tendency', fontsize=16, y=1.02)
    plt.show()
    
    # 2. HISTOGRAMS WITH KDE OVERLAY
    print("\n=== 2. HISTOGRAMS WITH KDE OVERLAY ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols[:n_vars]):
        data = df[col].dropna()
        
        if len(data) > 1:
            # Histogram
            axes[i].hist(data, bins=30, alpha=0.6, color='lightcoral', 
                        edgecolor='black', linewidth=0.5, density=True, label='Histogram')
            
            # KDE overlay
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            axes[i].plot(x_range, kde(x_range), 'navy', linewidth=2, label='KDE')
            
            # Normal distribution overlay for comparison
            mean_val = data.mean()
            std_val = data.std()
            normal_dist = stats.norm(mean_val, std_val)
            axes[i].plot(x_range, normal_dist.pdf(x_range), 'green', 
                        linestyle='--', linewidth=2, label='Normal', alpha=0.7)
            
            # Add skewness and kurtosis
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data)
            
            axes[i].set_title(f'{col}\nSkew: {skew_val:.2f}, Kurt: {kurt_val:.2f}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_vars, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Histograms with KDE and Normal Distribution Overlay', fontsize=16, y=1.02)
    plt.show()
    
    # 3. GROUPED HISTOGRAMS BY CATEGORY
    print("\n=== 3. GROUPED HISTOGRAMS BY CATEGORY ===")
    
    if categorical_cols:
        # Use first categorical column for grouping
        group_col = categorical_cols[0]
        
        # Only proceed if reasonable number of groups
        unique_groups = df[group_col].dropna().unique()
        if len(unique_groups) <= 5:
            num_col = numerical_cols[0]  # Use first numerical column
            
            plt.figure(figsize=(12, 8))
            
            # Get data for each group
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
            
            for i, group in enumerate(unique_groups):
                group_data = df[df[group_col] == group][num_col].dropna()
                
                if len(group_data) > 0:
                    plt.hist(group_data, bins=20, alpha=0.6, 
                            label=f'{group} (n={len(group_data)})',
                            color=colors[i], edgecolor='black', linewidth=0.5)
            
            plt.title(f'{num_col} Distribution by {group_col}', fontsize=14, fontweight='bold')
            plt.xlabel(num_col)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    # 4. FACETED HISTOGRAMS
    print("\n=== 4. FACETED HISTOGRAMS ===")
    
    if categorical_cols:
        group_col = categorical_cols[0]
        unique_groups = df[group_col].dropna().unique()
        
        if len(unique_groups) <= 6:
            num_col = numerical_cols[0]
            
            n_groups = len(unique_groups)
            n_cols = min(3, n_groups)
            n_rows = (n_groups + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, group in enumerate(unique_groups):
                row = i // n_cols
                col_idx = i % n_cols
                
                group_data = df[df[group_col] == group][num_col].dropna()
                
                if len(group_data) > 0:
                    # Histogram
                    axes[row][col_idx].hist(group_data, bins=20, alpha=0.7, 
                                           color='steelblue', edgecolor='black', linewidth=0.5)
                    
                    # Add statistics
                    mean_val = group_data.mean()
                    axes[row][col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2)
                    
                    # Customize subplot
                    axes[row][col_idx].set_title(f'{group}\nMean: {mean_val:.2f}, n={len(group_data)}')
                    axes[row][col_idx].set_xlabel(num_col)
                    axes[row][col_idx].set_ylabel('Frequency')
                    axes[row][col_idx].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_groups, n_rows * n_cols):
                row = i // n_cols
                col_idx = i % n_cols
                axes[row][col_idx].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle(f'{num_col} Distribution by {group_col}', fontsize=16, y=1.02)
            plt.show()
    
    # 5. STACKED HISTOGRAMS
    print("\n=== 5. STACKED HISTOGRAMS ===")
    
    if categorical_cols:
        group_col = categorical_cols[0]
        unique_groups = df[group_col].dropna().unique()
        
        if len(unique_groups) <= 5:
            num_col = numerical_cols[0]
            
            plt.figure(figsize=(12, 8))
            
            # Prepare data for each group
            group_data_list = []
            labels = []
            
            for group in unique_groups:
                group_data = df[df[group_col] == group][num_col].dropna()
                if len(group_data) > 0:
                    group_data_list.append(group_data)
                    labels.append(f'{group} (n={len(group_data)})')
            
            # Create stacked histogram
            plt.hist(group_data_list, bins=25, alpha=0.7, label=labels, 
                    stacked=True, edgecolor='black', linewidth=0.5)
            
            plt.title(f'Stacked Histogram: {num_col} by {group_col}', fontsize=14, fontweight='bold')
            plt.xlabel(num_col)
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    # 6. PROBABILITY DENSITY COMPARISON
    print("\n=== 6. PROBABILITY DENSITY COMPARISON ===")
    
    if len(numerical_cols) >= 2:
        plt.figure(figsize=(12, 8))
        
        # Compare KDE of multiple variables
        for i, col in enumerate(numerical_cols[:4]):  # Compare up to 4 variables
            data = df[col].dropna()
            
            if len(data) > 1:
                # Standardize data for comparison
                standardized_data = (data - data.mean()) / data.std()
                
                # Calculate KDE
                kde = stats.gaussian_kde(standardized_data)
                x_range = np.linspace(-4, 4, 200)
                
                plt.plot(x_range, kde(x_range), linewidth=2, label=f'{col}', alpha=0.8)
        
        # Add standard normal for reference
        x_range = np.linspace(-4, 4, 200)
        plt.plot(x_range, stats.norm.pdf(x_range), 'k--', linewidth=2, 
                label='Standard Normal', alpha=0.7)
        
        plt.title('Standardized Density Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Standardized Values')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 7. CUMULATIVE DISTRIBUTION
    print("\n=== 7. CUMULATIVE DISTRIBUTION FUNCTIONS ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Empirical CDF
    col = numerical_cols[0]
    data = df[col].dropna().sort_values()
    y_values = np.arange(1, len(data) + 1) / len(data)
    
    axes[0].plot(data, y_values, linewidth=2, color='blue', label='Empirical CDF')
    
    # Theoretical normal CDF for comparison
    mean_val = data.mean()
    std_val = data.std()
    x_range = np.linspace(data.min(), data.max(), 200)
    theoretical_cdf = stats.norm.cdf(x_range, mean_val, std_val)
    axes[0].plot(x_range, theoretical_cdf, 'r--', linewidth=2, label='Normal CDF')
    
    axes[0].set_title(f'Cumulative Distribution: {col}', fontweight='bold')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Cumulative Probability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # P-P plot
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)), mean_val, std_val)
    
    axes[1].scatter(theoretical_quantiles, sorted_data, alpha=0.6, s=30)
    
    # Add diagonal line
    min_val = min(theoretical_quantiles.min(), sorted_data.min())
    max_val = max(theoretical_quantiles.max(), sorted_data.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    axes[1].set_title(f'P-P Plot: {col} vs Normal', fontweight='bold')
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Sample Quantiles')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 8. DISTRIBUTION COMPARISON SUMMARY
    print("\n=== 8. DISTRIBUTION COMPARISON SUMMARY ===")
    
    # Create summary table of distribution characteristics
    dist_summary = pd.DataFrame(index=numerical_cols[:5])
    
    for col in numerical_cols[:5]:
        data = df[col].dropna()
        
        if len(data) > 0:
            dist_summary.loc[col, 'count'] = len(data)
            dist_summary.loc[col, 'mean'] = data.mean()
            dist_summary.loc[col, 'std'] = data.std()
            dist_summary.loc[col, 'skewness'] = stats.skew(data)
            dist_summary.loc[col, 'kurtosis'] = stats.kurtosis(data)
            
            # Normality test
            if len(data) >= 8:
                _, p_value = stats.shapiro(data[:5000])  # Limit for large datasets
                dist_summary.loc[col, 'shapiro_p'] = p_value
                dist_summary.loc[col, 'is_normal'] = p_value > 0.05
            
            # Distribution shape
            if abs(stats.skew(data)) < 0.5:
                shape = 'Symmetric'
            elif stats.skew(data) > 0.5:
                shape = 'Right-skewed'
            else:
                shape = 'Left-skewed'
            
            dist_summary.loc[col, 'shape'] = shape
    
    print("Distribution Summary:")
    print(dist_summary.round(3))

print("\n=== HISTOGRAM AND KDE BEST PRACTICES ===")
print("1. Use appropriate bin sizes - too few bins lose detail, too many create noise")
print("2. KDE smooths data but may hide important features")
print("3. Compare distributions using standardized values")
print("4. Use faceting to compare groups while maintaining scale")
print("5. Overlay normal distributions to assess normality")
print("6. Include sample sizes when comparing groups")
print("7. Consider log transformation for highly skewed data")
print("8. Use cumulative distributions for better comparison")
print("9. Check for multimodality which might indicate subgroups")
print("10. Always report both central tendency and spread measures")

print("\nHistogram and KDE analysis complete.")
