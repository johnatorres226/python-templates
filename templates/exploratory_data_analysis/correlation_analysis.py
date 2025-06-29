# Replace 'your_data.csv' with your dataset
# Correlation Analysis Template - Matrices, Pairplots, and Advanced Correlation Methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, kendalltau

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== CORRELATION ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Get numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns for correlation analysis: {len(numerical_cols)}")
print(f"Columns: {numerical_cols}")

if len(numerical_cols) < 2:
    print("Warning: Need at least 2 numerical columns for correlation analysis")
    print("Available columns:", df.columns.tolist())
else:
    # 1. PEARSON CORRELATION MATRIX
    print("\n=== PEARSON CORRELATION MATRIX ===")
    
    # Calculate Pearson correlations
    correlation_matrix = df[numerical_cols].corr(method='pearson')
    print("Pearson Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True, 
                fmt='.2f',
                mask=mask,
                cbar_kws={"shrink": .8})
    plt.title('Pearson Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # 2. SPEARMAN RANK CORRELATION
    print("\n=== SPEARMAN RANK CORRELATION ===")
    
    spearman_matrix = df[numerical_cols].corr(method='spearman')
    print("Spearman Correlation Matrix:")
    print(spearman_matrix.round(3))
    
    # Visualize Spearman correlations
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(spearman_matrix, dtype=bool))
    sns.heatmap(spearman_matrix, 
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True, 
                fmt='.2f',
                mask=mask,
                cbar_kws={"shrink": .8})
    plt.title('Spearman Rank Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # 3. KENDALL TAU CORRELATION (for small samples or non-linear relationships)
    print("\n=== KENDALL TAU CORRELATION ===")
    
    # Calculate Kendall's tau for selected pairs (can be computationally intensive)
    kendall_results = {}
    important_pairs = []
    
    # Find pairs with high Pearson or Spearman correlation
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            col1, col2 = numerical_cols[i], numerical_cols[j]
            pearson_corr = correlation_matrix.loc[col1, col2]
            spearman_corr = spearman_matrix.loc[col1, col2]
            
            if abs(pearson_corr) > 0.3 or abs(spearman_corr) > 0.3:
                important_pairs.append((col1, col2))
    
    print(f"Calculating Kendall's tau for {len(important_pairs)} important pairs...")
    
    for col1, col2 in important_pairs[:10]:  # Limit to first 10 pairs
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        
        # Align data (remove rows where either is missing)
        aligned_data = df[[col1, col2]].dropna()
        if len(aligned_data) > 0:
            tau, p_value = kendalltau(aligned_data[col1], aligned_data[col2])
            kendall_results[f"{col1} vs {col2}"] = {"tau": tau, "p_value": p_value}
            print(f"{col1} vs {col2}: τ = {tau:.3f}, p = {p_value:.3f}")
    
    # 4. CORRELATION STRENGTH ANALYSIS
    print("\n=== CORRELATION STRENGTH ANALYSIS ===")
    
    # Extract upper triangle of correlation matrix (avoid duplicates)
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Flatten and remove NaN values
    correlations = upper_triangle.stack().reset_index()
    correlations.columns = ['Variable1', 'Variable2', 'Correlation']
    correlations['Abs_Correlation'] = correlations['Correlation'].abs()
    
    # Sort by absolute correlation
    correlations_sorted = correlations.sort_values('Abs_Correlation', ascending=False)
    
    print("Strongest correlations:")
    print(correlations_sorted.head(10))
    
    # Categorize correlations by strength
    def correlation_strength(corr):
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    correlations_sorted['Strength'] = correlations_sorted['Correlation'].apply(correlation_strength)
    
    print("\nCorrelation strength distribution:")
    strength_counts = correlations_sorted['Strength'].value_counts()
    print(strength_counts)
    
    # 5. PAIRPLOT FOR STRONGEST CORRELATIONS
    print("\n=== PAIRPLOT VISUALIZATION ===")
    
    # Select variables with strongest correlations for pairplot
    top_correlated_vars = set()
    for _, row in correlations_sorted.head(6).iterrows():  # Top 6 pairs
        top_correlated_vars.add(row['Variable1'])
        top_correlated_vars.add(row['Variable2'])
    
    top_correlated_vars = list(top_correlated_vars)[:6]  # Limit to 6 variables for readability
    
    if len(top_correlated_vars) >= 2:
        print(f"Creating pairplot for: {top_correlated_vars}")
        
        # Create pairplot
        pairplot_data = df[top_correlated_vars].dropna()
        
        if len(pairplot_data) > 0:
            plt.figure(figsize=(15, 12))
            sns.pairplot(pairplot_data, diag_kind='hist', plot_kws={'alpha': 0.6})
            plt.suptitle('Pairplot of Strongest Correlated Variables', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.show()
    
    # 6. SCATTER PLOTS WITH CORRELATION DETAILS
    print("\n=== DETAILED SCATTER PLOTS ===")
    
    # Create detailed scatter plots for top correlations
    n_plots = min(6, len(correlations_sorted))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(n_plots):
        row = correlations_sorted.iloc[i]
        var1, var2, corr = row['Variable1'], row['Variable2'], row['Correlation']
        
        # Get clean data for plotting
        plot_data = df[[var1, var2]].dropna()
        
        if len(plot_data) > 0:
            # Scatter plot
            axes[i].scatter(plot_data[var1], plot_data[var2], alpha=0.6, s=30)
            
            # Add regression line
            z = np.polyfit(plot_data[var1], plot_data[var2], 1)
            p = np.poly1d(z)
            axes[i].plot(plot_data[var1], p(plot_data[var1]), "r--", alpha=0.8, linewidth=2)
            
            # Calculate additional statistics
            r_squared = corr ** 2
            n_points = len(plot_data)
            
            axes[i].set_xlabel(var1)
            axes[i].set_ylabel(var2)
            axes[i].set_title(f'{var1} vs {var2}\nr = {corr:.3f}, R² = {r_squared:.3f}, n = {n_points}')
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_plots, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Scatter Plots of Strongest Correlations', y=1.02, fontsize=16)
    plt.show()
    
    # 7. CORRELATION SIGNIFICANCE TESTING
    print("\n=== CORRELATION SIGNIFICANCE TESTING ===")
    
    # Test significance of correlations
    significant_correlations = []
    
    for _, row in correlations_sorted.head(10).iterrows():
        var1, var2, corr = row['Variable1'], row['Variable2'], row['Correlation']
        
        # Get clean data
        clean_data = df[[var1, var2]].dropna()
        
        if len(clean_data) > 3:  # Need at least 4 points for correlation test
            # Pearson correlation test
            pearson_corr, pearson_p = stats.pearsonr(clean_data[var1], clean_data[var2])
            
            # Spearman correlation test
            spearman_corr, spearman_p = spearmanr(clean_data[var1], clean_data[var2])
            
            result = {
                'pair': f"{var1} vs {var2}",
                'pearson_r': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_r': spearman_corr,
                'spearman_p': spearman_p,
                'n_observations': len(clean_data)
            }
            
            significant_correlations.append(result)
            
            print(f"{var1} vs {var2}:")
            print(f"  Pearson: r = {pearson_corr:.3f}, p = {pearson_p:.3f}")
            print(f"  Spearman: ρ = {spearman_corr:.3f}, p = {spearman_p:.3f}")
            print(f"  Sample size: {len(clean_data)}")
            
            if pearson_p < 0.05:
                print("  *** Pearson correlation is statistically significant ***")
            if spearman_p < 0.05:
                print("  *** Spearman correlation is statistically significant ***")
            print()
    
    # 8. MULTICOLLINEARITY DETECTION
    print("\n=== MULTICOLLINEARITY DETECTION ===")
    
    # Find highly correlated pairs (potential multicollinearity)
    high_corr_threshold = 0.8
    high_correlations = correlations_sorted[
        correlations_sorted['Abs_Correlation'] >= high_corr_threshold
    ]
    
    if len(high_correlations) > 0:
        print(f"⚠️ High correlations (|r| >= {high_corr_threshold}) detected:")
        print(high_correlations[['Variable1', 'Variable2', 'Correlation']])
        print("\nMulticollinearity concerns:")
        print("- These variables may be redundant in modeling")
        print("- Consider removing one variable from each pair")
        print("- Or use regularization techniques (Ridge, Lasso)")
    else:
        print(f"✅ No severe multicollinearity detected (threshold: {high_corr_threshold})")
    
    # 9. CORRELATION HEATMAP WITH DENDROGRAM
    print("\n=== CORRELATION CLUSTERING ===")
    
    # Create clustered heatmap
    plt.figure(figsize=(12, 10))
    sns.clustermap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   figsize=(12, 10))
    plt.title('Clustered Correlation Matrix', fontsize=16)
    plt.show()
    
    # 10. CORRELATION SUMMARY REPORT
    print("\n=== CORRELATION SUMMARY REPORT ===")
    
    total_pairs = len(correlations_sorted)
    strong_pairs = len(correlations_sorted[correlations_sorted['Abs_Correlation'] >= 0.7])
    moderate_pairs = len(correlations_sorted[
        (correlations_sorted['Abs_Correlation'] >= 0.5) & 
        (correlations_sorted['Abs_Correlation'] < 0.7)
    ])
    weak_pairs = len(correlations_sorted[
        (correlations_sorted['Abs_Correlation'] >= 0.3) & 
        (correlations_sorted['Abs_Correlation'] < 0.5)
    ])
    
    print(f"Total variable pairs analyzed: {total_pairs}")
    print(f"Strong correlations (|r| >= 0.7): {strong_pairs} ({strong_pairs/total_pairs*100:.1f}%)")
    print(f"Moderate correlations (0.5 <= |r| < 0.7): {moderate_pairs} ({moderate_pairs/total_pairs*100:.1f}%)")
    print(f"Weak correlations (0.3 <= |r| < 0.5): {weak_pairs} ({weak_pairs/total_pairs*100:.1f}%)")
    
    # Variables with most correlations
    var_correlation_counts = {}
    for _, row in correlations_sorted[correlations_sorted['Abs_Correlation'] >= 0.3].iterrows():
        var1, var2 = row['Variable1'], row['Variable2']
        var_correlation_counts[var1] = var_correlation_counts.get(var1, 0) + 1
        var_correlation_counts[var2] = var_correlation_counts.get(var2, 0) + 1
    
    if var_correlation_counts:
        print(f"\nVariables with most correlations (|r| >= 0.3):")
        sorted_vars = sorted(var_correlation_counts.items(), key=lambda x: x[1], reverse=True)
        for var, count in sorted_vars[:5]:
            print(f"  {var}: {count} correlations")

# RECOMMENDATIONS
print("\n=== RECOMMENDATIONS ===")
print("1. Focus on strong correlations (|r| >= 0.7) for feature selection")
print("2. Investigate moderate correlations (0.5 <= |r| < 0.7) for domain insights")
print("3. Use Spearman correlation for non-linear relationships")
print("4. Address multicollinearity before linear modeling")
print("5. Consider correlation with target variable for supervised learning")
print("6. Validate correlations with domain knowledge")

print("\nCorrelation analysis complete. Review visualizations and summary statistics.")
