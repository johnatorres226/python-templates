# Replace 'your_data.csv' with your dataset
# Feature Distributions by Group/Class Analysis Template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== FEATURE DISTRIBUTIONS BY GROUP ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Replace with your grouping/class variable
group_column = 'group_column'  # Replace with your grouping variable (e.g., 'category', 'class', 'treatment')

if group_column not in df.columns:
    print(f"Warning: '{group_column}' not found. Please replace with your actual grouping column.")
    print(f"Available columns: {df.columns.tolist()}")
    # Use first categorical column as example
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        group_column = categorical_cols[0]
        print(f"Using '{group_column}' as example grouping variable.")
    else:
        print("No categorical columns found. Creating example groups.")
        df['example_group'] = pd.cut(df.iloc[:, 0], bins=3, labels=['Group_A', 'Group_B', 'Group_C'])
        group_column = 'example_group'

# Identify numerical and categorical variables
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove grouping column from analysis columns
if group_column in numerical_cols:
    numerical_cols.remove(group_column)
if group_column in categorical_cols:
    categorical_cols.remove(group_column)

print(f"Grouping variable: {group_column}")
print(f"Groups: {sorted(df[group_column].unique())}")
print(f"Numerical variables to analyze: {len(numerical_cols)}")
print(f"Categorical variables to analyze: {len(categorical_cols)}")

# Group information
group_info = df[group_column].value_counts().sort_index()
print(f"\nGroup sizes:")
for group, count in group_info.items():
    print(f"  {group}: {count} ({count/len(df)*100:.1f}%)")

# 1. NUMERICAL DISTRIBUTIONS BY GROUP
print("\n=== NUMERICAL DISTRIBUTIONS BY GROUP ===")

if numerical_cols:
    # Calculate group statistics for numerical variables
    group_stats = []
    
    for col in numerical_cols[:8]:  # Limit to first 8 for display
        print(f"\nAnalyzing: {col}")
        
        # Group statistics
        stats_by_group = df.groupby(group_column)[col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max', 'skew'
        ]).round(3)
        
        print(stats_by_group)
        
        # Statistical tests
        groups_data = [df[df[group_column] == group][col].dropna() for group in df[group_column].unique()]
        groups_data = [group for group in groups_data if len(group) > 0]  # Remove empty groups
        
        if len(groups_data) >= 2:
            # ANOVA test (for multiple groups)
            if len(groups_data) > 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups_data)
                    print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
                    if p_value < 0.05:
                        print("  *** Significant difference between groups ***")
                    else:
                        print("  No significant difference between groups")
                except:
                    print("  ANOVA test failed")
            
            # Pairwise t-tests (for two groups or pairwise comparison)
            if len(groups_data) == 2:
                try:
                    t_stat, p_value = stats.ttest_ind(groups_data[0], groups_data[1])
                    print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
                    if p_value < 0.05:
                        print("  *** Significant difference between groups ***")
                except:
                    print("  T-test failed")
        
        # Effect size (Cohen's d for two groups)
        if len(groups_data) == 2:
            group1, group2 = groups_data[0], groups_data[1]
            if len(group1) > 1 and len(group2) > 1:
                pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / 
                                   (len(group1) + len(group2) - 2))
                cohens_d = (group1.mean() - group2.mean()) / pooled_std
                print(f"Cohen's d: {cohens_d:.3f}")
                
                if abs(cohens_d) < 0.2:
                    effect_size = "Small"
                elif abs(cohens_d) < 0.8:
                    effect_size = "Medium"
                else:
                    effect_size = "Large"
                print(f"Effect size: {effect_size}")
    
    # Create comprehensive visualizations
    n_plots = min(6, len(numerical_cols))
    if n_plots > 0:
        # Box plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols[:n_plots]):
            # Box plot by group
            df.boxplot(column=col, by=group_column, ax=axes[i])
            axes[i].set_title(f'{col} by {group_column}')
            axes[i].set_xlabel(group_column)
            axes[i].set_ylabel(col)
            
        # Hide empty subplots
        for i in range(n_plots, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Box Plots: Numerical Variables by Group', y=1.02, fontsize=16)
        plt.show()
        
        # Distribution plots (histograms with overlay)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(df[group_column].unique())))
        
        for i, col in enumerate(numerical_cols[:n_plots]):
            for j, group in enumerate(sorted(df[group_column].unique())):
                group_data = df[df[group_column] == group][col].dropna()
                if len(group_data) > 0:
                    axes[i].hist(group_data, alpha=0.6, label=f'{group} (n={len(group_data)})', 
                               color=colors[j], bins=20)
            
            axes[i].set_title(f'{col} Distribution by {group_column}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_plots, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Histograms: Numerical Variables by Group', y=1.02, fontsize=16)
        plt.show()
        
        # Violin plots for distribution shape comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols[:n_plots]):
            data_for_violin = [df[df[group_column] == group][col].dropna() 
                             for group in sorted(df[group_column].unique())]
            
            parts = axes[i].violinplot(data_for_violin, positions=range(len(data_for_violin)))
            axes[i].set_title(f'{col} Distribution Shape by {group_column}')
            axes[i].set_xlabel(group_column)
            axes[i].set_ylabel(col)
            axes[i].set_xticks(range(len(df[group_column].unique())))
            axes[i].set_xticklabels(sorted(df[group_column].unique()), rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_plots, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Violin Plots: Distribution Shapes by Group', y=1.02, fontsize=16)
        plt.show()

# 2. CATEGORICAL DISTRIBUTIONS BY GROUP
print("\n=== CATEGORICAL DISTRIBUTIONS BY GROUP ===")

if categorical_cols:
    for col in categorical_cols[:6]:  # Limit to first 6 categorical variables
        print(f"\nAnalyzing: {col}")
        
        # Create contingency table
        contingency_table = pd.crosstab(df[col], df[group_column])
        print("Contingency Table (counts):")
        print(contingency_table)
        
        # Percentage within each group
        percentage_table = pd.crosstab(df[col], df[group_column], normalize='columns') * 100
        print("\nPercentage within each group:")
        print(percentage_table.round(1))
        
        # Chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            print(f"\nChi-square test: χ²={chi2:.3f}, p={p_value:.3f}, dof={dof}")
            if p_value < 0.05:
                print("  *** Significant association between variables ***")
            else:
                print("  No significant association")
            
            # Cramér's V (effect size for categorical associations)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            print(f"Cramér's V: {cramers_v:.3f}")
            
            if cramers_v < 0.1:
                effect_size = "Negligible"
            elif cramers_v < 0.3:
                effect_size = "Small"
            elif cramers_v < 0.5:
                effect_size = "Medium"
            else:
                effect_size = "Large"
            print(f"Effect size: {effect_size}")
            
        except Exception as e:
            print(f"Chi-square test failed: {e}")
    
    # Visualize categorical distributions by group
    low_cardinality_cats = [col for col in categorical_cols if df[col].nunique() <= 10]
    n_cat_plots = min(4, len(low_cardinality_cats))
    
    if n_cat_plots > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(low_cardinality_cats[:n_cat_plots]):
            # Stacked bar chart
            contingency_table = pd.crosstab(df[col], df[group_column])
            contingency_table.plot(kind='bar', stacked=True, ax=axes[i])
            axes[i].set_title(f'{col} by {group_column} (Stacked)')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].legend(title=group_column)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(n_cat_plots, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Categorical Variables by Group (Stacked Bar Charts)', y=1.02, fontsize=16)
        plt.show()
        
        # Grouped bar charts (side-by-side)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(low_cardinality_cats[:n_cat_plots]):
            contingency_table = pd.crosstab(df[col], df[group_column])
            contingency_table.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{col} by {group_column} (Grouped)')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].legend(title=group_column)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(n_cat_plots, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Categorical Variables by Group (Grouped Bar Charts)', y=1.02, fontsize=16)
        plt.show()

# 3. CORRELATION DIFFERENCES BETWEEN GROUPS
print("\n=== CORRELATION DIFFERENCES BETWEEN GROUPS ===")

if len(numerical_cols) >= 2:
    # Calculate correlation matrices for each group
    for group in sorted(df[group_column].unique()):
        print(f"\nCorrelation matrix for {group}:")
        group_data = df[df[group_column] == group][numerical_cols]
        corr_matrix = group_data.corr()
        print(corr_matrix.round(3))
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title(f'Correlation Matrix: {group}')
        plt.tight_layout()
        plt.show()

# 4. SUMMARY INSIGHTS
print("\n=== SUMMARY INSIGHTS ===")
print("Key findings from group analysis:")
print("1. Check p-values < 0.05 for statistically significant differences")
print("2. Consider effect sizes - statistical significance ≠ practical significance")
print("3. Large effect sizes (Cohen's d > 0.8, Cramér's V > 0.5) indicate meaningful differences")
print("4. Different correlation patterns between groups suggest interaction effects")
print("5. Unequal group variances may require non-parametric tests or transformations")

# Recommendations
print("\n=== RECOMMENDATIONS ===")
print("For significant group differences:")
print("- Use stratified sampling or group-specific models")
print("- Consider group as a feature in predictive modeling")
print("- Apply appropriate statistical tests based on data distribution")
print("- Investigate causes of group differences for domain insights")
print("- Consider subgroup analysis in final reporting")

print("\nAnalysis complete. Review visualizations and statistical tests for actionable insights.")
