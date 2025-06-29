# Replace 'your_data.csv' with your dataset
# Inferential Statistics Template - T-tests, Chi-square, ANOVA, Confidence Intervals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency, f_oneway
import statsmodels.stats.api as sms
from statsmodels.stats.power import ttest_power

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== INFERENTIAL STATISTICS ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Get numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical variables: {len(numerical_cols)}")
print(f"Categorical variables: {len(categorical_cols)}")

# 1. ONE-SAMPLE T-TEST
print("\n=== ONE-SAMPLE T-TESTS ===")
print("Testing if sample means differ significantly from hypothesized population values")

# Example: Test if mean of numerical variables differs from specific values
# Replace these with your actual hypotheses
test_values = {
    # 'column_name': hypothesized_value
    # 'age': 30,
    # 'income': 50000,
    # 'score': 75
}

# If no specific hypotheses, test against overall mean
if not test_values and numerical_cols:
    # Use first numerical column as example
    col = numerical_cols[0]
    test_values[col] = df[col].mean() + 5  # Test if mean is significantly different from mean + 5

for col, test_value in test_values.items():
    if col in df.columns:
        data = df[col].dropna()
        
        if len(data) > 0:
            print(f"\nOne-sample t-test for {col}:")
            print(f"H0: μ = {test_value}")
            print(f"H1: μ ≠ {test_value}")
            
            # Perform one-sample t-test
            t_stat, p_value = ttest_1samp(data, test_value)
            
            # Calculate confidence interval
            confidence_level = 0.95
            degrees_freedom = len(data) - 1
            t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
            margin_error = t_critical * (data.std() / np.sqrt(len(data)))
            ci_lower = data.mean() - margin_error
            ci_upper = data.mean() + margin_error
            
            # Effect size (Cohen's d)
            cohens_d = (data.mean() - test_value) / data.std()
            
            print(f"Sample mean: {data.mean():.3f}")
            print(f"Sample std: {data.std():.3f}")
            print(f"Sample size: {len(data)}")
            print(f"t-statistic: {t_stat:.3f}")
            print(f"p-value: {p_value:.3f}")
            print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"Cohen's d: {cohens_d:.3f}")
            
            if p_value < 0.05:
                print("*** SIGNIFICANT: Reject H0 - Mean significantly differs from test value ***")
            else:
                print("Not significant: Fail to reject H0")

# 2. TWO-SAMPLE T-TESTS (INDEPENDENT)
print("\n=== TWO-SAMPLE T-TESTS (INDEPENDENT) ===")
print("Comparing means between two independent groups")

# Replace 'group_column' with your actual grouping variable
group_column = 'group_column'  # Replace with your grouping variable

# If group column doesn't exist, create example groups
if group_column not in df.columns and categorical_cols:
    group_column = categorical_cols[0]
    print(f"Using '{group_column}' as grouping variable")
elif group_column not in df.columns:
    # Create binary groups based on median split of first numerical column
    if numerical_cols:
        median_val = df[numerical_cols[0]].median()
        df['example_group'] = df[numerical_cols[0]].apply(lambda x: 'High' if x > median_val else 'Low')
        group_column = 'example_group'

if group_column in df.columns:
    # Get unique groups (limit to first 2 for two-sample test)
    unique_groups = df[group_column].dropna().unique()
    
    if len(unique_groups) >= 2:
        group1_name, group2_name = unique_groups[0], unique_groups[1]
        
        for col in numerical_cols[:5]:  # Test first 5 numerical columns
            print(f"\nTwo-sample t-test for {col} by {group_column}:")
            
            # Split data by groups
            group1_data = df[df[group_column] == group1_name][col].dropna()
            group2_data = df[df[group_column] == group2_name][col].dropna()
            
            if len(group1_data) > 1 and len(group2_data) > 1:
                print(f"H0: μ_{group1_name} = μ_{group2_name}")
                print(f"H1: μ_{group1_name} ≠ μ_{group2_name}")
                
                # Check for equal variances (Levene's test)
                levene_stat, levene_p = stats.levene(group1_data, group2_data)
                equal_var = levene_p > 0.05
                
                # Perform two-sample t-test
                t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=equal_var)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1_data)-1)*group1_data.std()**2 + 
                                     (len(group2_data)-1)*group2_data.std()**2) / 
                                    (len(group1_data) + len(group2_data) - 2))
                cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                print(f"{group1_name} mean: {group1_data.mean():.3f} (n={len(group1_data)})")
                print(f"{group2_name} mean: {group2_data.mean():.3f} (n={len(group2_data)})")
                print(f"Levene's test p-value: {levene_p:.3f} (Equal variances: {equal_var})")
                print(f"t-statistic: {t_stat:.3f}")
                print(f"p-value: {p_value:.3f}")
                print(f"Cohen's d: {cohens_d:.3f}")
                
                if p_value < 0.05:
                    print("*** SIGNIFICANT: Groups have significantly different means ***")
                else:
                    print("Not significant: No significant difference between groups")

# 3. PAIRED T-TEST
print("\n=== PAIRED T-TEST ===")
print("For comparing before/after or matched pairs data")

# Example: If you have before/after measurements
# Replace with your actual paired columns
paired_tests = [
    # ('before_column', 'after_column'),
    # ('pre_test', 'post_test'),
    # ('baseline', 'followup')
]

# If no paired columns specified, create example with correlated numerical columns
if not paired_tests and len(numerical_cols) >= 2:
    # Find pairs of columns with correlation > 0.3
    for i in range(len(numerical_cols)-1):
        for j in range(i+1, len(numerical_cols)):
            col1, col2 = numerical_cols[i], numerical_cols[j]
            clean_data = df[[col1, col2]].dropna()
            if len(clean_data) > 10:
                corr = clean_data[col1].corr(clean_data[col2])
                if abs(corr) > 0.3:
                    paired_tests.append((col1, col2))
                    break
        if paired_tests:
            break

for col1, col2 in paired_tests[:3]:  # Limit to 3 tests
    if col1 in df.columns and col2 in df.columns:
        print(f"\nPaired t-test: {col1} vs {col2}")
        
        # Get paired data (remove rows with missing values in either column)
        paired_data = df[[col1, col2]].dropna()
        
        if len(paired_data) > 1:
            data1 = paired_data[col1]
            data2 = paired_data[col2]
            differences = data1 - data2
            
            print(f"H0: μ_difference = 0")
            print(f"H1: μ_difference ≠ 0")
            
            # Perform paired t-test
            t_stat, p_value = ttest_rel(data1, data2)
            
            # Effect size (Cohen's d for paired data)
            cohens_d = differences.mean() / differences.std()
            
            print(f"{col1} mean: {data1.mean():.3f}")
            print(f"{col2} mean: {data2.mean():.3f}")
            print(f"Mean difference: {differences.mean():.3f}")
            print(f"Std of differences: {differences.std():.3f}")
            print(f"Sample size: {len(paired_data)}")
            print(f"t-statistic: {t_stat:.3f}")
            print(f"p-value: {p_value:.3f}")
            print(f"Cohen's d: {cohens_d:.3f}")
            
            if p_value < 0.05:
                print("*** SIGNIFICANT: Paired measurements are significantly different ***")
            else:
                print("Not significant: No significant difference between paired measurements")

# 4. CHI-SQUARE TEST OF INDEPENDENCE
print("\n=== CHI-SQUARE TESTS OF INDEPENDENCE ===")
print("Testing association between categorical variables")

# Test associations between categorical variables
for i, cat1 in enumerate(categorical_cols[:3]):
    for cat2 in categorical_cols[i+1:4]:  # Limit comparisons
        if cat1 != cat2:
            print(f"\nChi-square test: {cat1} vs {cat2}")
            
            # Create contingency table
            contingency_table = pd.crosstab(df[cat1], df[cat2])
            
            # Only proceed if we have sufficient data
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                print(f"H0: {cat1} and {cat2} are independent")
                print(f"H1: {cat1} and {cat2} are associated")
                
                # Perform chi-square test
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Effect size (Cramér's V)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                
                print(f"Contingency table shape: {contingency_table.shape}")
                print(f"Chi-square statistic: {chi2_stat:.3f}")
                print(f"Degrees of freedom: {dof}")
                print(f"p-value: {p_value:.3f}")
                print(f"Cramér's V: {cramers_v:.3f}")
                
                # Check assumptions (expected frequencies >= 5)
                min_expected = expected.min()
                cells_below_5 = (expected < 5).sum().sum()
                total_cells = expected.size
                
                print(f"Minimum expected frequency: {min_expected:.2f}")
                print(f"Cells with expected < 5: {cells_below_5}/{total_cells}")
                
                if cells_below_5 / total_cells > 0.2:
                    print("⚠️ Warning: >20% of cells have expected frequency < 5")
                
                if p_value < 0.05:
                    print("*** SIGNIFICANT: Variables are significantly associated ***")
                else:
                    print("Not significant: No significant association")

# 5. ONE-WAY ANOVA
print("\n=== ONE-WAY ANOVA ===")
print("Comparing means across multiple groups")

if group_column in df.columns:
    unique_groups = df[group_column].dropna().unique()
    
    if len(unique_groups) >= 3:  # ANOVA needs at least 3 groups
        for col in numerical_cols[:3]:  # Test first 3 numerical columns
            print(f"\nOne-way ANOVA: {col} by {group_column}")
            
            # Prepare group data
            group_data = []
            group_stats = []
            
            for group in unique_groups:
                data = df[df[group_column] == group][col].dropna()
                if len(data) > 1:  # Need at least 2 observations per group
                    group_data.append(data)
                    group_stats.append({
                        'group': group,
                        'n': len(data),
                        'mean': data.mean(),
                        'std': data.std()
                    })
            
            if len(group_data) >= 3:
                print(f"H0: All group means are equal")
                print(f"H1: At least one group mean differs")
                
                # Perform one-way ANOVA
                f_stat, p_value = f_oneway(*group_data)
                
                # Calculate effect size (eta squared)
                # SSB (sum of squares between groups)
                overall_mean = np.concatenate(group_data).mean()
                ssb = sum(len(group) * (group.mean() - overall_mean)**2 for group in group_data)
                
                # SSW (sum of squares within groups)
                ssw = sum(sum((x - group.mean())**2) for group in group_data for x in group)
                
                # SST (total sum of squares)
                sst = ssb + ssw
                eta_squared = ssb / sst if sst > 0 else 0
                
                print("Group Statistics:")
                for stat in group_stats:
                    print(f"  {stat['group']}: n={stat['n']}, mean={stat['mean']:.3f}, std={stat['std']:.3f}")
                
                print(f"F-statistic: {f_stat:.3f}")
                print(f"p-value: {p_value:.3f}")
                print(f"Eta squared (η²): {eta_squared:.3f}")
                
                if p_value < 0.05:
                    print("*** SIGNIFICANT: At least one group mean differs ***")
                    print("Consider post-hoc tests (e.g., Tukey HSD) to identify which groups differ")
                else:
                    print("Not significant: No significant differences between group means")

# 6. CONFIDENCE INTERVALS
print("\n=== CONFIDENCE INTERVALS ===")
print("95% Confidence intervals for means")

confidence_level = 0.95
alpha = 1 - confidence_level

for col in numerical_cols[:5]:  # First 5 numerical columns
    data = df[col].dropna()
    
    if len(data) > 1:
        print(f"\n{col}:")
        
        # Sample statistics
        sample_mean = data.mean()
        sample_std = data.std()
        n = len(data)
        
        # Standard error
        se = sample_std / np.sqrt(n)
        
        # t-critical value
        df_t = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, df_t)
        
        # Margin of error
        margin_error = t_critical * se
        
        # Confidence interval
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        print(f"Sample mean: {sample_mean:.3f}")
        print(f"Standard error: {se:.3f}")
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"Interpretation: We are 95% confident the true population mean is between {ci_lower:.3f} and {ci_upper:.3f}")

# 7. POWER ANALYSIS
print("\n=== POWER ANALYSIS ===")
print("Statistical power for t-tests")

# Example power analysis for two-sample t-test
if len(numerical_cols) > 0:
    # Use first numerical column as example
    col = numerical_cols[0]
    data = df[col].dropna()
    
    if len(data) > 10:
        sample_std = data.std()
        
        print(f"\nPower analysis for {col}:")
        print("Scenario: Two-sample t-test with equal sample sizes")
        
        # Calculate power for different effect sizes
        effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large effects
        sample_sizes = [10, 20, 50, 100]
        
        print("\nPower analysis results:")
        print("Effect Size | n per group | Power")
        print("-" * 35)
        
        for effect_size in effect_sizes:
            for n in sample_sizes:
                power = ttest_power(effect_size, n, alpha=0.05)
                print(f"   {effect_size:4.1f}     |    {n:3d}      | {power:5.3f}")

# SUMMARY
print("\n=== INFERENTIAL STATISTICS SUMMARY ===")
print("Key considerations:")
print("1. Check assumptions before applying tests (normality, equal variances, independence)")
print("2. p < 0.05 indicates statistical significance (adjust for multiple comparisons)")
print("3. Effect sizes indicate practical significance:")
print("   - Cohen's d: 0.2 (small), 0.5 (medium), 0.8 (large)")
print("   - Cramér's V: 0.1 (small), 0.3 (medium), 0.5 (large)")
print("   - Eta squared: 0.01 (small), 0.06 (medium), 0.14 (large)")
print("4. Consider sample size and power when interpreting results")
print("5. Confidence intervals provide range of plausible values")

print("\nInferential statistics analysis complete.")
