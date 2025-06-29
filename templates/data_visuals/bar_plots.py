# Replace 'your_data.csv' with your dataset
# Bar Plot Template - Simple, Grouped, and Stacked Bar Charts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== BAR PLOT VISUALIZATION ===")
print(f"Dataset shape: {df.shape}")

# Get categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# 1. SIMPLE BAR CHART - Frequency count of categorical variable
print("\n=== 1. SIMPLE BAR CHARTS ===")

if categorical_cols:
    # Select categorical columns with reasonable number of categories
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= 15]
    
    if low_cardinality_cols:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(low_cardinality_cols[:4]):
            # Count values
            value_counts = df[col].value_counts()
            
            # Create bar plot
            bars = axes[i].bar(range(len(value_counts)), value_counts.values, 
                              color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            axes[i].set_title(f'{col} Distribution', fontsize=14, fontweight='bold', pad=20)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Count', fontsize=12)
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[i].grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + max(value_counts)*0.01,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            # Add percentage labels
            total = value_counts.sum()
            for j, bar in enumerate(bars):
                height = bar.get_height()
                percentage = (height / total) * 100
                axes[i].text(bar.get_x() + bar.get_width()/2., height/2,
                           f'{percentage:.1f}%', ha='center', va='center', 
                           color='white', fontweight='bold')
        
        # Hide empty subplots
        for i in range(len(low_cardinality_cols), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Simple Bar Charts - Category Frequencies', fontsize=16, y=1.02)
        plt.show()

# 2. HORIZONTAL BAR CHART
print("\n=== 2. HORIZONTAL BAR CHARTS ===")

if categorical_cols:
    col = low_cardinality_cols[0] if low_cardinality_cols else categorical_cols[0]
    value_counts = df[col].value_counts()
    
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(value_counts)), value_counts.values, 
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title(f'{col} Distribution (Horizontal)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.yticks(range(len(value_counts)), value_counts.index)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(value_counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 3. GROUPED BAR CHART - Compare categories across groups
print("\n=== 3. GROUPED BAR CHARTS ===")

if len(categorical_cols) >= 2:
    # Use first two categorical columns for grouping
    cat1, cat2 = categorical_cols[0], categorical_cols[1]
    
    # Create crosstab
    crosstab = pd.crosstab(df[cat1], df[cat2])
    
    # Only proceed if both variables have reasonable cardinality
    if crosstab.shape[0] <= 8 and crosstab.shape[1] <= 6:
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        x = np.arange(len(crosstab.index))
        width = 0.8 / len(crosstab.columns)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(crosstab.columns)))
        
        for i, col in enumerate(crosstab.columns):
            bars = plt.bar(x + i * width, crosstab[col], width, 
                          label=f'{cat2}: {col}', color=colors[i], 
                          alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + crosstab.values.max()*0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        plt.title(f'{cat1} by {cat2} (Grouped)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(cat1, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(x + width * (len(crosstab.columns) - 1) / 2, crosstab.index, rotation=45, ha='right')
        plt.legend(title=cat2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()

# 4. STACKED BAR CHART
print("\n=== 4. STACKED BAR CHARTS ===")

if len(categorical_cols) >= 2:
    plt.figure(figsize=(12, 8))
    
    # Create stacked bar chart
    crosstab.plot(kind='bar', stacked=True, figsize=(12, 8), 
                 colormap='Set3', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title(f'{cat1} by {cat2} (Stacked)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(cat1, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title=cat2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

# 5. PERCENTAGE STACKED BAR CHART
print("\n=== 5. PERCENTAGE STACKED BAR CHARTS ===")

if len(categorical_cols) >= 2:
    plt.figure(figsize=(12, 8))
    
    # Convert to percentages
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    # Create percentage stacked bar chart
    ax = crosstab_pct.plot(kind='bar', stacked=True, figsize=(12, 8), 
                          colormap='Set3', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title(f'{cat1} by {cat2} (Percentage Stacked)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(cat1, fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title=cat2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on segments
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{v:.1f}%' if v > 5 else '' for v in container.datavalues],
                    label_type='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.show()

# 6. BAR CHART WITH NUMERICAL AGGREGATION
print("\n=== 6. BAR CHARTS WITH NUMERICAL AGGREGATION ===")

if categorical_cols and numerical_cols:
    cat_col = categorical_cols[0]
    num_col = numerical_cols[0]
    
    # Calculate aggregated statistics by category
    agg_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'sum', 'count']).reset_index()
    
    # Only proceed if reasonable number of categories
    if len(agg_stats) <= 12:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['mean', 'median', 'sum', 'count']
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
        
        for i, metric in enumerate(metrics):
            bars = axes[i].bar(range(len(agg_stats)), agg_stats[metric], 
                              color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            axes[i].set_title(f'{metric.title()} of {num_col} by {cat_col}', 
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel(cat_col, fontsize=10)
            axes[i].set_ylabel(f'{metric.title()}', fontsize=10)
            axes[i].set_xticks(range(len(agg_stats)))
            axes[i].set_xticklabels(agg_stats[cat_col], rotation=45, ha='right')
            axes[i].grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.suptitle(f'Aggregated Statistics: {num_col} by {cat_col}', fontsize=16, y=1.02)
        plt.show()

# 7. ERROR BARS WITH BAR CHART
print("\n=== 7. BAR CHARTS WITH ERROR BARS ===")

if categorical_cols and numerical_cols:
    cat_col = categorical_cols[0]
    num_col = numerical_cols[0]
    
    # Calculate mean and standard error by category
    stats_df = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count']).reset_index()
    stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])  # Standard error
    
    if len(stats_df) <= 10:
        plt.figure(figsize=(12, 8))
        
        # Create bar chart with error bars
        bars = plt.bar(range(len(stats_df)), stats_df['mean'], 
                      yerr=stats_df['se'], capsize=5, 
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5,
                      error_kw={'ecolor': 'red', 'elinewidth': 2})
        
        # Customize the plot
        plt.title(f'Mean {num_col} by {cat_col} (with Standard Error)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(cat_col, fontsize=12)
        plt.ylabel(f'Mean {num_col}', fontsize=12)
        plt.xticks(range(len(stats_df)), stats_df[cat_col], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + stats_df['se'].iloc[i] + height*0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# 8. SORTED BAR CHART
print("\n=== 8. SORTED BAR CHARTS ===")

if categorical_cols:
    col = categorical_cols[0]
    value_counts = df[col].value_counts().sort_values(ascending=True)  # Sort ascending for better visualization
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart (better for sorted data)
    bars = plt.barh(range(len(value_counts)), value_counts.values, 
                   color='mediumpurple', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title(f'{col} Distribution (Sorted)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.yticks(range(len(value_counts)), value_counts.index)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(value_counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 9. BAR CHART STYLING EXAMPLES
print("\n=== 9. STYLED BAR CHARTS ===")

if categorical_cols:
    col = categorical_cols[0]
    value_counts = df[col].value_counts().head(8)  # Top 8 categories
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Style 1: Gradient colors
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(value_counts)))
    bars1 = axes[0].bar(range(len(value_counts)), value_counts.values, color=colors1, edgecolor='black')
    axes[0].set_title('Gradient Colors', fontweight='bold')
    axes[0].set_xticks(range(len(value_counts)))
    axes[0].set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    # Style 2: Custom color palette
    colors2 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    bars2 = axes[1].bar(range(len(value_counts)), value_counts.values, 
                       color=colors2[:len(value_counts)], edgecolor='white', linewidth=2)
    axes[1].set_title('Custom Palette', fontweight='bold')
    axes[1].set_xticks(range(len(value_counts)))
    axes[1].set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    # Style 3: Different bar widths
    bars3 = axes[2].bar(range(len(value_counts)), value_counts.values, 
                       width=0.6, color='coral', alpha=0.7, edgecolor='navy', linewidth=1.5)
    axes[2].set_title('Custom Width & Edge', fontweight='bold')
    axes[2].set_xticks(range(len(value_counts)))
    axes[2].set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    # Style 4: Pattern fills (using hatching)
    patterns = ['///', '|||', '---', '+++', 'xxx', '...', 'ooo', '***']
    bars4 = axes[3].bar(range(len(value_counts)), value_counts.values, 
                       color='lightblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5,
                       hatch=[patterns[i % len(patterns)] for i in range(len(value_counts))])
    axes[3].set_title('Pattern Fills', fontweight='bold')
    axes[3].set_xticks(range(len(value_counts)))
    axes[3].set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.suptitle('Bar Chart Styling Examples', fontsize=16, y=1.02)
    plt.show()

print("\n=== BAR CHART BEST PRACTICES ===")
print("1. Sort bars by value for better readability (especially for frequency charts)")
print("2. Use horizontal bars when category labels are long")
print("3. Include value labels on bars for precise reading")
print("4. Choose colors that are colorblind-friendly")
print("5. Add grid lines for easier value estimation")
print("6. Keep consistent spacing and bar widths")
print("7. Use grouped bars to compare multiple categories")
print("8. Use stacked bars to show part-to-whole relationships")
print("9. Include error bars when showing means or aggregated values")
print("10. Limit the number of categories to avoid cluttered charts")

print("\nBar plot template complete. All visualizations are report-ready with proper labels and formatting.")
