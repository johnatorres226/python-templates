# Replace 'your_data.csv' with your dataset
# Scatter Plot Template - Basic, with Hue, Size, Fit Lines, and Advanced Options

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== SCATTER PLOT VISUALIZATION ===")
print(f"Dataset shape: {df.shape}")

# Get numerical columns for scatter plots
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

if len(numerical_cols) < 2:
    print("Warning: Need at least 2 numerical columns for scatter plots")
else:
    # 1. BASIC SCATTER PLOTS
    print("\n=== 1. BASIC SCATTER PLOTS ===")
    
    # Create scatter plots for top correlated pairs
    correlations = []
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            col1, col2 = numerical_cols[i], numerical_cols[j]
            clean_data = df[[col1, col2]].dropna()
            if len(clean_data) > 10:
                corr = clean_data[col1].corr(clean_data[col2])
                correlations.append((col1, col2, abs(corr)))
    
    # Sort by correlation and take top pairs
    correlations.sort(key=lambda x: x[2], reverse=True)
    top_pairs = correlations[:4]
    
    if top_pairs:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (col1, col2, corr) in enumerate(top_pairs):
            clean_data = df[[col1, col2]].dropna()
            
            # Basic scatter plot
            axes[i].scatter(clean_data[col1], clean_data[col2], 
                          alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
            
            # Customize plot
            axes[i].set_xlabel(col1, fontsize=12)
            axes[i].set_ylabel(col2, fontsize=12)
            axes[i].set_title(f'{col1} vs {col2}\nCorrelation: {clean_data[col1].corr(clean_data[col2]):.3f}', 
                            fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.suptitle('Basic Scatter Plots - Top Correlated Pairs', fontsize=16, y=1.02)
        plt.show()
    
    # 2. SCATTER PLOT WITH REGRESSION LINE
    print("\n=== 2. SCATTER PLOTS WITH REGRESSION LINES ===")
    
    if top_pairs:
        col1, col2, _ = top_pairs[0]  # Use most correlated pair
        clean_data = df[[col1, col2]].dropna()
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.scatter(clean_data[col1], clean_data[col2], 
                   alpha=0.6, s=60, color='darkblue', edgecolors='white', linewidth=0.5)
        
        # Add linear regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[col1], clean_data[col2])
        line_x = np.array([clean_data[col1].min(), clean_data[col1].max()])
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, 'r-', linewidth=2, label=f'Linear fit (R² = {r_value**2:.3f})')
        
        # Add confidence interval
        x_pred = np.linspace(clean_data[col1].min(), clean_data[col1].max(), 100)
        y_pred = slope * x_pred + intercept
        
        # Calculate prediction interval (approximate)
        residuals = clean_data[col2] - (slope * clean_data[col1] + intercept)
        mse = np.mean(residuals**2)
        prediction_std = np.sqrt(mse)
        
        plt.fill_between(x_pred, y_pred - 1.96*prediction_std, y_pred + 1.96*prediction_std, 
                        alpha=0.2, color='red', label='95% Prediction Interval')
        
        # Customize plot
        plt.xlabel(col1, fontsize=12)
        plt.ylabel(col2, fontsize=12)
        plt.title(f'{col1} vs {col2} with Regression Line\np-value: {p_value:.3e}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 3. SCATTER PLOT WITH HUE (COLOR BY CATEGORY)
    print("\n=== 3. SCATTER PLOTS WITH HUE ===")
    
    if top_pairs and categorical_cols:
        col1, col2, _ = top_pairs[0]
        hue_col = categorical_cols[0]
        
        # Only proceed if hue column has reasonable number of categories
        if df[hue_col].nunique() <= 8:
            plt.figure(figsize=(12, 8))
            
            # Get unique categories and assign colors
            categories = df[hue_col].dropna().unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
            
            # Plot each category separately
            for i, category in enumerate(categories):
                category_data = df[df[hue_col] == category][[col1, col2]].dropna()
                
                if len(category_data) > 0:
                    plt.scatter(category_data[col1], category_data[col2], 
                              alpha=0.7, s=60, color=colors[i], 
                              label=f'{category} (n={len(category_data)})',
                              edgecolors='black', linewidth=0.5)
            
            # Customize plot
            plt.xlabel(col1, fontsize=12)
            plt.ylabel(col2, fontsize=12)
            plt.title(f'{col1} vs {col2} colored by {hue_col}', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # 4. SCATTER PLOT WITH SIZE MAPPING
    print("\n=== 4. SCATTER PLOTS WITH SIZE MAPPING ===")
    
    if len(numerical_cols) >= 3:
        col1, col2 = numerical_cols[0], numerical_cols[1]
        size_col = numerical_cols[2]
        
        clean_data = df[[col1, col2, size_col]].dropna()
        
        if len(clean_data) > 0:
            plt.figure(figsize=(12, 8))
            
            # Normalize sizes for better visualization
            sizes = clean_data[size_col]
            normalized_sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min())) * 300 + 20
            
            # Create scatter plot with size mapping
            scatter = plt.scatter(clean_data[col1], clean_data[col2], 
                                s=normalized_sizes, alpha=0.6, 
                                c=clean_data[size_col], cmap='viridis',
                                edgecolors='black', linewidth=0.5)
            
            # Add colorbar for size reference
            cbar = plt.colorbar(scatter)
            cbar.set_label(size_col, fontsize=12)
            
            # Customize plot
            plt.xlabel(col1, fontsize=12)
            plt.ylabel(col2, fontsize=12)
            plt.title(f'{col1} vs {col2}\nPoint size and color represent {size_col}', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # 5. SCATTER PLOT MATRIX (PAIRPLOT)
    print("\n=== 5. SCATTER PLOT MATRIX ===")
    
    # Select top numerical columns (limit to 5 for readability)
    top_num_cols = numerical_cols[:5]
    
    if len(top_num_cols) >= 2:
        # Create pairplot data
        pairplot_data = df[top_num_cols].dropna()
        
        if len(pairplot_data) > 0:
            # Using matplotlib for custom pairplot
            n_vars = len(top_num_cols)
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(3*n_vars, 3*n_vars))
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        # Diagonal: histogram
                        axes[i, j].hist(pairplot_data[top_num_cols[i]], bins=20, 
                                       alpha=0.7, color='skyblue', edgecolor='black')
                        axes[i, j].set_title(f'{top_num_cols[i]} Distribution')
                    else:
                        # Off-diagonal: scatter plot
                        axes[i, j].scatter(pairplot_data[top_num_cols[j]], 
                                          pairplot_data[top_num_cols[i]], 
                                          alpha=0.5, s=20, color='steelblue')
                        
                        # Add correlation
                        corr = pairplot_data[top_num_cols[j]].corr(pairplot_data[top_num_cols[i]])
                        axes[i, j].text(0.05, 0.95, f'r = {corr:.3f}', 
                                       transform=axes[i, j].transAxes, 
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Set labels
                    if i == n_vars - 1:
                        axes[i, j].set_xlabel(top_num_cols[j])
                    if j == 0:
                        axes[i, j].set_ylabel(top_num_cols[i])
                    
                    axes[i, j].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.suptitle('Scatter Plot Matrix', fontsize=16, y=1.02)
            plt.show()
    
    # 6. POLYNOMIAL FIT SCATTER PLOT
    print("\n=== 6. SCATTER PLOT WITH POLYNOMIAL FIT ===")
    
    if top_pairs:
        col1, col2, _ = top_pairs[0]
        clean_data = df[[col1, col2]].dropna()
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.scatter(clean_data[col1], clean_data[col2], 
                   alpha=0.6, s=60, color='darkgreen', edgecolors='white', linewidth=0.5)
        
        # Fit polynomials of different degrees
        x_smooth = np.linspace(clean_data[col1].min(), clean_data[col1].max(), 100)
        colors = ['red', 'blue', 'orange']
        
        for degree, color in zip([1, 2, 3], colors):
            # Fit polynomial
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(clean_data[col1].values.reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(X_poly, clean_data[col2])
            
            # Predict
            X_smooth_poly = poly_features.transform(x_smooth.reshape(-1, 1))
            y_smooth = model.predict(X_smooth_poly)
            
            # Calculate R²
            r2 = model.score(X_poly, clean_data[col2])
            
            plt.plot(x_smooth, y_smooth, color=color, linewidth=2, 
                    label=f'Degree {degree} (R² = {r2:.3f})')
        
        # Customize plot
        plt.xlabel(col1, fontsize=12)
        plt.ylabel(col2, fontsize=12)
        plt.title(f'{col1} vs {col2} with Polynomial Fits', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 7. ANNOTATED SCATTER PLOT
    print("\n=== 7. ANNOTATED SCATTER PLOT ===")
    
    if top_pairs and len(df) <= 50:  # Only annotate if not too many points
        col1, col2, _ = top_pairs[0]
        clean_data = df[[col1, col2]].dropna()
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.scatter(clean_data[col1], clean_data[col2], 
                   alpha=0.7, s=80, color='purple', edgecolors='black', linewidth=1)
        
        # Add annotations for outliers or interesting points
        # Find outliers using IQR method
        Q1_x, Q3_x = clean_data[col1].quantile([0.25, 0.75])
        Q1_y, Q3_y = clean_data[col2].quantile([0.25, 0.75])
        IQR_x, IQR_y = Q3_x - Q1_x, Q3_y - Q1_y
        
        outlier_mask = ((clean_data[col1] < Q1_x - 1.5*IQR_x) | 
                       (clean_data[col1] > Q3_x + 1.5*IQR_x) |
                       (clean_data[col2] < Q1_y - 1.5*IQR_y) | 
                       (clean_data[col2] > Q3_y + 1.5*IQR_y))
        
        outliers = clean_data[outlier_mask]
        
        # Annotate outliers
        for idx, row in outliers.iterrows():
            plt.annotate(f'Point {idx}', 
                        xy=(row[col1], row[col2]), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Customize plot
        plt.xlabel(col1, fontsize=12)
        plt.ylabel(col2, fontsize=12)
        plt.title(f'{col1} vs {col2} with Outlier Annotations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 8. MULTIPLE SCATTER PLOTS WITH SUBPLOTS
    print("\n=== 8. MULTIPLE SCATTER PLOTS BY CATEGORY ===")
    
    if categorical_cols and top_pairs:
        hue_col = categorical_cols[0]
        
        if df[hue_col].nunique() <= 6:  # Reasonable number of categories
            col1, col2, _ = top_pairs[0]
            categories = sorted(df[hue_col].dropna().unique())
            
            n_cats = len(categories)
            n_cols = min(3, n_cats)
            n_rows = (n_cats + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, category in enumerate(categories):
                row = i // n_cols
                col_idx = i % n_cols
                
                category_data = df[df[hue_col] == category][[col1, col2]].dropna()
                
                if len(category_data) > 0:
                    # Scatter plot for this category
                    axes[row][col_idx].scatter(category_data[col1], category_data[col2], 
                                             alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
                    
                    # Add correlation
                    if len(category_data) > 2:
                        corr = category_data[col1].corr(category_data[col2])
                        axes[row][col_idx].text(0.05, 0.95, f'r = {corr:.3f}', 
                                               transform=axes[row][col_idx].transAxes,
                                               verticalalignment='top',
                                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Customize subplot
                    axes[row][col_idx].set_xlabel(col1)
                    axes[row][col_idx].set_ylabel(col2)
                    axes[row][col_idx].set_title(f'{category} (n={len(category_data)})')
                    axes[row][col_idx].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_cats, n_rows * n_cols):
                row = i // n_cols
                col_idx = i % n_cols
                axes[row][col_idx].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle(f'{col1} vs {col2} by {hue_col}', fontsize=16, y=1.02)
            plt.show()

print("\n=== SCATTER PLOT BEST PRACTICES ===")
print("1. Use transparency (alpha) when points overlap")
print("2. Add regression lines to show relationships")
print("3. Use color (hue) to show additional categorical dimensions")
print("4. Use size to represent a third numerical variable")
print("5. Include correlation coefficients for quantitative relationships")
print("6. Annotate outliers or interesting points")
print("7. Add confidence/prediction intervals for regression lines")
print("8. Use jittering for discrete variables to avoid overplotting")
print("9. Consider log scales for highly skewed data")
print("10. Keep aspect ratios reasonable to avoid distorting relationships")

print("\nScatter plot template complete. All visualizations include proper styling and statistical information.")
