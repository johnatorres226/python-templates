# Replace 'your_data.csv' with your dataset
# Linear Regression Template - Comprehensive Analysis with Diagnostics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== LINEAR REGRESSION ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Replace 'target_column' with your actual target variable
target_column = 'target_column'  # Replace with your target variable name

# If target column doesn't exist, use first numerical column as example
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_column not in df.columns and numerical_cols:
    target_column = numerical_cols[0]
    print(f"Using '{target_column}' as example target variable")

if target_column not in df.columns:
    print("Error: No suitable target variable found. Please specify a numerical target column.")
else:
    # Get feature columns (all numerical except target)
    feature_cols = [col for col in numerical_cols if col != target_column]
    
    if len(feature_cols) == 0:
        print("Error: No numerical features found for regression.")
    else:
        print(f"Target variable: {target_column}")
        print(f"Feature variables: {feature_cols}")
        
        # Prepare data (remove missing values)
        analysis_cols = [target_column] + feature_cols
        clean_data = df[analysis_cols].dropna()
        
        print(f"Clean data shape: {clean_data.shape}")
        
        if len(clean_data) < 10:
            print("Error: Insufficient data for regression analysis after removing missing values.")
        else:
            X = clean_data[feature_cols]
            y = clean_data[target_column]
            
            # 1. EXPLORATORY DATA ANALYSIS
            print("\n=== 1. EXPLORATORY DATA ANALYSIS ===")
            
            # Target variable statistics
            print(f"Target variable ({target_column}) statistics:")
            print(f"Mean: {y.mean():.3f}")
            print(f"Std: {y.std():.3f}")
            print(f"Min: {y.min():.3f}")
            print(f"Max: {y.max():.3f}")
            print(f"Skewness: {stats.skew(y):.3f}")
            print(f"Kurtosis: {stats.kurtosis(y):.3f}")
            
            # Correlation with target
            print(f"\nCorrelations with {target_column}:")
            correlations = X.corrwith(y).sort_values(key=abs, ascending=False)
            print(correlations)
            
            # Visualize target distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'{target_column} Distribution')
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 2)
            stats.probplot(y, dist="norm", plot=plt)
            plt.title('Q-Q Plot (Normality Check)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            plt.boxplot(y)
            plt.title(f'{target_column} Box Plot')
            plt.ylabel(target_column)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # 2. CORRELATION ANALYSIS
            print("\n=== 2. CORRELATION ANALYSIS ===")
            
            # Correlation matrix
            corr_matrix = clean_data[analysis_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                       square=True, fmt='.3f')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
            
            # Check for multicollinearity (VIF)
            print("Variance Inflation Factors (VIF):")
            vif_data = pd.DataFrame()
            vif_data["Feature"] = feature_cols
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(feature_cols))]
            print(vif_data)
            print("VIF > 10 indicates potential multicollinearity issues")
            
            # 3. TRAIN-TEST SPLIT
            print("\n=== 3. TRAIN-TEST SPLIT ===")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            
            # 4. MODEL TRAINING
            print("\n=== 4. MODEL TRAINING ===")
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Model coefficients
            print("Model Coefficients:")
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            print(coef_df)
            print(f"Intercept: {model.intercept_:.3f}")
            
            # 5. MODEL EVALUATION
            print("\n=== 5. MODEL EVALUATION ===")
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Training metrics
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            # Test metrics
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            print("Model Performance:")
            print(f"Training R²: {train_r2:.3f}")
            print(f"Test R²: {test_r2:.3f}")
            print(f"Training RMSE: {train_rmse:.3f}")
            print(f"Test RMSE: {test_rmse:.3f}")
            print(f"Training MAE: {train_mae:.3f}")
            print(f"Test MAE: {test_mae:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            print(f"Cross-validation R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # 6. STATISTICAL SIGNIFICANCE (using statsmodels)
            print("\n=== 6. STATISTICAL SIGNIFICANCE ===")
            
            # Add constant for intercept
            X_train_sm = sm.add_constant(X_train)
            
            # Fit OLS model
            ols_model = sm.OLS(y_train, X_train_sm).fit()
            print(ols_model.summary())
            
            # 7. RESIDUAL ANALYSIS
            print("\n=== 7. RESIDUAL ANALYSIS ===")
            
            # Calculate residuals
            train_residuals = y_train - y_train_pred
            test_residuals = y_test - y_test_pred
            
            # Residual plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Residuals vs Fitted
            axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted (Training)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Q-Q plot of residuals
            stats.probplot(train_residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot of Residuals')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Histogram of residuals
            axes[0, 2].hist(train_residuals, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 2].set_xlabel('Residuals')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Residual Distribution')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Scale-Location plot
            sqrt_abs_resid = np.sqrt(np.abs(train_residuals))
            axes[1, 0].scatter(y_train_pred, sqrt_abs_resid, alpha=0.6)
            axes[1, 0].set_xlabel('Fitted Values')
            axes[1, 0].set_ylabel('√|Residuals|')
            axes[1, 0].set_title('Scale-Location Plot')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Actual vs Predicted
            min_val = min(y_train.min(), y_train_pred.min())
            max_val = max(y_train.max(), y_train_pred.max())
            axes[1, 1].scatter(y_train, y_train_pred, alpha=0.6)
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[1, 1].set_xlabel('Actual Values')
            axes[1, 1].set_ylabel('Predicted Values')
            axes[1, 1].set_title('Actual vs Predicted (Training)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Test set: Actual vs Predicted
            min_val_test = min(y_test.min(), y_test_pred.min())
            max_val_test = max(y_test.max(), y_test_pred.max())
            axes[1, 2].scatter(y_test, y_test_pred, alpha=0.6, color='orange')
            axes[1, 2].plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', linewidth=2)
            axes[1, 2].set_xlabel('Actual Values')
            axes[1, 2].set_ylabel('Predicted Values')
            axes[1, 2].set_title('Actual vs Predicted (Test)')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # 8. ASSUMPTION TESTING
            print("\n=== 8. ASSUMPTION TESTING ===")
            
            # Normality of residuals (Shapiro-Wilk test)
            if len(train_residuals) <= 5000:  # Shapiro-Wilk has sample size limits
                shapiro_stat, shapiro_p = stats.shapiro(train_residuals)
                print(f"Shapiro-Wilk test for normality of residuals:")
                print(f"Statistic: {shapiro_stat:.3f}, p-value: {shapiro_p:.3f}")
                if shapiro_p > 0.05:
                    print("✓ Residuals appear normally distributed")
                else:
                    print("⚠ Residuals may not be normally distributed")
            
            # Homoscedasticity (Breusch-Pagan test)
            X_train_const = sm.add_constant(X_train)
            bp_stat, bp_p, _, _ = het_breuschpagan(train_residuals, X_train_const)
            print(f"\nBreusch-Pagan test for homoscedasticity:")
            print(f"Statistic: {bp_stat:.3f}, p-value: {bp_p:.3f}")
            if bp_p > 0.05:
                print("✓ Homoscedasticity assumption satisfied")
            else:
                print("⚠ Heteroscedasticity detected")
            
            # White test for heteroscedasticity
            white_stat, white_p, _, _ = het_white(train_residuals, X_train_const)
            print(f"\nWhite test for heteroscedasticity:")
            print(f"Statistic: {white_stat:.3f}, p-value: {white_p:.3f}")
            if white_p > 0.05:
                print("✓ Homoscedasticity assumption satisfied (White test)")
            else:
                print("⚠ Heteroscedasticity detected (White test)")
            
            # 9. FEATURE IMPORTANCE
            print("\n=== 9. FEATURE IMPORTANCE ===")
            
            # Standardized coefficients for importance comparison
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model_scaled = LinearRegression()
            model_scaled.fit(X_train_scaled, y_train)
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_),
                'Standardized_Coefficient': model_scaled.coef_
            }).sort_values('Abs_Coefficient', ascending=False)
            
            print("Feature Importance (by absolute coefficient):")
            print(importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Abs_Coefficient'])
            plt.xlabel('Absolute Coefficient Value')
            plt.title('Feature Importance (Linear Regression)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # 10. MODEL INTERPRETATION
            print("\n=== 10. MODEL INTERPRETATION ===")
            
            print("Model Equation:")
            equation = f"{target_column} = {model.intercept_:.3f}"
            for feature, coef in zip(feature_cols, model.coef_):
                equation += f" + ({coef:.3f} × {feature})"
            print(equation)
            
            print("\nCoefficient Interpretation:")
            for feature, coef in zip(feature_cols, model.coef_):
                print(f"- {feature}: A 1-unit increase is associated with a {coef:.3f} change in {target_column}")
            
            # 11. PREDICTION INTERVALS
            print("\n=== 11. PREDICTION INTERVALS ===")
            
            # Calculate prediction intervals for test set
            X_test_const = sm.add_constant(X_test)
            ols_predictions = ols_model.get_prediction(X_test_const)
            prediction_summary = ols_predictions.summary_frame(alpha=0.05)  # 95% intervals
            
            print("Sample predictions with 95% confidence intervals:")
            sample_predictions = pd.DataFrame({
                'Actual': y_test.values[:5],
                'Predicted': prediction_summary['mean'][:5],
                'Lower_CI': prediction_summary['obs_ci_lower'][:5],
                'Upper_CI': prediction_summary['obs_ci_upper'][:5]
            })
            print(sample_predictions)
            
            # SUMMARY AND RECOMMENDATIONS
            print("\n=== SUMMARY AND RECOMMENDATIONS ===")
            print(f"Model Performance: R² = {test_r2:.3f} (explains {test_r2*100:.1f}% of variance)")
            print(f"Model Error: RMSE = {test_rmse:.3f}, MAE = {test_mae:.3f}")
            
            print("\nDiagnostic Summary:")
            if test_r2 > 0.7:
                print("✓ Good model fit (R² > 0.7)")
            elif test_r2 > 0.5:
                print("○ Moderate model fit (R² > 0.5)")
            else:
                print("⚠ Poor model fit (R² < 0.5)")
            
            if abs(train_r2 - test_r2) < 0.1:
                print("✓ No significant overfitting detected")
            else:
                print("⚠ Potential overfitting (large train-test R² difference)")
            
            print("\nNext Steps:")
            print("1. If assumptions are violated, consider:")
            print("   - Transforming variables (log, square root)")
            print("   - Adding polynomial features")
            print("   - Using robust regression methods")
            print("2. If multicollinearity is high, consider:")
            print("   - Removing correlated features")
            print("   - Using regularization (Ridge, Lasso)")
            print("3. If model fit is poor, consider:")
            print("   - Adding more relevant features")
            print("   - Trying non-linear models")
            print("   - Checking for outliers and influential points")
            
            print("\nLinear regression analysis complete.")
