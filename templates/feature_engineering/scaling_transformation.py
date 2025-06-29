"""
===============================================================================
SCALING AND TRANSFORMATION TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Advanced scaling and transformation techniques for feature engineering

This template covers:
- Standard scaling and normalization methods
- Robust scaling techniques
- Power transformations (Box-Cox, Yeo-Johnson)
- Quantile transformations
- Custom scaling methods
- Transformation validation and selection

Prerequisites:
- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, Normalizer
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# LOAD AND PREPARE DATA
# ===============================================================================

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Sample dataset creation with various distributions
np.random.seed(42)
n_samples = 1000

# Create features with different distributions
df = pd.DataFrame({
    'normal_feature': np.random.normal(50, 15, n_samples),
    'skewed_right': np.random.exponential(2, n_samples),
    'skewed_left': 10 - np.random.exponential(1, n_samples),
    'uniform_feature': np.random.uniform(0, 100, n_samples),
    'bimodal_feature': np.concatenate([
        np.random.normal(20, 5, n_samples//2),
        np.random.normal(80, 5, n_samples//2)
    ]),
    'outlier_feature': np.concatenate([
        np.random.normal(50, 10, int(0.95 * n_samples)),
        np.random.normal(200, 20, int(0.05 * n_samples))
    ]),
    'categorical_numeric': np.random.choice([1, 2, 3, 4, 5], n_samples),
    'zero_inflated': np.where(
        np.random.random(n_samples) < 0.3, 
        0, 
        np.random.exponential(5, n_samples)
    )
})

# Add target variable (regression)
df['target'] = (
    0.5 * df['normal_feature'] + 
    0.3 * np.log1p(df['skewed_right']) + 
    0.2 * df['uniform_feature'] + 
    np.random.normal(0, 10, n_samples)
)

print("Dataset Shape:", df.shape)
print("\nFeature Distributions Summary:")
print(df.describe())

# Visualize original distributions
plt.figure(figsize=(20, 15))
features = df.columns[:-1]  # Exclude target

for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    plt.hist(df[feature], bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{feature}\nSkewness: {df[feature].skew():.3f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# ===============================================================================
# 1. STANDARD SCALING METHODS
# ===============================================================================

print("\n" + "="*60)
print("1. STANDARD SCALING METHODS")
print("="*60)

# Split data for transformation
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define scaling methods
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'MaxAbsScaler': MaxAbsScaler(),
    'Normalizer_L2': Normalizer(norm='l2'),
    'Normalizer_L1': Normalizer(norm='l1')
}

# Apply scaling methods
scaled_results = {}
for scaler_name, scaler in scalers.items():
    print(f"Applying {scaler_name}...")
    
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    scaled_results[scaler_name] = {
        'scaler': scaler,
        'X_train': X_train_scaled_df,
        'X_test': X_test_scaled_df
    }

# Visualize scaling effects
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

feature_to_plot = 'skewed_right'  # Choose a skewed feature for demonstration

for i, (scaler_name, results) in enumerate(scaled_results.items()):
    if i < 9:  # We have 9 subplot positions
        ax = axes[i]
        
        original_data = X_train[feature_to_plot]
        scaled_data = results['X_train'][feature_to_plot]
        
        ax.hist(original_data, alpha=0.5, bins=30, label='Original', density=True)
        ax.hist(scaled_data, alpha=0.5, bins=30, label='Scaled', density=True)
        ax.set_title(f'{scaler_name}\n{feature_to_plot}')
        ax.legend()
        
        # Add statistics
        ax.text(0.05, 0.95, f'Original: μ={original_data.mean():.2f}, σ={original_data.std():.2f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=8)
        ax.text(0.05, 0.85, f'Scaled: μ={scaled_data.mean():.2f}, σ={scaled_data.std():.2f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=8)

plt.tight_layout()
plt.show()

# ===============================================================================
# 2. POWER TRANSFORMATIONS
# ===============================================================================

print("\n" + "="*60)
print("2. POWER TRANSFORMATIONS")
print("="*60)

def apply_power_transformations(data):
    """Apply various power transformations"""
    results = {}
    
    for column in data.columns:
        feature_data = data[column].copy()
        
        # Skip if data contains non-positive values for Box-Cox
        if (feature_data <= 0).any():
            print(f"Skipping Box-Cox for {column} (contains non-positive values)")
            boxcox_transformed = None
            boxcox_lambda = None
        else:
            try:
                boxcox_transformed, boxcox_lambda = boxcox(feature_data)
            except:
                boxcox_transformed = None
                boxcox_lambda = None
        
        # Yeo-Johnson can handle negative values
        try:
            yeojohnson_transformed, yeojohnson_lambda = yeojohnson(feature_data)
        except:
            yeojohnson_transformed = None
            yeojohnson_lambda = None
        
        # Log transformation (add 1 to handle zeros)
        log_transformed = np.log1p(feature_data - feature_data.min() + 1)
        
        # Square root transformation
        sqrt_transformed = np.sqrt(feature_data - feature_data.min() + 1)
        
        # Square transformation
        square_transformed = np.square(feature_data)
        
        results[column] = {
            'original': feature_data,
            'boxcox': boxcox_transformed,
            'boxcox_lambda': boxcox_lambda,
            'yeojohnson': yeojohnson_transformed,
            'yeojohnson_lambda': yeojohnson_lambda,
            'log1p': log_transformed,
            'sqrt': sqrt_transformed,
            'square': square_transformed
        }
    
    return results

# Apply power transformations
power_results = apply_power_transformations(X_train)

# Visualize power transformations for most skewed feature
most_skewed_feature = X_train.skew().abs().idxmax()
print(f"Most skewed feature: {most_skewed_feature} (skewness: {X_train[most_skewed_feature].skew():.3f})")

plt.figure(figsize=(20, 12))
transformations = ['original', 'boxcox', 'yeojohnson', 'log1p', 'sqrt', 'square']

for i, transform in enumerate(transformations):
    plt.subplot(2, 3, i + 1)
    
    data = power_results[most_skewed_feature][transform]
    if data is not None:
        plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
        skewness = stats.skew(data)
        plt.title(f'{transform.title()}\nSkewness: {skewness:.3f}')
        
        if transform in ['boxcox', 'yeojohnson']:
            lambda_val = power_results[most_skewed_feature][f'{transform}_lambda']
            if lambda_val is not None:
                plt.title(f'{transform.title()}\nλ={lambda_val:.3f}, Skew: {skewness:.3f}')
    else:
        plt.text(0.5, 0.5, 'Not Applicable', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{transform.title()}\nNot Applicable')
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# ===============================================================================
# 3. SKLEARN POWER TRANSFORMERS
# ===============================================================================

print("\n" + "="*60)
print("3. SKLEARN POWER TRANSFORMERS")
print("="*60)

# PowerTransformer with different methods
power_transformers = {
    'BoxCox': PowerTransformer(method='box-cox', standardize=True),
    'YeoJohnson': PowerTransformer(method='yeo-johnson', standardize=True),
    'BoxCox_NoStandardize': PowerTransformer(method='box-cox', standardize=False),
    'YeoJohnson_NoStandardize': PowerTransformer(method='yeo-johnson', standardize=False)
}

power_transformer_results = {}

for transformer_name, transformer in power_transformers.items():
    print(f"Applying {transformer_name}...")
    
    try:
        # Ensure all values are positive for Box-Cox
        if 'BoxCox' in transformer_name:
            X_train_positive = X_train - X_train.min() + 1
            X_test_positive = X_test - X_test.min() + 1
            
            X_train_transformed = transformer.fit_transform(X_train_positive)
            X_test_transformed = transformer.transform(X_test_positive)
        else:
            X_train_transformed = transformer.fit_transform(X_train)
            X_test_transformed = transformer.transform(X_test)
        
        # Convert back to DataFrame
        X_train_transformed_df = pd.DataFrame(
            X_train_transformed, columns=X.columns, index=X_train.index
        )
        X_test_transformed_df = pd.DataFrame(
            X_test_transformed, columns=X.columns, index=X_test.index
        )
        
        power_transformer_results[transformer_name] = {
            'transformer': transformer,
            'X_train': X_train_transformed_df,
            'X_test': X_test_transformed_df,
            'lambdas': getattr(transformer, 'lambdas_', None)
        }
        
    except Exception as e:
        print(f"Error with {transformer_name}: {e}")

# Visualize power transformer results
plt.figure(figsize=(20, 10))

for i, (transformer_name, results) in enumerate(power_transformer_results.items()):
    plt.subplot(2, 2, i + 1)
    
    # Plot before and after for the most skewed feature
    original_data = X_train[most_skewed_feature]
    transformed_data = results['X_train'][most_skewed_feature]
    
    plt.hist(original_data, alpha=0.5, bins=30, label='Original', density=True)
    plt.hist(transformed_data, alpha=0.5, bins=30, label='Transformed', density=True)
    
    original_skew = stats.skew(original_data)
    transformed_skew = stats.skew(transformed_data)
    
    plt.title(f'{transformer_name}\n{most_skewed_feature}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Add skewness information
    plt.text(0.05, 0.95, f'Original skew: {original_skew:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.05, 0.85, f'Transformed skew: {transformed_skew:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    if results['lambdas'] is not None:
        lambda_val = results['lambdas'][X.columns.get_loc(most_skewed_feature)]
        plt.text(0.05, 0.75, f'λ: {lambda_val:.3f}', 
                 transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()

# ===============================================================================
# 4. QUANTILE TRANSFORMATIONS
# ===============================================================================

print("\n" + "="*60)
print("4. QUANTILE TRANSFORMATIONS")
print("="*60)

# Different quantile transformation methods
quantile_transformers = {
    'Uniform': QuantileTransformer(output_distribution='uniform', random_state=42),
    'Normal': QuantileTransformer(output_distribution='normal', random_state=42),
    'Uniform_Subsample': QuantileTransformer(
        output_distribution='uniform', 
        n_quantiles=500, 
        subsample=1000, 
        random_state=42
    ),
    'Normal_Robust': QuantileTransformer(
        output_distribution='normal', 
        n_quantiles=100, 
        random_state=42
    )
}

quantile_results = {}

for transformer_name, transformer in quantile_transformers.items():
    print(f"Applying Quantile Transformer - {transformer_name}...")
    
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    
    # Convert back to DataFrame
    X_train_transformed_df = pd.DataFrame(
        X_train_transformed, columns=X.columns, index=X_train.index
    )
    X_test_transformed_df = pd.DataFrame(
        X_test_transformed, columns=X.columns, index=X_test.index
    )
    
    quantile_results[transformer_name] = {
        'transformer': transformer,
        'X_train': X_train_transformed_df,
        'X_test': X_test_transformed_df
    }

# Visualize quantile transformations
plt.figure(figsize=(20, 10))

# Test on bimodal feature which should show clear differences
feature_to_test = 'bimodal_feature'

for i, (transformer_name, results) in enumerate(quantile_results.items()):
    plt.subplot(2, 2, i + 1)
    
    original_data = X_train[feature_to_test]
    transformed_data = results['X_train'][feature_to_test]
    
    plt.hist(original_data, alpha=0.5, bins=30, label='Original', density=True)
    plt.hist(transformed_data, alpha=0.5, bins=30, label='Transformed', density=True)
    
    plt.title(f'Quantile - {transformer_name}\n{feature_to_test}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Add distribution information
    original_skew = stats.skew(original_data)
    transformed_skew = stats.skew(transformed_data)
    plt.text(0.05, 0.95, f'Original skew: {original_skew:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.05, 0.85, f'Transformed skew: {transformed_skew:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()

# ===============================================================================
# 5. CUSTOM TRANSFORMATION METHODS
# ===============================================================================

print("\n" + "="*60)
print("5. CUSTOM TRANSFORMATION METHODS")
print("="*60)

class CustomTransformers:
    """Custom transformation methods"""
    
    @staticmethod
    def rank_transform(data):
        """Rank-based transformation"""
        return data.rank(pct=True)
    
    @staticmethod
    def winsorize_transform(data, limits=(0.05, 0.05)):
        """Winsorization to handle outliers"""
        from scipy.stats.mstats import winsorize
        return pd.Series(winsorize(data, limits=limits), index=data.index)
    
    @staticmethod
    def sigmoid_transform(data):
        """Sigmoid transformation"""
        # Standardize first
        standardized = (data - data.mean()) / data.std()
        return 1 / (1 + np.exp(-standardized))
    
    @staticmethod
    def inverse_transform(data, constant=1):
        """Inverse transformation"""
        return constant / (data + constant)
    
    @staticmethod
    def arcsin_sqrt_transform(data):
        """Arcsine square root transformation for proportions"""
        # Normalize to [0,1] first
        normalized = (data - data.min()) / (data.max() - data.min())
        return np.arcsin(np.sqrt(normalized))

# Apply custom transformations
custom_transformations = {
    'Rank': CustomTransformers.rank_transform,
    'Winsorize': CustomTransformers.winsorize_transform,
    'Sigmoid': CustomTransformers.sigmoid_transform,
    'Inverse': CustomTransformers.inverse_transform,
    'ArcsinSqrt': CustomTransformers.arcsin_sqrt_transform
}

custom_results = {}
for transform_name, transform_func in custom_transformations.items():
    print(f"Applying {transform_name} transformation...")
    
    X_train_transformed = X_train.apply(transform_func)
    
    custom_results[transform_name] = {
        'X_train': X_train_transformed
    }

# Visualize custom transformations
plt.figure(figsize=(20, 12))

# Use outlier feature to show effect of different transformations
feature_to_test = 'outlier_feature'

for i, (transform_name, results) in enumerate(custom_results.items()):
    plt.subplot(2, 3, i + 1)
    
    original_data = X_train[feature_to_test]
    transformed_data = results['X_train'][feature_to_test]
    
    plt.hist(original_data, alpha=0.5, bins=30, label='Original', density=True)
    plt.hist(transformed_data, alpha=0.5, bins=30, label='Transformed', density=True)
    
    plt.title(f'{transform_name}\n{feature_to_test}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()

# ===============================================================================
# 6. TRANSFORMATION EVALUATION
# ===============================================================================

print("\n" + "="*60)
print("6. TRANSFORMATION EVALUATION")
print("="*60)

def evaluate_transformation(original_data, transformed_data, transformation_name):
    """Evaluate transformation effectiveness"""
    
    # Normality tests
    from scipy.stats import shapiro, jarque_bera
    
    # Original data statistics
    orig_shapiro_stat, orig_shapiro_p = shapiro(original_data.sample(min(5000, len(original_data))))
    orig_jb_stat, orig_jb_p = jarque_bera(original_data)
    orig_skewness = stats.skew(original_data)
    orig_kurtosis = stats.kurtosis(original_data)
    
    # Transformed data statistics
    trans_shapiro_stat, trans_shapiro_p = shapiro(transformed_data.sample(min(5000, len(transformed_data))))
    trans_jb_stat, trans_jb_p = jarque_bera(transformed_data)
    trans_skewness = stats.skew(transformed_data)
    trans_kurtosis = stats.kurtosis(transformed_data)
    
    return {
        'transformation': transformation_name,
        'orig_shapiro_p': orig_shapiro_p,
        'trans_shapiro_p': trans_shapiro_p,
        'orig_jb_p': orig_jb_p,
        'trans_jb_p': trans_jb_p,
        'orig_skewness': orig_skewness,
        'trans_skewness': trans_skewness,
        'orig_kurtosis': orig_kurtosis,
        'trans_kurtosis': trans_kurtosis,
        'skewness_improvement': abs(orig_skewness) - abs(trans_skewness),
        'kurtosis_improvement': abs(orig_kurtosis) - abs(trans_kurtosis)
    }

# Evaluate all transformations for the most skewed feature
evaluation_results = []

# Original data
evaluation_results.append({
    'transformation': 'Original',
    'orig_shapiro_p': None,
    'trans_shapiro_p': None,
    'orig_jb_p': None,
    'trans_jb_p': None,
    'orig_skewness': X_train[most_skewed_feature].skew(),
    'trans_skewness': X_train[most_skewed_feature].skew(),
    'orig_kurtosis': X_train[most_skewed_feature].kurtosis(),
    'trans_kurtosis': X_train[most_skewed_feature].kurtosis(),
    'skewness_improvement': 0,
    'kurtosis_improvement': 0
})

# Scaling methods
for scaler_name, results in scaled_results.items():
    if scaler_name in ['StandardScaler', 'RobustScaler', 'MinMaxScaler']:  # Skip normalizers
        eval_result = evaluate_transformation(
            X_train[most_skewed_feature],
            results['X_train'][most_skewed_feature],
            scaler_name
        )
        evaluation_results.append(eval_result)

# Power transformers
for transformer_name, results in power_transformer_results.items():
    eval_result = evaluate_transformation(
        X_train[most_skewed_feature],
        results['X_train'][most_skewed_feature],
        transformer_name
    )
    evaluation_results.append(eval_result)

# Quantile transformers
for transformer_name, results in quantile_results.items():
    eval_result = evaluate_transformation(
        X_train[most_skewed_feature],
        results['X_train'][most_skewed_feature],
        f'Quantile_{transformer_name}'
    )
    evaluation_results.append(eval_result)

# Create evaluation DataFrame
eval_df = pd.DataFrame(evaluation_results)
eval_df = eval_df.sort_values('skewness_improvement', ascending=False)

print(f"Transformation Evaluation for {most_skewed_feature}:")
print(f"Original Skewness: {X_train[most_skewed_feature].skew():.3f}")
print("\nTop 5 Transformations by Skewness Improvement:")
print(eval_df[['transformation', 'trans_skewness', 'skewness_improvement']].head().round(4))

# ===============================================================================
# 7. MODEL PERFORMANCE COMPARISON
# ===============================================================================

print("\n" + "="*60)
print("7. MODEL PERFORMANCE COMPARISON")
print("="*60)

def evaluate_model_performance(X_train_data, X_test_data, y_train, y_test, transformation_name):
    """Evaluate model performance with different transformations"""
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_data, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_data)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # R² score
    r2 = model.score(X_test_data, y_test)
    
    return {
        'transformation': transformation_name,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

# Compare model performance with different transformations
performance_results = []

# Original data
perf_result = evaluate_model_performance(X_train, X_test, y_train, y_test, 'Original')
performance_results.append(perf_result)

# Scaling methods
for scaler_name, results in scaled_results.items():
    perf_result = evaluate_model_performance(
        results['X_train'], results['X_test'], y_train, y_test, scaler_name
    )
    performance_results.append(perf_result)

# Power transformers
for transformer_name, results in power_transformer_results.items():
    perf_result = evaluate_model_performance(
        results['X_train'], results['X_test'], y_train, y_test, transformer_name
    )
    performance_results.append(perf_result)

# Quantile transformers
for transformer_name, results in quantile_results.items():
    perf_result = evaluate_model_performance(
        results['X_train'], results['X_test'], y_train, y_test, f'Quantile_{transformer_name}'
    )
    performance_results.append(perf_result)

# Create performance DataFrame
perf_df = pd.DataFrame(performance_results)
perf_df = perf_df.sort_values('r2', ascending=False)

print("Model Performance Comparison (Random Forest):")
print(perf_df.round(4))

# Visualize performance comparison
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.bar(range(len(perf_df)), perf_df['r2'])
plt.xlabel('Transformation Method')
plt.ylabel('R² Score')
plt.title('Model Performance: R² Score')
plt.xticks(range(len(perf_df)), perf_df['transformation'], rotation=45)

plt.subplot(2, 2, 2)
plt.bar(range(len(perf_df)), perf_df['rmse'])
plt.xlabel('Transformation Method')
plt.ylabel('RMSE')
plt.title('Model Performance: RMSE')
plt.xticks(range(len(perf_df)), perf_df['transformation'], rotation=45)

plt.subplot(2, 2, 3)
# Skewness improvement vs R² score
eval_df_matched = eval_df.set_index('transformation')
perf_df_matched = perf_df.set_index('transformation')
common_transformations = eval_df_matched.index.intersection(perf_df_matched.index)

if len(common_transformations) > 0:
    skew_improvements = eval_df_matched.loc[common_transformations, 'skewness_improvement']
    r2_scores = perf_df_matched.loc[common_transformations, 'r2']
    
    plt.scatter(skew_improvements, r2_scores)
    for i, trans in enumerate(common_transformations):
        plt.annotate(trans, (skew_improvements[i], r2_scores[i]), fontsize=8)
    
    plt.xlabel('Skewness Improvement')
    plt.ylabel('R² Score')
    plt.title('Skewness Improvement vs Model Performance')

plt.subplot(2, 2, 4)
# Performance ranking
perf_ranking = perf_df.reset_index(drop=True)
plt.plot(range(len(perf_ranking)), perf_ranking['r2'], 'o-')
plt.xlabel('Rank (by R²)')
plt.ylabel('R² Score')
plt.title('Transformation Performance Ranking')

plt.tight_layout()
plt.show()

# ===============================================================================
# 8. TRANSFORMATION RECOMMENDATIONS
# ===============================================================================

print("\n" + "="*60)
print("8. TRANSFORMATION RECOMMENDATIONS")
print("="*60)

# Best transformation by different criteria
best_by_skewness = eval_df.iloc[0] if len(eval_df) > 1 else None
best_by_performance = perf_df.iloc[0]

print("TRANSFORMATION ANALYSIS SUMMARY:")
print("=" * 40)

if best_by_skewness is not None:
    print(f"Best for Skewness Reduction: {best_by_skewness['transformation']}")
    print(f"  Skewness improvement: {best_by_skewness['skewness_improvement']:.4f}")
    print(f"  Final skewness: {best_by_skewness['trans_skewness']:.4f}")

print(f"\nBest for Model Performance: {best_by_performance['transformation']}")
print(f"  R² Score: {best_by_performance['r2']:.4f}")
print(f"  RMSE: {best_by_performance['rmse']:.4f}")

# Feature-specific recommendations
print(f"\nFEATURE-SPECIFIC RECOMMENDATIONS:")
print("=" * 35)

for feature in X.columns:
    skewness = X[feature].skew()
    
    if abs(skewness) < 0.5:
        recommendation = "StandardScaler or MinMaxScaler"
    elif skewness > 2:
        recommendation = "Log1p or Box-Cox transformation"
    elif skewness < -2:
        recommendation = "Square or Yeo-Johnson transformation"
    elif abs(skewness) > 1:
        recommendation = "Yeo-Johnson or Quantile transformation"
    else:
        recommendation = "RobustScaler or light power transformation"
    
    print(f"{feature}: Skewness={skewness:.3f} → {recommendation}")

print(f"\nGENERAL GUIDELINES:")
print("=" * 20)
print("✓ Use StandardScaler for normally distributed features")
print("✓ Use RobustScaler for features with outliers")
print("✓ Use PowerTransformer for skewed distributions")
print("✓ Use QuantileTransformer for complex distributions")
print("✓ Always validate transformations with cross-validation")
print("✓ Consider the interpretability trade-off")

# Export transformation results
transformation_summary = pd.DataFrame({
    'Feature': X.columns,
    'Original_Skewness': [X[col].skew() for col in X.columns],
    'Original_Kurtosis': [X[col].kurtosis() for col in X.columns],
    'Has_Outliers': [(X[col] > X[col].quantile(0.75) + 1.5 * (X[col].quantile(0.75) - X[col].quantile(0.25))).any() for col in X.columns],
    'Recommended_Transformation': ['TBD' for _ in X.columns]  # To be filled based on analysis
})

print(f"\nTransformation analysis complete!")
print(f"Summary statistics and recommendations generated!")
print(f"Best performing transformation: {best_by_performance['transformation']}")
