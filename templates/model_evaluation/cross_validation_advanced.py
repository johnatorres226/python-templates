"""
===============================================================================
ADVANCED CROSS-VALIDATION TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Advanced cross-validation strategies for robust model evaluation

This template covers:
- Stratified cross-validation
- Time series cross-validation
- Nested cross-validation
- Group-based cross-validation
- Leave-one-out and leave-p-out CV
- Custom cross-validation strategies

Prerequisites:
- pandas, numpy, scikit-learn, matplotlib, seaborn
- Dataset loaded as 'df' with target variable
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    TimeSeriesSplit, GroupKFold, LeaveOneOut, LeavePOut,
    GridSearchCV, RandomizedSearchCV, validation_curve
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===============================================================================
# LOAD AND PREPARE DATA
# ===============================================================================

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Create comprehensive synthetic dataset for demonstration
n_samples = 1000
n_features = 10

# Generate features
X = np.random.randn(n_samples, n_features)

# Create different types of target variables
# Classification target with class imbalance
y_class = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])

# Regression target with some correlation to features
y_reg = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5

# Create time series index
dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

# Create groups for group-based CV
groups = np.random.randint(0, 20, n_samples)  # 20 different groups

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['target_class'] = y_class
df['target_reg'] = y_reg
df['date'] = dates
df['group'] = groups

print("Dataset Shape:", df.shape)
print("Class Distribution:")
print(df['target_class'].value_counts().sort_index())
print("\nRegression Target Statistics:")
print(df['target_reg'].describe())

# ===============================================================================
# 1. STRATIFIED CROSS-VALIDATION
# ===============================================================================

print("\n" + "="*60)
print("1. STRATIFIED CROSS-VALIDATION")
print("="*60)

# Prepare data for classification
X_class = df.drop(['target_class', 'target_reg', 'date', 'group'], axis=1)
y_class = df['target_class']

# Standard K-Fold vs Stratified K-Fold comparison
k_folds = 5

# Standard K-Fold
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Regular cross-validation
cv_scores_regular = cross_val_score(rf_classifier, X_class, y_class, cv=kfold, scoring='accuracy')

# Stratified cross-validation
stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
cv_scores_stratified = cross_val_score(rf_classifier, X_class, y_class, cv=stratified_kfold, scoring='accuracy')

print("Cross-Validation Results:")
print(f"Regular K-Fold CV: {cv_scores_regular.mean():.4f} (+/- {cv_scores_regular.std() * 2:.4f})")
print(f"Stratified K-Fold CV: {cv_scores_stratified.mean():.4f} (+/- {cv_scores_stratified.std() * 2:.4f})")

# Detailed stratified cross-validation with multiple metrics
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(
    rf_classifier, X_class, y_class, 
    cv=stratified_kfold, 
    scoring=scoring_metrics,
    return_train_score=True
)

# Display results
print("\nDetailed Stratified Cross-Validation Results:")
for metric in scoring_metrics:
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']
    print(f"{metric.capitalize()}:")
    print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")
    print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")

# Visualize cross-validation results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Stratified Cross-Validation Results', fontsize=16)

for i, metric in enumerate(scoring_metrics):
    ax = axes[i//2, i%2]
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']
    
    x_pos = np.arange(len(test_scores))
    ax.bar(x_pos - 0.2, train_scores, 0.4, label='Train', alpha=0.7)
    ax.bar(x_pos + 0.2, test_scores, 0.4, label='Test', alpha=0.7)
    
    ax.set_title(f'{metric.capitalize()} Scores')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(test_scores))])

plt.tight_layout()
plt.show()

# Check class distribution in each fold
print("\nClass Distribution in Each Fold:")
for fold_idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X_class, y_class)):
    test_distribution = y_class.iloc[test_idx].value_counts().sort_index()
    print(f"Fold {fold_idx + 1}: {dict(test_distribution)}")

# ===============================================================================
# 2. TIME SERIES CROSS-VALIDATION
# ===============================================================================

print("\n" + "="*60)
print("2. TIME SERIES CROSS-VALIDATION")
print("="*60)

# Prepare time series data
X_ts = df.drop(['target_reg', 'target_class', 'date', 'group'], axis=1)
y_ts = df['target_reg']

# Time Series Split
n_splits_ts = 5
tscv = TimeSeriesSplit(n_splits=n_splits_ts)

# Create regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform time series cross-validation
ts_scores = cross_val_score(rf_regressor, X_ts, y_ts, cv=tscv, scoring='r2')

print("Time Series Cross-Validation Results:")
print(f"R² Scores: {ts_scores}")
print(f"Mean R²: {ts_scores.mean():.4f} (+/- {ts_scores.std() * 2:.4f})")

# Detailed time series CV with multiple metrics
ts_scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
ts_results = cross_validate(
    rf_regressor, X_ts, y_ts,
    cv=tscv,
    scoring=ts_scoring,
    return_train_score=True
)

print("\nDetailed Time Series Cross-Validation:")
for metric in ts_scoring:
    test_scores = ts_results[f'test_{metric}']
    train_scores = ts_results[f'train_{metric}']
    print(f"{metric.replace('neg_', '').replace('_', ' ').title()}:")
    print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std() * 2:.4f})")
    print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std() * 2:.4f})")

# Visualize time series splits
fig, ax = plt.subplots(figsize=(15, 8))

for i, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    # Plot training data
    ax.plot(train_idx, [i] * len(train_idx), 'b-', alpha=0.6, label='Training' if i == 0 else "")
    # Plot test data
    ax.plot(test_idx, [i] * len(test_idx), 'r-', alpha=0.8, label='Testing' if i == 0 else "")

ax.set_xlabel('Sample Index')
ax.set_ylabel('CV Fold')
ax.set_title('Time Series Cross-Validation Splits')
ax.legend()
plt.tight_layout()
plt.show()

# Walk-forward validation example
print("\nWalk-Forward Validation Results:")
walk_forward_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    X_train, X_test = X_ts.iloc[train_idx], X_ts.iloc[test_idx]
    y_train, y_test = y_ts.iloc[train_idx], y_ts.iloc[test_idx]
    
    # Fit model
    rf_regressor.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_regressor.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    walk_forward_scores.append({'fold': fold_idx + 1, 'mse': mse, 'r2': r2})
    print(f"Fold {fold_idx + 1}: MSE = {mse:.4f}, R² = {r2:.4f}")

# ===============================================================================
# 3. NESTED CROSS-VALIDATION
# ===============================================================================

print("\n" + "="*60)
print("3. NESTED CROSS-VALIDATION")
print("="*60)

# Nested CV for hyperparameter tuning and unbiased performance estimation
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Outer CV for performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Nested cross-validation
nested_scores = []
best_params_list = []

print("Performing Nested Cross-Validation...")
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_class, y_class)):
    X_train_outer, X_test_outer = X_class.iloc[train_idx], X_class.iloc[test_idx]
    y_train_outer, y_test_outer = y_class.iloc[train_idx], y_class.iloc[test_idx]
    
    # Inner loop: hyperparameter tuning
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Store best parameters
    best_params_list.append(grid_search.best_params_)
    
    # Outer loop: performance estimation
    best_model = grid_search.best_estimator_
    y_pred_outer = best_model.predict(X_test_outer)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test_outer, y_pred_outer)
    precision = precision_score(y_test_outer, y_pred_outer, average='macro')
    recall = recall_score(y_test_outer, y_pred_outer, average='macro')
    f1 = f1_score(y_test_outer, y_pred_outer, average='macro')
    
    nested_scores.append({
        'fold': fold_idx + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    print(f"Outer Fold {fold_idx + 1}:")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Test accuracy: {accuracy:.4f}")

# Calculate final nested CV scores
nested_df = pd.DataFrame(nested_scores)
print("\nNested Cross-Validation Results:")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    scores = nested_df[metric]
    print(f"{metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Analyze hyperparameter stability
print("\nHyperparameter Stability Across Folds:")
param_df = pd.DataFrame(best_params_list)
for param in param_df.columns:
    print(f"{param}: {param_df[param].value_counts().to_dict()}")

# ===============================================================================
# 4. GROUP-BASED CROSS-VALIDATION
# ===============================================================================

print("\n" + "="*60)
print("4. GROUP-BASED CROSS-VALIDATION")
print("="*60)

# Group K-Fold for data with natural groupings
group_kfold = GroupKFold(n_splits=5)

# Perform group-based cross-validation
group_scores = cross_val_score(
    rf_classifier, X_class, y_class, 
    cv=group_kfold, 
    groups=df['group'],
    scoring='accuracy'
)

print("Group-Based Cross-Validation Results:")
print(f"Accuracy scores: {group_scores}")
print(f"Mean accuracy: {group_scores.mean():.4f} (+/- {group_scores.std() * 2:.4f})")

# Visualize group distribution across folds
print("\nGroup Distribution Analysis:")
for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X_class, y_class, groups=df['group'])):
    train_groups = set(df.iloc[train_idx]['group'])
    test_groups = set(df.iloc[test_idx]['group'])
    overlap = train_groups.intersection(test_groups)
    
    print(f"Fold {fold_idx + 1}:")
    print(f"  Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
    print(f"  Overlapping groups: {len(overlap)} (should be 0)")

# ===============================================================================
# 5. LEAVE-ONE-OUT AND LEAVE-P-OUT CV
# ===============================================================================

print("\n" + "="*60)
print("5. LEAVE-ONE-OUT AND LEAVE-P-OUT CV")
print("="*60)

# Leave-One-Out CV (computationally expensive, use small subset)
X_small = X_class.head(100)
y_small = y_class.head(100)

loo = LeaveOneOut()
loo_scores = cross_val_score(rf_classifier, X_small, y_small, cv=loo, scoring='accuracy')

print(f"Leave-One-Out CV Results (n={len(X_small)}):")
print(f"Mean accuracy: {loo_scores.mean():.4f} (+/- {loo_scores.std() * 2:.4f})")
print(f"Number of iterations: {len(loo_scores)}")

# Leave-P-Out CV
p_out = 5
lpo = LeavePOut(p=p_out)

# Note: LeavePOut can be computationally expensive
# Calculate number of combinations
from math import comb
n_combinations = comb(len(X_small), p_out)
print(f"\nLeave-{p_out}-Out CV would require {n_combinations} iterations")

if n_combinations <= 1000:  # Only run if reasonable number of iterations
    lpo_scores = cross_val_score(rf_classifier, X_small, y_small, cv=lpo, scoring='accuracy')
    print(f"Leave-{p_out}-Out CV Results:")
    print(f"Mean accuracy: {lpo_scores.mean():.4f} (+/- {lpo_scores.std() * 2:.4f})")
else:
    print(f"Skipping Leave-{p_out}-Out CV due to computational complexity")

# ===============================================================================
# 6. CUSTOM CROSS-VALIDATION STRATEGIES
# ===============================================================================

print("\n" + "="*60)
print("6. CUSTOM CROSS-VALIDATION STRATEGIES")
print("="*60)

# Custom CV: Temporal blocks for time series
class TemporalBlockCV:
    def __init__(self, n_splits=5, block_size=None):
        self.n_splits = n_splits
        self.block_size = block_size
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if self.block_size is None:
            self.block_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Training: all data before test block
            train_end = (i + 1) * self.block_size
            # Test: next block
            test_start = train_end
            test_end = test_start + self.block_size
            
            if test_end > n_samples:
                break
                
            train_indices = list(range(train_end))
            test_indices = list(range(test_start, min(test_end, n_samples)))
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Apply custom temporal block CV
temporal_cv = TemporalBlockCV(n_splits=5)
temporal_scores = cross_val_score(rf_regressor, X_ts, y_ts, cv=temporal_cv, scoring='r2')

print("Custom Temporal Block CV Results:")
print(f"R² scores: {temporal_scores}")
print(f"Mean R²: {temporal_scores.mean():.4f} (+/- {temporal_scores.std() * 2:.4f})")

# Purged cross-validation for financial time series
class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, embargo_td=0):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test set
            test_start = i * test_size
            test_end = test_start + test_size
            
            # Training set (before test, with embargo)
            train_end = test_start - self.embargo_td
            
            if train_end <= 0:
                continue
                
            train_indices = list(range(train_end))
            test_indices = list(range(test_start, min(test_end, n_samples)))
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Apply purged time series CV
purged_cv = PurgedTimeSeriesSplit(n_splits=5, embargo_td=10)
purged_scores = cross_val_score(rf_regressor, X_ts, y_ts, cv=purged_cv, scoring='r2')

print("\nPurged Time Series CV Results:")
print(f"R² scores: {purged_scores}")
print(f"Mean R²: {purged_scores.mean():.4f} (+/- {purged_scores.std() * 2:.4f})")

# ===============================================================================
# 7. CROSS-VALIDATION DIAGNOSTICS AND VALIDATION CURVES
# ===============================================================================

print("\n" + "="*60)
print("7. CROSS-VALIDATION DIAGNOSTICS")
print("="*60)

# Validation curve for hyperparameter analysis
param_range = [10, 50, 100, 200, 500]
train_scores, test_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_class, y_class,
    param_name='n_estimators',
    param_range=param_range,
    cv=stratified_kfold,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(12, 8))
plt.plot(param_range, train_mean, 'o-', color='blue', label='Training accuracy')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(param_range, test_mean, 'o-', color='red', label='Cross-validation accuracy')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Validation Curve: Random Forest n_estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Learning curve analysis
from sklearn.model_selection import learning_curve

train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_class, y_class,
    cv=stratified_kfold,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

# Calculate means and standard deviations for learning curve
train_mean_lc = np.mean(train_scores_lc, axis=1)
train_std_lc = np.std(train_scores_lc, axis=1)
test_mean_lc = np.mean(test_scores_lc, axis=1)
test_std_lc = np.std(test_scores_lc, axis=1)

# Plot learning curve
plt.figure(figsize=(12, 8))
plt.plot(train_sizes, train_mean_lc, 'o-', color='blue', label='Training accuracy')
plt.fill_between(train_sizes, train_mean_lc - train_std_lc, train_mean_lc + train_std_lc, alpha=0.1, color='blue')
plt.plot(train_sizes, test_mean_lc, 'o-', color='red', label='Cross-validation accuracy')
plt.fill_between(train_sizes, test_mean_lc - test_std_lc, test_mean_lc + test_std_lc, alpha=0.1, color='red')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Random Forest Classifier')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ===============================================================================
# 8. CROSS-VALIDATION SUMMARY AND BEST PRACTICES
# ===============================================================================

print("\n" + "="*60)
print("8. CROSS-VALIDATION SUMMARY")
print("="*60)

# Summary of all CV strategies used
cv_summary = {
    'Strategy': [
        'Standard K-Fold',
        'Stratified K-Fold',
        'Time Series CV',
        'Nested CV',
        'Group K-Fold',
        'Leave-One-Out',
        'Custom Temporal',
        'Purged Time Series'
    ],
    'Use Case': [
        'General regression problems',
        'Classification with imbalanced classes',
        'Time series and temporal data',
        'Hyperparameter tuning + evaluation',
        'Data with natural groupings',
        'Small datasets, maximum data usage',
        'Financial/temporal data',
        'Overlapping time series features'
    ],
    'Pros': [
        'Simple, widely applicable',
        'Maintains class distribution',
        'Respects temporal order',
        'Unbiased performance estimate',
        'Prevents data leakage',
        'Uses all data for training',
        'Custom validation logic',
        'Prevents look-ahead bias'
    ],
    'Cons': [
        'May not preserve distributions',
        'Slightly more complex',
        'Smaller training sets',
        'Computationally expensive',
        'Requires group information',
        'Very computationally expensive',
        'Requires domain knowledge',
        'Complex implementation'
    ]
}

cv_summary_df = pd.DataFrame(cv_summary)
print("Cross-Validation Strategy Comparison:")
print(cv_summary_df.to_string(index=False))

# Best practices summary
print("\n" + "="*50)
print("CROSS-VALIDATION BEST PRACTICES")
print("="*50)

best_practices = [
    "1. Choose CV strategy based on data characteristics",
    "2. Use stratified CV for classification problems",
    "3. Use time series CV for temporal data",
    "4. Apply nested CV for hyperparameter tuning",
    "5. Consider group CV when data has natural clusters",
    "6. Monitor both training and validation scores",
    "7. Check for overfitting using learning curves",
    "8. Ensure reproducibility with random seeds",
    "9. Use multiple metrics for comprehensive evaluation",
    "10. Consider computational costs vs. accuracy gains"
]

for practice in best_practices:
    print(practice)

print(f"\nAdvanced cross-validation analysis complete!")
print(f"Use these strategies based on your specific data characteristics and problem type.")
