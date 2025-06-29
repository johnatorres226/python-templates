"""
===============================================================================
FEATURE SELECTION TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive template for selecting the most relevant features

This template covers:
- Statistical feature selection methods
- Model-based feature selection
- Recursive feature elimination
- Correlation-based selection
- Variance-based selection
- Feature importance ranking

Prerequisites:
- pandas, numpy, matplotlib, seaborn, scikit-learn
- Dataset with features and target variable
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# LOAD AND PREPARE DATA
# ===============================================================================

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Sample dataset creation for demonstration
np.random.seed(42)
n_samples = 1000
n_features = 20

# Create synthetic dataset with mix of relevant and irrelevant features
X = np.random.randn(n_samples, n_features)
# Make some features correlated with target
y_regression = (2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + 
                np.random.randn(n_samples) * 0.1)
y_classification = (y_regression > np.median(y_regression)).astype(int)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target_regression'] = y_regression
df['target_classification'] = y_classification

print("Dataset Shape:", df.shape)
print("Features:", len(feature_names))
print("Regression Target - Range:", f"{y_regression.min():.2f} to {y_regression.max():.2f}")
print("Classification Target - Distribution:", pd.Series(y_classification).value_counts().to_dict())

# ===============================================================================
# 1. VARIANCE-BASED FEATURE SELECTION
# ===============================================================================

print("\n" + "="*60)
print("1. VARIANCE-BASED FEATURE SELECTION")
print("="*60)

# Remove low variance features
variance_selector = VarianceThreshold(threshold=0.1)
X_features = df[feature_names]

# Fit and transform
X_variance_selected = variance_selector.fit_transform(X_features)
selected_features_variance = np.array(feature_names)[variance_selector.get_support()]

print(f"Original features: {len(feature_names)}")
print(f"Features after variance selection: {len(selected_features_variance)}")
print(f"Removed {len(feature_names) - len(selected_features_variance)} low-variance features")

# Show feature variances
feature_variances = pd.DataFrame({
    'feature': feature_names,
    'variance': X_features.var()
}).sort_values('variance', ascending=False)

print("\nFeature Variances:")
print(feature_variances.head(10))

# Plot variance distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(feature_variances['variance'], bins=20, edgecolor='black')
plt.xlabel('Variance')
plt.ylabel('Frequency')
plt.title('Distribution of Feature Variances')

plt.subplot(1, 2, 2)
plt.barh(range(len(feature_variances)), feature_variances['variance'])
plt.yticks(range(len(feature_variances)), feature_variances['feature'])
plt.xlabel('Variance')
plt.title('Feature Variances')
plt.tight_layout()
plt.show()

# ===============================================================================
# 2. CORRELATION-BASED FEATURE SELECTION
# ===============================================================================

print("\n" + "="*60)
print("2. CORRELATION-BASED FEATURE SELECTION")
print("="*60)

# Calculate correlation matrix
correlation_matrix = X_features.corr()

# Find highly correlated feature pairs
def find_correlated_features(corr_matrix, threshold=0.8):
    """Find pairs of features with correlation above threshold"""
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    return pd.DataFrame(corr_pairs)

highly_correlated = find_correlated_features(correlation_matrix, threshold=0.7)
print(f"Highly correlated feature pairs (>0.7): {len(highly_correlated)}")
if len(highly_correlated) > 0:
    print(highly_correlated.head())

# Remove highly correlated features
def remove_highly_correlated(df, threshold=0.8):
    """Remove one feature from each highly correlated pair"""
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

X_features_uncorr, dropped_corr = remove_highly_correlated(X_features, threshold=0.8)
print(f"Features removed due to high correlation: {len(dropped_corr)}")
print(f"Remaining features: {X_features_uncorr.shape[1]}")

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# ===============================================================================
# 3. UNIVARIATE STATISTICAL SELECTION
# ===============================================================================

print("\n" + "="*60)
print("3. UNIVARIATE STATISTICAL SELECTION")
print("="*60)

# For regression target
print("REGRESSION TARGET:")
print("-" * 30)

# F-test for regression
f_selector_reg = SelectKBest(score_func=f_regression, k=10)
X_f_selected_reg = f_selector_reg.fit_transform(X_features, df['target_regression'])
selected_features_f_reg = np.array(feature_names)[f_selector_reg.get_support()]

# Get F-scores
f_scores_reg = pd.DataFrame({
    'feature': feature_names,
    'f_score': f_selector_reg.scores_,
    'p_value': f_selector_reg.pvalues_
}).sort_values('f_score', ascending=False)

print(f"Top 10 features selected by F-test:")
print(selected_features_f_reg)
print("\nF-scores and p-values:")
print(f_scores_reg.head(10))

# Mutual information for regression
mi_selector_reg = SelectKBest(score_func=mutual_info_regression, k=10)
X_mi_selected_reg = mi_selector_reg.fit_transform(X_features, df['target_regression'])
selected_features_mi_reg = np.array(feature_names)[mi_selector_reg.get_support()]

mi_scores_reg = pd.DataFrame({
    'feature': feature_names,
    'mi_score': mi_selector_reg.scores_
}).sort_values('mi_score', ascending=False)

print(f"\nTop 10 features selected by Mutual Information:")
print(selected_features_mi_reg)

# For classification target
print("\nCLASSIFICATION TARGET:")
print("-" * 30)

# F-test for classification
f_selector_clf = SelectKBest(score_func=f_classif, k=10)
X_f_selected_clf = f_selector_clf.fit_transform(X_features, df['target_classification'])
selected_features_f_clf = np.array(feature_names)[f_selector_clf.get_support()]

f_scores_clf = pd.DataFrame({
    'feature': feature_names,
    'f_score': f_selector_clf.scores_,
    'p_value': f_selector_clf.pvalues_
}).sort_values('f_score', ascending=False)

print(f"Top 10 features selected by F-test:")
print(selected_features_f_clf)

# Plot feature scores
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.barh(range(10), f_scores_reg.head(10)['f_score'])
plt.yticks(range(10), f_scores_reg.head(10)['feature'])
plt.xlabel('F-Score')
plt.title('Top 10 F-Scores (Regression)')

plt.subplot(2, 2, 2)
plt.barh(range(10), mi_scores_reg.head(10)['mi_score'])
plt.yticks(range(10), mi_scores_reg.head(10)['feature'])
plt.xlabel('MI Score')
plt.title('Top 10 Mutual Information Scores (Regression)')

plt.subplot(2, 2, 3)
plt.barh(range(10), f_scores_clf.head(10)['f_score'])
plt.yticks(range(10), f_scores_clf.head(10)['feature'])
plt.xlabel('F-Score')
plt.title('Top 10 F-Scores (Classification)')

plt.subplot(2, 2, 4)
plt.scatter(f_scores_reg['f_score'], f_scores_reg['p_value'])
plt.xlabel('F-Score')
plt.ylabel('P-Value')
plt.title('F-Score vs P-Value (Regression)')
plt.yscale('log')

plt.tight_layout()
plt.show()

# ===============================================================================
# 4. MODEL-BASED FEATURE SELECTION
# ===============================================================================

print("\n" + "="*60)
print("4. MODEL-BASED FEATURE SELECTION")
print("="*60)

# Random Forest feature importance
print("RANDOM FOREST IMPORTANCE:")
print("-" * 30)

# For regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_features, df['target_regression'])

rf_importance_reg = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 features by Random Forest (Regression):")
print(rf_importance_reg.head(10))

# Select features using Random Forest
rf_selector_reg = SelectFromModel(rf_reg, prefit=True)
X_rf_selected_reg = rf_selector_reg.transform(X_features)
selected_features_rf_reg = np.array(feature_names)[rf_selector_reg.get_support()]

print(f"Features selected by Random Forest: {len(selected_features_rf_reg)}")
print(selected_features_rf_reg)

# For classification
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_features, df['target_classification'])

rf_importance_clf = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

# Lasso feature selection (L1 regularization)
print("\nLASSO FEATURE SELECTION:")
print("-" * 30)

# For regression
lasso_reg = LassoCV(cv=5, random_state=42)
lasso_reg.fit(X_features, df['target_regression'])

lasso_coefs = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lasso_reg.coef_
}).sort_values('coefficient', key=abs, ascending=False)

# Features with non-zero coefficients
selected_features_lasso = lasso_coefs[lasso_coefs['coefficient'] != 0]['feature'].values
print(f"Features selected by Lasso: {len(selected_features_lasso)}")
print(selected_features_lasso)
print(f"Lasso alpha: {lasso_reg.alpha_:.4f}")

# Plot model-based feature importance
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.barh(range(10), rf_importance_reg.head(10)['importance'])
plt.yticks(range(10), rf_importance_reg.head(10)['feature'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance (Regression)')

plt.subplot(2, 2, 2)
plt.barh(range(10), rf_importance_clf.head(10)['importance'])
plt.yticks(range(10), rf_importance_clf.head(10)['feature'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance (Classification)')

plt.subplot(2, 2, 3)
nonzero_lasso = lasso_coefs[lasso_coefs['coefficient'] != 0]
plt.barh(range(len(nonzero_lasso)), abs(nonzero_lasso['coefficient']))
plt.yticks(range(len(nonzero_lasso)), nonzero_lasso['feature'])
plt.xlabel('|Coefficient|')
plt.title('Lasso Coefficients (Non-zero)')

plt.subplot(2, 2, 4)
plt.scatter(rf_importance_reg['importance'], abs(lasso_coefs['coefficient']))
plt.xlabel('Random Forest Importance')
plt.ylabel('|Lasso Coefficient|')
plt.title('RF Importance vs Lasso Coefficients')

plt.tight_layout()
plt.show()

# ===============================================================================
# 5. RECURSIVE FEATURE ELIMINATION
# ===============================================================================

print("\n" + "="*60)
print("5. RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*60)

# RFE with Random Forest
print("RFE WITH RANDOM FOREST:")
print("-" * 30)

rfe_rf = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=42), 
             n_features_to_select=10)
X_rfe_selected = rfe_rf.fit_transform(X_features, df['target_regression'])
selected_features_rfe = np.array(feature_names)[rfe_rf.get_support()]

print(f"Features selected by RFE: {len(selected_features_rfe)}")
print(selected_features_rfe)

# Feature ranking
rfe_ranking = pd.DataFrame({
    'feature': feature_names,
    'ranking': rfe_rf.ranking_,
    'selected': rfe_rf.get_support()
}).sort_values('ranking')

print("\nRFE Feature Rankings:")
print(rfe_ranking.head(15))

# RFE with Cross-Validation
print("\nRFE WITH CROSS-VALIDATION:")
print("-" * 30)

rfecv = RFECV(estimator=RandomForestRegressor(n_estimators=50, random_state=42),
              cv=5, scoring='neg_mean_squared_error')
X_rfecv_selected = rfecv.fit_transform(X_features, df['target_regression'])
selected_features_rfecv = np.array(feature_names)[rfecv.get_support()]

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Features selected by RFECV: {len(selected_features_rfecv)}")
print(selected_features_rfecv)

# Plot RFE results
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.barh(range(len(rfe_ranking)), rfe_ranking['ranking'])
plt.yticks(range(len(rfe_ranking)), rfe_ranking['feature'])
plt.xlabel('Ranking')
plt.title('RFE Feature Rankings')

plt.subplot(2, 2, 2)
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
         -rfecv.cv_results_['mean_test_score'])
plt.axvline(x=rfecv.n_features_, color='red', linestyle='--', 
            label=f'Optimal: {rfecv.n_features_}')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Score (MSE)')
plt.title('RFECV Performance')
plt.legend()

plt.subplot(2, 2, 3)
selected_counts = pd.Series([
    'Variance', 'F-test', 'MI', 'RF', 'Lasso', 'RFE', 'RFECV'
], index=[
    len(selected_features_variance),
    len(selected_features_f_reg),
    len(selected_features_mi_reg),
    len(selected_features_rf_reg),
    len(selected_features_lasso),
    len(selected_features_rfe),
    len(selected_features_rfecv)
])
plt.bar(selected_counts.values, selected_counts.index)
plt.xlabel('Number of Features Selected')
plt.title('Features Selected by Different Methods')

plt.tight_layout()
plt.show()

# ===============================================================================
# 6. FEATURE SELECTION COMPARISON
# ===============================================================================

print("\n" + "="*60)
print("6. FEATURE SELECTION COMPARISON")
print("="*60)

# Compare different selection methods
selection_methods = {
    'Variance_Selection': selected_features_variance,
    'F_test_Regression': selected_features_f_reg,
    'MI_Regression': selected_features_mi_reg,
    'Random_Forest': selected_features_rf_reg,
    'Lasso': selected_features_lasso,
    'RFE': selected_features_rfe,
    'RFECV': selected_features_rfecv
}

# Create comparison matrix
comparison_df = pd.DataFrame(index=feature_names)
for method, features in selection_methods.items():
    comparison_df[method] = comparison_df.index.isin(features)

# Count how many methods selected each feature
comparison_df['selection_count'] = comparison_df.sum(axis=1)
comparison_df = comparison_df.sort_values('selection_count', ascending=False)

print("Features selected by multiple methods:")
print(comparison_df[comparison_df['selection_count'] > 1].head(10))

# Most consistently selected features
consistently_selected = comparison_df[comparison_df['selection_count'] >= 3].index.tolist()
print(f"\nFeatures selected by 3+ methods: {len(consistently_selected)}")
print(consistently_selected)

# ===============================================================================
# 7. PERFORMANCE EVALUATION
# ===============================================================================

print("\n" + "="*60)
print("7. PERFORMANCE EVALUATION")
print("="*60)

# Evaluate different feature sets
X_train, X_test, y_train, y_test = train_test_split(
    X_features, df['target_regression'], test_size=0.2, random_state=42
)

feature_sets = {
    'All_Features': feature_names,
    'Top_10_F_test': selected_features_f_reg,
    'Random_Forest': selected_features_rf_reg,
    'Lasso': selected_features_lasso,
    'RFE': selected_features_rfe,
    'RFECV': selected_features_rfecv,
    'Consistently_Selected': consistently_selected
}

results = []
for name, features in feature_sets.items():
    if len(features) > 0:
        # Train model with selected features
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_subset, y_train)
        
        # Evaluate
        train_score = model.score(X_train_subset, y_train)
        test_score = model.score(X_test_subset, y_test)
        y_pred = model.predict(X_test_subset)
        mse = mean_squared_error(y_test, y_pred)
        
        results.append({
            'method': name,
            'n_features': len(features),
            'train_r2': train_score,
            'test_r2': test_score,
            'mse': mse,
            'features': features
        })

results_df = pd.DataFrame(results).sort_values('test_r2', ascending=False)
print("Performance comparison:")
print(results_df[['method', 'n_features', 'train_r2', 'test_r2', 'mse']])

# Plot performance comparison
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(results_df['n_features'], results_df['test_r2'])
for i, txt in enumerate(results_df['method']):
    plt.annotate(txt, (results_df.iloc[i]['n_features'], results_df.iloc[i]['test_r2']))
plt.xlabel('Number of Features')
plt.ylabel('Test R²')
plt.title('Test Performance vs Number of Features')

plt.subplot(2, 2, 2)
plt.bar(results_df['method'], results_df['test_r2'])
plt.xticks(rotation=45)
plt.ylabel('Test R²')
plt.title('Test Performance by Method')

plt.subplot(2, 2, 3)
plt.scatter(results_df['train_r2'], results_df['test_r2'])
for i, txt in enumerate(results_df['method']):
    plt.annotate(txt, (results_df.iloc[i]['train_r2'], results_df.iloc[i]['test_r2']))
plt.xlabel('Train R²')
plt.ylabel('Test R²')
plt.title('Train vs Test Performance')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

plt.tight_layout()
plt.show()

# ===============================================================================
# 8. FINAL RECOMMENDATIONS
# ===============================================================================

print("\n" + "="*60)
print("8. FINAL RECOMMENDATIONS")
print("="*60)

best_method = results_df.iloc[0]
print(f"Best performing method: {best_method['method']}")
print(f"Number of features: {best_method['n_features']}")
print(f"Test R²: {best_method['test_r2']:.4f}")
print(f"Selected features: {best_method['features']}")

# Summary statistics
print(f"\nSummary:")
print(f"- Original features: {len(feature_names)}")
print(f"- Best feature set size: {best_method['n_features']}")
print(f"- Feature reduction: {(1 - best_method['n_features']/len(feature_names))*100:.1f}%")
print(f"- Performance improvement over all features: {(best_method['test_r2'] - results_df[results_df['method']=='All_Features']['test_r2'].iloc[0])*100:.2f}%")

# Export selected features
selected_features_final = best_method['features']
# df_selected = df[list(selected_features_final) + ['target_regression', 'target_classification']]
# df_selected.to_csv('selected_features_dataset.csv', index=False)

print("\nFeature selection complete!")
print("Best features ready for modeling!")
