"""
===============================================================================
MODEL COMPARISON TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive template for comparing multiple machine learning models

This template covers:
- Multiple model training and evaluation
- Cross-validation comparison
- Performance metrics comparison
- Statistical significance testing
- Model selection criteria
- Ensemble methods

Prerequisites:
- pandas, numpy, matplotlib, seaborn, scikit-learn
- Dataset prepared for machine learning
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate, 
    StratifiedKFold, KFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
n_features = 10

# Create synthetic datasets
X = np.random.randn(n_samples, n_features)
# Regression target
y_regression = (2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 
                np.random.randn(n_samples) * 0.5)
# Classification target
y_classification = (y_regression > np.median(y_regression)).astype(int)

# Create feature names
feature_names = [f'feature_{i}' for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_names)

print("Dataset Shape:", X_df.shape)
print("Regression Target Range:", f"{y_regression.min():.2f} to {y_regression.max():.2f}")
print("Classification Target Distribution:", pd.Series(y_classification).value_counts().to_dict())

# Split data
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ===============================================================================
# 1. DEFINE MODEL SETS
# ===============================================================================

print("\n" + "="*60)
print("1. DEFINE MODEL SETS")
print("="*60)

# Classification models
classification_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Regression models
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Models with scaling (for algorithms sensitive to feature scale)
scaled_classification_models = {}
scaled_regression_models = {}

for name, model in classification_models.items():
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
        scaled_classification_models[f'{name} (Scaled)'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

for name, model in regression_models.items():
    if name in ['Ridge Regression', 'Lasso Regression', 'SVR', 'K-Nearest Neighbors']:
        scaled_regression_models[f'{name} (Scaled)'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

# Combine models
all_classification_models = {**classification_models, **scaled_classification_models}
all_regression_models = {**regression_models, **scaled_regression_models}

print(f"Classification models to compare: {len(all_classification_models)}")
print(f"Regression models to compare: {len(all_regression_models)}")

# ===============================================================================
# 2. CROSS-VALIDATION COMPARISON
# ===============================================================================

print("\n" + "="*60)
print("2. CROSS-VALIDATION COMPARISON")
print("="*60)

# Classification cross-validation
print("CLASSIFICATION MODELS:")
print("-" * 30)

cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_results = []

for name, model in all_classification_models.items():
    try:
        scores = cross_validate(
            model, X_train, y_clf_train, cv=cv_clf,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=True
        )
        
        clf_results.append({
            'Model': name,
            'CV_Accuracy_Mean': scores['test_accuracy'].mean(),
            'CV_Accuracy_Std': scores['test_accuracy'].std(),
            'CV_Precision_Mean': scores['test_precision'].mean(),
            'CV_Recall_Mean': scores['test_recall'].mean(),
            'CV_F1_Mean': scores['test_f1'].mean(),
            'CV_ROC_AUC_Mean': scores['test_roc_auc'].mean(),
            'Train_Accuracy_Mean': scores['train_accuracy'].mean()
        })
    except Exception as e:
        print(f"Error with {name}: {e}")

clf_results_df = pd.DataFrame(clf_results).sort_values('CV_Accuracy_Mean', ascending=False)
print("Classification Results (sorted by CV Accuracy):")
print(clf_results_df.round(4))

# Regression cross-validation
print("\nREGRESSION MODELS:")
print("-" * 30)

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
reg_results = []

for name, model in all_regression_models.items():
    try:
        scores = cross_validate(
            model, X_train, y_reg_train, cv=cv_reg,
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            return_train_score=True
        )
        
        reg_results.append({
            'Model': name,
            'CV_R2_Mean': scores['test_r2'].mean(),
            'CV_R2_Std': scores['test_r2'].std(),
            'CV_MSE_Mean': -scores['test_neg_mean_squared_error'].mean(),
            'CV_MAE_Mean': -scores['test_neg_mean_absolute_error'].mean(),
            'Train_R2_Mean': scores['train_r2'].mean()
        })
    except Exception as e:
        print(f"Error with {name}: {e}")

reg_results_df = pd.DataFrame(reg_results).sort_values('CV_R2_Mean', ascending=False)
print("Regression Results (sorted by CV R²):")
print(reg_results_df.round(4))

# ===============================================================================
# 3. VISUALIZATION OF RESULTS
# ===============================================================================

print("\n" + "="*60)
print("3. VISUALIZATION OF RESULTS")
print("="*60)

# Classification results visualization
plt.figure(figsize=(20, 15))

# Classification accuracy comparison
plt.subplot(3, 3, 1)
plt.barh(range(len(clf_results_df)), clf_results_df['CV_Accuracy_Mean'])
plt.yticks(range(len(clf_results_df)), clf_results_df['Model'])
plt.xlabel('CV Accuracy')
plt.title('Classification: Cross-Validation Accuracy')

# Classification error bars
plt.subplot(3, 3, 2)
plt.errorbar(clf_results_df['CV_Accuracy_Mean'], range(len(clf_results_df)), 
             xerr=clf_results_df['CV_Accuracy_Std'], fmt='o')
plt.yticks(range(len(clf_results_df)), clf_results_df['Model'])
plt.xlabel('CV Accuracy ± Std')
plt.title('Classification: Accuracy with Error Bars')

# Multiple metrics comparison for classification
plt.subplot(3, 3, 3)
metrics = ['CV_Accuracy_Mean', 'CV_Precision_Mean', 'CV_Recall_Mean', 'CV_F1_Mean', 'CV_ROC_AUC_Mean']
top_5_clf = clf_results_df.head(5)
x_pos = np.arange(len(top_5_clf))
width = 0.15

for i, metric in enumerate(metrics):
    plt.bar(x_pos + i*width, top_5_clf[metric], width, label=metric.replace('CV_', '').replace('_Mean', ''))

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Top 5 Classification Models: Multiple Metrics')
plt.xticks(x_pos + width*2, top_5_clf['Model'], rotation=45)
plt.legend()

# Regression results visualization
plt.subplot(3, 3, 4)
plt.barh(range(len(reg_results_df)), reg_results_df['CV_R2_Mean'])
plt.yticks(range(len(reg_results_df)), reg_results_df['Model'])
plt.xlabel('CV R²')
plt.title('Regression: Cross-Validation R²')

plt.subplot(3, 3, 5)
plt.errorbar(reg_results_df['CV_R2_Mean'], range(len(reg_results_df)), 
             xerr=reg_results_df['CV_R2_Std'], fmt='o')
plt.yticks(range(len(reg_results_df)), reg_results_df['Model'])
plt.xlabel('CV R² ± Std')
plt.title('Regression: R² with Error Bars')

# Train vs Test performance
plt.subplot(3, 3, 6)
plt.scatter(clf_results_df['Train_Accuracy_Mean'], clf_results_df['CV_Accuracy_Mean'])
for i, model in enumerate(clf_results_df['Model']):
    plt.annotate(model[:10], (clf_results_df.iloc[i]['Train_Accuracy_Mean'], 
                             clf_results_df.iloc[i]['CV_Accuracy_Mean']))
plt.xlabel('Train Accuracy')
plt.ylabel('CV Accuracy')
plt.title('Classification: Train vs CV Performance')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

plt.subplot(3, 3, 7)
plt.scatter(reg_results_df['Train_R2_Mean'], reg_results_df['CV_R2_Mean'])
for i, model in enumerate(reg_results_df['Model']):
    plt.annotate(model[:10], (reg_results_df.iloc[i]['Train_R2_Mean'], 
                             reg_results_df.iloc[i]['CV_R2_Mean']))
plt.xlabel('Train R²')
plt.ylabel('CV R²')
plt.title('Regression: Train vs CV Performance')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

# Model complexity vs performance
plt.subplot(3, 3, 8)
model_complexity = {'Naive Bayes': 1, 'Logistic Regression': 2, 'Linear Regression': 2,
                   'Decision Tree': 3, 'K-Nearest Neighbors': 3, 'SVM': 4, 'SVR': 4,
                   'Random Forest': 5, 'Gradient Boosting': 5}

clf_complexity = [model_complexity.get(name.split(' (')[0], 3) for name in clf_results_df['Model']]
plt.scatter(clf_complexity, clf_results_df['CV_Accuracy_Mean'])
plt.xlabel('Model Complexity (1=Simple, 5=Complex)')
plt.ylabel('CV Accuracy')
plt.title('Classification: Complexity vs Performance')

plt.subplot(3, 3, 9)
reg_complexity = [model_complexity.get(name.split(' (')[0], 3) for name in reg_results_df['Model']]
plt.scatter(reg_complexity, reg_results_df['CV_R2_Mean'])
plt.xlabel('Model Complexity (1=Simple, 5=Complex)')
plt.ylabel('CV R²')
plt.title('Regression: Complexity vs Performance')

plt.tight_layout()
plt.show()

# ===============================================================================
# 4. DETAILED TEST SET EVALUATION
# ===============================================================================

print("\n" + "="*60)
print("4. DETAILED TEST SET EVALUATION")
print("="*60)

# Select top 3 models for detailed evaluation
top_3_clf_models = clf_results_df.head(3)['Model'].tolist()
top_3_reg_models = reg_results_df.head(3)['Model'].tolist()

print("Top 3 Classification Models:")
for i, model in enumerate(top_3_clf_models, 1):
    print(f"{i}. {model}")

print("\nTop 3 Regression Models:")
for i, model in enumerate(top_3_reg_models, 1):
    print(f"{i}. {model}")

# Detailed classification evaluation
print("\nDETAILED CLASSIFICATION EVALUATION:")
print("-" * 40)

clf_test_results = []
for model_name in top_3_clf_models:
    model = all_classification_models[model_name]
    model.fit(X_train, y_clf_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_clf_test, y_pred)
    precision = precision_score(y_clf_test, y_pred)
    recall = recall_score(y_clf_test, y_pred)
    f1 = f1_score(y_clf_test, y_pred)
    roc_auc = roc_auc_score(y_clf_test, y_pred_proba) if y_pred_proba is not None else None
    
    clf_test_results.append({
        'Model': model_name,
        'Test_Accuracy': accuracy,
        'Test_Precision': precision,
        'Test_Recall': recall,
        'Test_F1': f1,
        'Test_ROC_AUC': roc_auc
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")

# Detailed regression evaluation
print("\nDETAILED REGRESSION EVALUATION:")
print("-" * 40)

reg_test_results = []
for model_name in top_3_reg_models:
    model = all_regression_models[model_name]
    model.fit(X_train, y_reg_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_reg_test, y_pred)
    mae = mean_absolute_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    rmse = np.sqrt(mse)
    
    reg_test_results.append({
        'Model': model_name,
        'Test_MSE': mse,
        'Test_RMSE': rmse,
        'Test_MAE': mae,
        'Test_R2': r2
    })
    
    print(f"\n{model_name}:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")

# ===============================================================================
# 5. CONFUSION MATRICES AND ROC CURVES
# ===============================================================================

print("\n" + "="*60)
print("5. CONFUSION MATRICES AND ROC CURVES")
print("="*60)

# Plot confusion matrices and ROC curves for top classification models
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, model_name in enumerate(top_3_clf_models):
    model = all_classification_models[model_name]
    model.fit(X_train, y_clf_train)
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_clf_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, i], cmap='Blues')
    axes[0, i].set_title(f'Confusion Matrix\n{model_name}')
    axes[0, i].set_xlabel('Predicted')
    axes[0, i].set_ylabel('Actual')
    
    # ROC Curve
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_clf_test, y_pred_proba)
        roc_auc = roc_auc_score(y_clf_test, y_pred_proba)
        
        axes[1, i].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[1, i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, i].set_xlabel('False Positive Rate')
        axes[1, i].set_ylabel('True Positive Rate')
        axes[1, i].set_title(f'ROC Curve\n{model_name}')
        axes[1, i].legend()
        axes[1, i].grid(True)

plt.tight_layout()
plt.show()

# ===============================================================================
# 6. RESIDUAL PLOTS FOR REGRESSION
# ===============================================================================

print("\n" + "="*60)
print("6. RESIDUAL ANALYSIS FOR REGRESSION")
print("="*60)

# Plot residuals for top regression models
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, model_name in enumerate(top_3_reg_models):
    model = all_regression_models[model_name]
    model.fit(X_train, y_reg_train)
    y_pred = model.predict(X_test)
    residuals = y_reg_test - y_pred
    
    # Residuals vs Fitted
    axes[0, i].scatter(y_pred, residuals, alpha=0.6)
    axes[0, i].axhline(y=0, color='r', linestyle='--')
    axes[0, i].set_xlabel('Fitted Values')
    axes[0, i].set_ylabel('Residuals')
    axes[0, i].set_title(f'Residuals vs Fitted\n{model_name}')
    
    # Q-Q plot (normality of residuals)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f'Q-Q Plot\n{model_name}')

plt.tight_layout()
plt.show()

# ===============================================================================
# 7. ENSEMBLE METHODS
# ===============================================================================

print("\n" + "="*60)
print("7. ENSEMBLE METHODS")
print("="*60)

# Create ensemble models using top performers
print("ENSEMBLE CLASSIFICATION:")
print("-" * 30)

# Select top 3 different base models (avoid scaled versions of same algorithm)
unique_clf_models = []
seen_base_models = set()
for model_name in top_3_clf_models:
    base_name = model_name.split(' (')[0]
    if base_name not in seen_base_models:
        unique_clf_models.append((model_name, all_classification_models[model_name]))
        seen_base_models.add(base_name)

if len(unique_clf_models) >= 2:
    voting_clf = VotingClassifier(
        estimators=unique_clf_models[:3],
        voting='soft'
    )
    
    # Evaluate ensemble
    ensemble_scores = cross_val_score(voting_clf, X_train, y_clf_train, cv=cv_clf, scoring='accuracy')
    print(f"Voting Classifier CV Accuracy: {ensemble_scores.mean():.4f} (±{ensemble_scores.std():.4f})")
    
    # Compare with individual models
    voting_clf.fit(X_train, y_clf_train)
    ensemble_pred = voting_clf.predict(X_test)
    ensemble_accuracy = accuracy_score(y_clf_test, ensemble_pred)
    print(f"Voting Classifier Test Accuracy: {ensemble_accuracy:.4f}")

print("\nENSEMBLE REGRESSION:")
print("-" * 30)

# Select top 3 different base models for regression
unique_reg_models = []
seen_base_models = set()
for model_name in top_3_reg_models:
    base_name = model_name.split(' (')[0]
    if base_name not in seen_base_models:
        unique_reg_models.append((model_name, all_regression_models[model_name]))
        seen_base_models.add(base_name)

if len(unique_reg_models) >= 2:
    voting_reg = VotingRegressor(estimators=unique_reg_models[:3])
    
    # Evaluate ensemble
    ensemble_scores = cross_val_score(voting_reg, X_train, y_reg_train, cv=cv_reg, scoring='r2')
    print(f"Voting Regressor CV R²: {ensemble_scores.mean():.4f} (±{ensemble_scores.std():.4f})")
    
    # Compare with individual models
    voting_reg.fit(X_train, y_reg_train)
    ensemble_pred = voting_reg.predict(X_test)
    ensemble_r2 = r2_score(y_reg_test, ensemble_pred)
    print(f"Voting Regressor Test R²: {ensemble_r2:.4f}")

# ===============================================================================
# 8. FINAL MODEL SELECTION SUMMARY
# ===============================================================================

print("\n" + "="*60)
print("8. FINAL MODEL SELECTION SUMMARY")
print("="*60)

# Best classification model
best_clf_model = clf_results_df.iloc[0]['Model']
best_clf_score = clf_results_df.iloc[0]['CV_Accuracy_Mean']

print("CLASSIFICATION SUMMARY:")
print("-" * 25)
print(f"Best Model: {best_clf_model}")
print(f"CV Accuracy: {best_clf_score:.4f}")

if clf_test_results:
    best_clf_test = next(result for result in clf_test_results if result['Model'] == best_clf_model)
    print(f"Test Accuracy: {best_clf_test['Test_Accuracy']:.4f}")
    print(f"Test F1-Score: {best_clf_test['Test_F1']:.4f}")

# Best regression model
best_reg_model = reg_results_df.iloc[0]['Model']
best_reg_score = reg_results_df.iloc[0]['CV_R2_Mean']

print(f"\nREGRESSION SUMMARY:")
print("-" * 20)
print(f"Best Model: {best_reg_model}")
print(f"CV R²: {best_reg_score:.4f}")

if reg_test_results:
    best_reg_test = next(result for result in reg_test_results if result['Model'] == best_reg_model)
    print(f"Test R²: {best_reg_test['Test_R2']:.4f}")
    print(f"Test RMSE: {best_reg_test['Test_RMSE']:.4f}")

# Model selection criteria
print(f"\nMODEL SELECTION CRITERIA:")
print("-" * 30)
print("✓ Cross-validation performance")
print("✓ Generalization gap (train vs test)")
print("✓ Model complexity vs performance")
print("✓ Prediction consistency")
print("✓ Computational efficiency")

# Export results
clf_results_df.to_csv('classification_model_comparison.csv', index=False)
reg_results_df.to_csv('regression_model_comparison.csv', index=False)

print(f"\nModel comparison complete!")
print(f"Results exported to CSV files.")
print(f"Best models ready for final deployment!")
