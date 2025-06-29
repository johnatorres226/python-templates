# Replace 'your_data.csv' with your dataset
# Random Forest Template - Comprehensive Analysis with Feature Importance and Tuning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== RANDOM FOREST ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Replace 'target_column' with your actual target variable
target_column = 'target_column'  # Replace with your target variable name

# Auto-detect target if not specified
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if target_column not in df.columns:
    if numerical_cols:
        target_column = numerical_cols[0]
        print(f"Using '{target_column}' as example target variable")
    else:
        print("Error: No suitable target variable found.")
        exit()

# Determine if this is classification or regression
target_type = 'classification' if df[target_column].nunique() <= 10 else 'regression'
print(f"Analysis type: {target_type}")
print(f"Target variable: {target_column}")

if target_type == 'classification':
    print(f"Target classes: {sorted(df[target_column].unique())}")
else:
    print(f"Target range: {df[target_column].min():.3f} to {df[target_column].max():.3f}")

# 1. DATA PREPARATION
print("\n=== 1. DATA PREPARATION ===")

# Get feature columns
feature_cols = [col for col in df.columns if col != target_column]
print(f"Features: {len(feature_cols)} columns")

# Handle missing values and prepare features
df_clean = df.copy()

# Simple preprocessing
for col in categorical_cols:
    if col != target_column:
        # Label encode categorical features
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].fillna('Missing'))

# Handle missing numerical values
for col in numerical_cols:
    if col != target_column:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Remove rows with missing target
df_clean = df_clean.dropna(subset=[target_column])

print(f"Clean data shape: {df_clean.shape}")
print(f"Features after preprocessing: {len(feature_cols)}")

# Prepare final X and y
X = df_clean[feature_cols]
y = df_clean[target_column]

# Encode target if classification
if target_type == 'classification':
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    target_classes = le_target.classes_
    print(f"Encoded target classes: {target_classes}")
else:
    y_encoded = y
    target_classes = None

# 2. TRAIN-TEST SPLIT
print("\n=== 2. TRAIN-TEST SPLIT ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, 
    stratify=y_encoded if target_type == 'classification' else None
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

if target_type == 'classification':
    print("Class distribution in training set:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    for i, count in train_counts.items():
        class_name = target_classes[i] if target_classes is not None else i
        print(f"  {class_name}: {count} ({count/len(y_train)*100:.1f}%)")

# 3. BASELINE MODEL
print("\n=== 3. BASELINE RANDOM FOREST ===")

if target_type == 'classification':
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit baseline model
rf_model.fit(X_train, y_train)

# Baseline predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Baseline evaluation
if target_type == 'classification':
    train_accuracy = rf_model.score(X_train, y_train)
    test_accuracy = rf_model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y_encoded, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # AUC for binary classification
    if len(np.unique(y_encoded)) == 2:
        y_test_proba = rf_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_test_proba)
        print(f"Test AUC: {auc_score:.3f}")

else:
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y_encoded, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 4. FEATURE IMPORTANCE ANALYSIS
print("\n=== 4. FEATURE IMPORTANCE ANALYSIS ===")

# Get feature importances
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'][::-1])
plt.yticks(range(len(top_features)), top_features['feature'][::-1])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance (Top 15)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cumulative importance
cumulative_importance = np.cumsum(feature_importance_df['importance'])
n_features_90 = np.argmax(cumulative_importance >= 0.9) + 1
n_features_95 = np.argmax(cumulative_importance >= 0.95) + 1

print(f"\nFeatures needed for 90% importance: {n_features_90}")
print(f"Features needed for 95% importance: {n_features_95}")

# Plot cumulative importance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
plt.axhline(y=0.9, color='r', linestyle='--', label='90%')
plt.axhline(y=0.95, color='orange', linestyle='--', label='95%')
plt.axvline(x=n_features_90, color='r', linestyle=':', alpha=0.7)
plt.axvline(x=n_features_95, color='orange', linestyle=':', alpha=0.7)
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. PERMUTATION IMPORTANCE
print("\n=== 5. PERMUTATION IMPORTANCE ===")

# Calculate permutation importance
perm_importance = permutation_importance(rf_model, X_test, y_test, 
                                       random_state=42, n_repeats=5)

perm_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'perm_importance_mean': perm_importance.importances_mean,
    'perm_importance_std': perm_importance.importances_std
}).sort_values('perm_importance_mean', ascending=False)

print("Top 10 Features by Permutation Importance:")
print(perm_importance_df.head(10))

# Compare built-in vs permutation importance
comparison_df = feature_importance_df.merge(
    perm_importance_df[['feature', 'perm_importance_mean']], on='feature'
)

plt.figure(figsize=(10, 8))
plt.scatter(comparison_df['importance'], comparison_df['perm_importance_mean'], alpha=0.6)
plt.xlabel('Built-in Feature Importance')
plt.ylabel('Permutation Importance')
plt.title('Built-in vs Permutation Feature Importance')
plt.grid(True, alpha=0.3)

# Add diagonal line
max_val = max(comparison_df['importance'].max(), comparison_df['perm_importance_mean'].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. HYPERPARAMETER TUNING
print("\n=== 6. HYPERPARAMETER TUNING ===")

# Define parameter grid
if target_type == 'classification':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    scoring = 'accuracy'
else:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    scoring = 'r2'

# Use RandomizedSearchCV for efficiency
print("Performing randomized hyperparameter search...")

if target_type == 'classification':
    rf_random = RandomForestClassifier(random_state=42)
else:
    rf_random = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    rf_random, param_grid, n_iter=20, cv=3, 
    scoring=scoring, random_state=42, n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"Best cross-validation score: {random_search.best_score_:.3f}")

# 7. FINAL MODEL EVALUATION
print("\n=== 7. FINAL MODEL EVALUATION ===")

# Use best model
best_model = random_search.best_estimator_

# Predictions
y_train_pred_best = best_model.predict(X_train)
y_test_pred_best = best_model.predict(X_test)

if target_type == 'classification':
    # Classification metrics
    train_acc_best = best_model.score(X_train, y_train)
    test_acc_best = best_model.score(X_test, y_test)
    
    print(f"Tuned Model Training Accuracy: {train_acc_best:.3f}")
    print(f"Tuned Model Test Accuracy: {test_acc_best:.3f}")
    print(f"Improvement: {test_acc_best - test_accuracy:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    if target_classes is not None:
        target_names = [str(cls) for cls in target_classes]
        print(classification_report(y_test, y_test_pred_best, target_names=target_names))
    else:
        print(classification_report(y_test, y_test_pred_best))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred_best)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_classes if target_classes is not None else None,
                yticklabels=target_classes if target_classes is not None else None)
    plt.title('Confusion Matrix (Tuned Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # ROC curve for binary classification
    if len(np.unique(y_encoded)) == 2:
        y_test_proba_best = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba_best)
        auc_best = roc_auc_score(y_test, y_test_proba_best)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_best:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Tuned Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

else:
    # Regression metrics
    train_r2_best = r2_score(y_train, y_train_pred_best)
    test_r2_best = r2_score(y_test, y_test_pred_best)
    train_rmse_best = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
    test_rmse_best = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
    test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
    
    print(f"Tuned Model Training R²: {train_r2_best:.3f}")
    print(f"Tuned Model Test R²: {test_r2_best:.3f}")
    print(f"Tuned Model Test RMSE: {test_rmse_best:.3f}")
    print(f"Tuned Model Test MAE: {test_mae_best:.3f}")
    print(f"R² Improvement: {test_r2_best - test_r2:.3f}")
    
    # Actual vs predicted plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_test_pred_best, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred_best.min())
    max_val = max(y_test.max(), y_test_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted (R² = {test_r2_best:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Residuals plot
    residuals = y_test - y_test_pred_best
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_pred_best, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 8. MODEL INSIGHTS
print("\n=== 8. MODEL INSIGHTS ===")

# Tree depth analysis
tree_depths = [tree.get_depth() for tree in best_model.estimators_]
print(f"Average tree depth: {np.mean(tree_depths):.1f}")
print(f"Tree depth range: {min(tree_depths)} to {max(tree_depths)}")

# Out-of-bag score (if available)
if hasattr(best_model, 'oob_score_') and best_model.oob_score_ is not None:
    print(f"Out-of-bag score: {best_model.oob_score_:.3f}")

# Feature selection based on importance
important_features = feature_importance_df[feature_importance_df['importance'] > 0.01]['feature'].tolist()
print(f"\nFeatures with >1% importance: {len(important_features)}")
print(f"Total importance captured: {feature_importance_df[feature_importance_df['importance'] > 0.01]['importance'].sum():.3f}")

print("\n=== RANDOM FOREST SUMMARY ===")
print("Model Characteristics:")
print(f"- Number of trees: {best_model.n_estimators}")
print(f"- Max depth: {best_model.max_depth}")
print(f"- Min samples split: {best_model.min_samples_split}")
print(f"- Min samples leaf: {best_model.min_samples_leaf}")
print(f"- Max features: {best_model.max_features}")

print("\nModel Strengths:")
print("✓ Handles mixed data types well")
print("✓ Provides feature importance rankings")
print("✓ Robust to outliers")
print("✓ Less prone to overfitting than single trees")
print("✓ Can capture non-linear relationships")

print("\nConsiderations:")
print("- May overfit with very noisy data")
print("- Can be biased toward features with more levels")
print("- Less interpretable than single decision trees")
print("- Memory intensive for large datasets")

print("\nRandom Forest analysis complete.")
