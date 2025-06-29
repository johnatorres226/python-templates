# ================================================================
# K-NEAREST NEIGHBORS (KNN) TEMPLATE
# ================================================================
# Replace 'your_data.csv' with your actual dataset
# Replace 'target_column' with your actual target variable name
# This template covers both classification and regression variants
# KNN is a lazy learning algorithm - no explicit training phase
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# LOAD AND PREPARE DATA
# ================================================================

# Load your dataset
df = pd.read_csv('your_data.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Define target variable (replace with your actual target column)
target_column = 'target_column'
y = df[target_column]
X = df.drop(target_column, axis=1)

# Determine if this is classification or regression
is_classification = y.dtype == 'object' or len(y.unique()) < 10
print(f"\nProblem type: {'Classification' if is_classification else 'Regression'}")
print(f"Target variable unique values: {len(y.unique())}")

# ================================================================
# DATA PREPROCESSING
# ================================================================

# Handle categorical variables in features
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=[np.number]).columns

print(f"\nCategorical columns: {list(categorical_columns)}")
print(f"Numerical columns: {list(numerical_columns)}")

# Encode categorical variables
le_dict = {}
X_processed = X.copy()

for col in categorical_columns:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    le_dict[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Handle missing values (simple imputation)
if X_processed.isnull().sum().sum() > 0:
    print("\nImputing missing values with median for numerical, mode for categorical...")
    for col in X_processed.columns:
        if X_processed[col].isnull().sum() > 0:
            if col in numerical_columns:
                X_processed[col].fillna(X_processed[col].median(), inplace=True)
            else:
                X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)

# Encode target variable if classification
if is_classification:
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        print(f"\nTarget encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    else:
        y_encoded = y
        le_target = None
else:
    y_encoded = y
    le_target = None

# ================================================================
# FEATURE SCALING (CRITICAL FOR KNN)
# ================================================================

# Split data first to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, 
    stratify=y_encoded if is_classification else None
)

# Scale features (essential for KNN since it uses distance metrics)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Check scaling results
print(f"\nFeature scaling verification:")
print(f"Training set mean: {X_train_scaled.mean(axis=0)[:5]}")  # Should be ~0
print(f"Training set std: {X_train_scaled.std(axis=0)[:5]}")    # Should be ~1

# ================================================================
# OPTIMAL K SELECTION
# ================================================================

print("\n" + "="*50)
print("FINDING OPTIMAL K")
print("="*50)

# Test different values of k
k_range = range(1, min(31, len(X_train) // 2))  # Reasonable range for k
cv_scores = []

# Choose appropriate model
if is_classification:
    base_model = KNeighborsClassifier()
    scoring = 'accuracy'
else:
    base_model = KNeighborsRegressor()
    scoring = 'neg_mean_squared_error'

# Cross-validation for different k values
for k in k_range:
    if is_classification:
        knn = KNeighborsClassifier(n_neighbors=k)
    else:
        knn = KNeighborsRegressor(n_neighbors=k)
    
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring=scoring)
    cv_scores.append(scores.mean())

# Find optimal k
optimal_k = k_range[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")
print(f"Best CV score: {max(cv_scores):.4f}")

# Plot k selection
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, cv_scores, 'bo-')
plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Score')
plt.title('K Selection via Cross-Validation')
plt.grid(True, alpha=0.3)

# Plot error rate for classification or MSE for regression
if is_classification:
    error_rates = [1 - score for score in cv_scores]
    plt.subplot(1, 2, 2)
    plt.plot(k_range, error_rates, 'ro-')
    plt.axvline(x=optimal_k, color='blue', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs K')
    plt.grid(True, alpha=0.3)
else:
    mse_scores = [-score for score in cv_scores]  # Convert back from negative MSE
    plt.subplot(1, 2, 2)
    plt.plot(k_range, mse_scores, 'ro-')
    plt.axvline(x=optimal_k, color='blue', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs K')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================================================
# HYPERPARAMETER TUNING
# ================================================================

print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

# Define parameter grid
param_grid = {
    'n_neighbors': [optimal_k-2, optimal_k-1, optimal_k, optimal_k+1, optimal_k+2],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Remove negative k values
param_grid['n_neighbors'] = [k for k in param_grid['n_neighbors'] if k > 0]

# Grid search
if is_classification:
    knn_grid = KNeighborsClassifier()
    scoring = 'accuracy'
else:
    knn_grid = KNeighborsRegressor()
    scoring = 'neg_mean_squared_error'

grid_search = GridSearchCV(knn_grid, param_grid, cv=5, scoring=scoring, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Get the best model
best_knn = grid_search.best_estimator_

# ================================================================
# MODEL TRAINING AND PREDICTION
# ================================================================

print("\n" + "="*50)
print("MODEL TRAINING AND PREDICTION")
print("="*50)

# Make predictions
y_pred = best_knn.predict(X_test_scaled)

# For classification, also get prediction probabilities
if is_classification:
    try:
        y_pred_proba = best_knn.predict_proba(X_test_scaled)
    except:
        y_pred_proba = None

# ================================================================
# MODEL EVALUATION
# ================================================================

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

if is_classification:
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print(f"\nClassification Report:")
    if le_target:
        target_names = le_target.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
    else:
        print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve for binary classification
    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

else:
    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Residual plots
    residuals = y_test - y_pred
    
    plt.figure(figsize=(15, 5))
    
    # Predicted vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Residuals histogram
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ================================================================
# FEATURE IMPORTANCE (DISTANCE-BASED ANALYSIS)
# ================================================================

print("\n" + "="*50)
print("FEATURE ANALYSIS")
print("="*50)

# Since KNN doesn't provide feature importance directly, we can analyze
# the impact of each feature by looking at their variance after scaling
feature_variance = np.var(X_train_scaled, axis=0)
feature_names = X.columns

# Create feature importance plot
plt.figure(figsize=(12, 6))

# Sort features by variance
sorted_idx = np.argsort(feature_variance)[::-1]
sorted_features = feature_names[sorted_idx]
sorted_variance = feature_variance[sorted_idx]

plt.subplot(1, 2, 1)
plt.barh(range(len(sorted_features)), sorted_variance)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('Variance (after scaling)')
plt.title('Feature Variance Analysis')
plt.grid(True, alpha=0.3)

# Distance analysis - compute average distance to k nearest neighbors for each feature
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=optimal_k)
nn.fit(X_train_scaled)
distances, indices = nn.kneighbors(X_train_scaled)
avg_distances = np.mean(distances, axis=1)

plt.subplot(1, 2, 2)
plt.hist(avg_distances, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Average Distance to K Nearest Neighbors')
plt.ylabel('Frequency')
plt.title('Distribution of Average Distances')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nTop 5 features by variance:")
for i in range(min(5, len(sorted_features))):
    print(f"{i+1}. {sorted_features[i]}: {sorted_variance[i]:.4f}")

# ================================================================
# CROSS-VALIDATION ANALYSIS
# ================================================================

print("\n" + "="*50)
print("CROSS-VALIDATION ANALYSIS")
print("="*50)

# Perform detailed cross-validation
cv_scores_detailed = cross_val_score(best_knn, X_train_scaled, y_train, cv=5, scoring=scoring)

print(f"Cross-validation scores: {cv_scores_detailed}")
print(f"Mean CV score: {cv_scores_detailed.mean():.4f}")
print(f"Standard deviation: {cv_scores_detailed.std():.4f}")
print(f"95% Confidence interval: [{cv_scores_detailed.mean() - 1.96*cv_scores_detailed.std():.4f}, {cv_scores_detailed.mean() + 1.96*cv_scores_detailed.std():.4f}]")

# ================================================================
# MODEL INSIGHTS AND RECOMMENDATIONS
# ================================================================

print("\n" + "="*50)
print("MODEL INSIGHTS AND RECOMMENDATIONS")
print("="*50)

print(f"Final Model Configuration:")
print(f"- Algorithm: K-Nearest Neighbors")
print(f"- Number of neighbors (k): {best_knn.n_neighbors}")
print(f"- Weight function: {best_knn.weights}")
print(f"- Distance metric: {best_knn.metric}")
print(f"- Problem type: {'Classification' if is_classification else 'Regression'}")

print(f"\nKey Insights:")
print(f"- Dataset size: {len(df)} samples, {len(X.columns)} features")
print(f"- Feature scaling was applied (critical for KNN)")
print(f"- Optimal k found through cross-validation")

print(f"\nStrengths of this KNN model:")
print(f"- Simple and interpretable")
print(f"- No assumptions about data distribution")
print(f"- Handles non-linear relationships naturally")
print(f"- Good for local pattern recognition")

print(f"\nLimitations to consider:")
print(f"- Computationally expensive for large datasets")
print(f"- Sensitive to irrelevant features")
print(f"- Performance degrades in high dimensions (curse of dimensionality)")
print(f"- Sensitive to class imbalance (if classification)")

print(f"\nRecommendations:")
if len(X.columns) > 20:
    print(f"- Consider dimensionality reduction (PCA) due to high feature count")
if is_classification and len(np.unique(y_encoded)) > 2:
    print(f"- Monitor per-class performance for multi-class problem")
print(f"- Consider ensemble methods if you need higher performance")
print(f"- Feature selection might improve performance and reduce computation")

# ================================================================
# SAVE RESULTS
# ================================================================

# Create a summary dictionary
results_summary = {
    'model_type': 'K-Nearest Neighbors',
    'problem_type': 'Classification' if is_classification else 'Regression',
    'optimal_k': best_knn.n_neighbors,
    'best_params': grid_search.best_params_,
    'cv_score': grid_search.best_score_,
    'dataset_shape': df.shape
}

if is_classification:
    results_summary['test_accuracy'] = accuracy_score(y_test, y_pred)
else:
    results_summary['test_r2'] = r2_score(y_test, y_pred)
    results_summary['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nResults Summary:")
for key, value in results_summary.items():
    print(f"{key}: {value}")

print(f"\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print(f"Your KNN model is ready!")
print(f"Key files generated: Feature importance analysis, residual plots, performance metrics")
