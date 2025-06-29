"""
===============================================================================
DIAGNOSTIC PLOTS TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive model diagnostic and evaluation plots

This template covers:
- Classification diagnostic plots (ROC, PR, confusion matrix)
- Regression diagnostic plots (residuals, Q-Q, prediction vs actual)
- Feature importance visualizations
- Learning curves and validation curves
- Calibration plots and reliability diagrams
- Error analysis and outlier detection
- Cross-validation visualization

Prerequisites:
- pandas, numpy, scikit-learn, matplotlib, seaborn
- Dataset with trained models for analysis
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, validation_curve,
    learning_curve, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, log_loss,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    # Calibration
    calibration_curve, brier_score_loss
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.calibration import CalibratedClassifierCV
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

# Create comprehensive datasets for demonstration
def create_classification_data():
    """Create classification dataset with different patterns"""
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with complex patterns
    y = np.zeros(n_samples)
    y[X[:, 0] + X[:, 1] > 0] = 1  # Linear boundary
    y[(X[:, 2]**2 + X[:, 3]**2) > 2] = 2  # Non-linear boundary
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = np.random.choice([0, 1, 2], size=len(noise_idx))
    
    return X, y.astype(int)

def create_regression_data():
    """Create regression dataset with different patterns"""
    n_samples = 1000
    n_features = 8
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with linear and non-linear components
    y = (2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] +  # Linear
         0.3 * X[:, 3] * X[:, 4] +  # Interaction
         np.sin(X[:, 5]) +  # Non-linear
         np.random.randn(n_samples) * 0.5)  # Noise
    
    return X, y

# Create datasets
X_class, y_class = create_classification_data()
X_reg, y_reg = create_regression_data()

print("Classification Dataset Shape:", X_class.shape)
print("Classification Classes:", np.unique(y_class))
print("Regression Dataset Shape:", X_reg.shape)

# Split datasets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train models for demonstration
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_class, y_train_class)

lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
lr_classifier.fit(X_train_class, y_train_class)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)

lr_regressor = LinearRegression()
lr_regressor.fit(X_train_reg, y_train_reg)

print("Models trained successfully!")

# ===============================================================================
# 1. CLASSIFICATION DIAGNOSTIC PLOTS
# ===============================================================================

print("\n" + "="*60)
print("1. CLASSIFICATION DIAGNOSTIC PLOTS")
print("="*60)

# ROC Curves
def plot_roc_curves(models, X_test, y_test, model_names):
    """Plot ROC curves for multiple models and classes"""
    plt.figure(figsize=(15, 5))
    
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        plt.subplot(1, len(models), idx + 1)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            # For SVM without probability
            y_proba = model.decision_function(X_test)
            if y_proba.ndim == 1:
                y_proba = np.column_stack([-y_proba, y_proba])
        
        # Plot ROC curve for each class
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(min(n_classes, len(colors))):
            if y_proba.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {name}')
        plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

# Precision-Recall Curves
def plot_precision_recall_curves(models, X_test, y_test, model_names):
    """Plot Precision-Recall curves for multiple models"""
    plt.figure(figsize=(15, 5))
    
    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        plt.subplot(1, len(models), idx + 1)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            continue  # Skip models without probability predictions
        
        # Plot PR curve for each class
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(min(n_classes, len(colors))):
            if y_proba.shape[1] > i:
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                avg_precision = average_precision_score(y_test_bin[:, i], y_proba[:, i])
                plt.plot(recall, precision, color=colors[i], lw=2,
                        label=f'Class {i} (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {name}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Confusion Matrix Heatmaps
def plot_confusion_matrices(models, X_test, y_test, model_names):
    """Plot confusion matrices for multiple models"""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   ax=axes[idx], cbar_kws={'label': 'Proportion'})
        axes[idx].set_title(f'Confusion Matrix - {name}')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()

# Classification Calibration Plots
def plot_calibration_curves(models, X_test, y_test, model_names):
    """Plot calibration curves to assess probability calibration"""
    plt.figure(figsize=(12, 8))
    
    # For binary classification, convert to binary problem
    y_binary = (y_test > 0).astype(int)
    
    for model, name in zip(models, model_names):
        if hasattr(model, 'predict_proba'):
            # Get probabilities for positive class
            if model.predict_proba(X_test).shape[1] == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # For multi-class, use max probability
                y_proba = np.max(model.predict_proba(X_test), axis=1)
                
            fraction_pos, mean_pred_value = calibration_curve(y_binary, y_proba, n_bins=10)
            
            plt.plot(mean_pred_value, fraction_pos, 's-', label=f'{name}', linewidth=2)
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Generate classification diagnostic plots
models_class = [rf_classifier, lr_classifier]
model_names_class = ['Random Forest', 'Logistic Regression']

print("1.1 ROC Curves")
plot_roc_curves(models_class, X_test_class, y_test_class, model_names_class)

print("1.2 Precision-Recall Curves")
plot_precision_recall_curves(models_class, X_test_class, y_test_class, model_names_class)

print("1.3 Confusion Matrices")
plot_confusion_matrices(models_class, X_test_class, y_test_class, model_names_class)

print("1.4 Calibration Curves")
plot_calibration_curves(models_class, X_test_class, y_test_class, model_names_class)

# Classification Report
print("1.5 Classification Reports")
print("-" * 40)
for model, name in zip(models_class, model_names_class):
    y_pred = model.predict(X_test_class)
    print(f"\n{name}:")
    print(classification_report(y_test_class, y_pred))

# ===============================================================================
# 2. REGRESSION DIAGNOSTIC PLOTS
# ===============================================================================

print("\n" + "="*60)
print("2. REGRESSION DIAGNOSTIC PLOTS")
print("="*60)

# Residual Plots
def plot_residuals(models, X_test, y_test, model_names):
    """Plot residual analysis for regression models"""
    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 10))
    if len(models) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Residuals vs Predicted
        axes[0, idx].scatter(y_pred, residuals, alpha=0.6)
        axes[0, idx].axhline(y=0, color='red', linestyle='--')
        axes[0, idx].set_xlabel('Predicted Values')
        axes[0, idx].set_ylabel('Residuals')
        axes[0, idx].set_title(f'Residuals vs Predicted - {name}')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Q-Q Plot (Normal Probability Plot)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, idx])
        axes[1, idx].set_title(f'Q-Q Plot - {name}')
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Prediction vs Actual Plots
def plot_prediction_vs_actual(models, X_test, y_test, model_names):
    """Plot predicted vs actual values"""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Scatter plot
        axes[idx].scatter(y_test, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        axes[idx].set_xlabel('Actual Values')
        axes[idx].set_ylabel('Predicted Values')
        axes[idx].set_title(f'{name}\nR² = {r2:.3f}, MSE = {mse:.3f}, MAE = {mae:.3f}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Distribution of Residuals
def plot_residual_distributions(models, X_test, y_test, model_names):
    """Plot distribution of residuals"""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Histogram with normal curve overlay
        axes[idx].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[idx].plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * 
                      np.exp(-0.5 * ((x - mu) / sigma)**2), 'r-', lw=2, label='Normal')
        
        axes[idx].set_xlabel('Residuals')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'Residual Distribution - {name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Generate regression diagnostic plots
models_reg = [rf_regressor, lr_regressor]
model_names_reg = ['Random Forest', 'Linear Regression']

print("2.1 Residual Analysis")
plot_residuals(models_reg, X_test_reg, y_test_reg, model_names_reg)

print("2.2 Prediction vs Actual")
plot_prediction_vs_actual(models_reg, X_test_reg, y_test_reg, model_names_reg)

print("2.3 Residual Distributions")
plot_residual_distributions(models_reg, X_test_reg, y_test_reg, model_names_reg)

# ===============================================================================
# 3. FEATURE IMPORTANCE VISUALIZATIONS
# ===============================================================================

print("\n" + "="*60)
print("3. FEATURE IMPORTANCE VISUALIZATIONS")
print("="*60)

# Feature Importance Plots
def plot_feature_importance(model, feature_names, title, top_n=10):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.title(f'Feature Importance - {title}')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
        
        # Print importance values
        print(f"\nTop {top_n} Features - {title}:")
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx]:<15}: {importances[idx]:.4f}")
    else:
        print(f"Model {title} does not have feature_importances_ attribute")

# Permutation Importance
def plot_permutation_importance(model, X_test, y_test, feature_names, title, top_n=10):
    """Plot permutation importance"""
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort by importance
    indices = np.argsort(perm_importance.importances_mean)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), perm_importance.importances_mean[indices])
    plt.errorbar(range(len(indices)), perm_importance.importances_mean[indices],
                yerr=perm_importance.importances_std[indices], fmt='none', color='red')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Permutation Importance - {title}')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # Print importance values
    print(f"\nTop {top_n} Features (Permutation) - {title}:")
    for i, idx in enumerate(indices):
        mean_imp = perm_importance.importances_mean[idx]
        std_imp = perm_importance.importances_std[idx]
        print(f"{i+1:2d}. {feature_names[idx]:<15}: {mean_imp:.4f} ± {std_imp:.4f}")

# Generate feature importance plots
feature_names_class = [f'feature_{i}' for i in range(X_class.shape[1])]
feature_names_reg = [f'feature_{i}' for i in range(X_reg.shape[1])]

print("3.1 Tree-based Feature Importance")
plot_feature_importance(rf_classifier, feature_names_class, 'Random Forest Classifier')
plot_feature_importance(rf_regressor, feature_names_reg, 'Random Forest Regressor')

print("3.2 Permutation Importance")
plot_permutation_importance(rf_classifier, X_test_class, y_test_class, 
                          feature_names_class, 'Random Forest Classifier')
plot_permutation_importance(lr_classifier, X_test_class, y_test_class, 
                          feature_names_class, 'Logistic Regression')

# ===============================================================================
# 4. LEARNING CURVES AND VALIDATION CURVES
# ===============================================================================

print("\n" + "="*60)
print("4. LEARNING CURVES AND VALIDATION CURVES")
print("="*60)

# Learning Curves
def plot_learning_curves(model, X, y, title, cv=5):
    """Plot learning curves to diagnose bias vs variance"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'Learning Curves - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Validation Curves
def plot_validation_curves(model, X, y, param_name, param_range, title, cv=5):
    """Plot validation curves for hyperparameter analysis"""
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Validation Curves - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Generate learning and validation curves
print("4.1 Learning Curves")
plot_learning_curves(rf_classifier, X_class, y_class, 'Random Forest Classifier')
plot_learning_curves(rf_regressor, X_reg, y_reg, 'Random Forest Regressor')

print("4.2 Validation Curves")
# For Random Forest - n_estimators
plot_validation_curves(
    RandomForestClassifier(random_state=42), 
    X_class, y_class,
    'n_estimators', [10, 50, 100, 200, 300, 500],
    'Random Forest Classifier (n_estimators)'
)

# ===============================================================================
# 5. ERROR ANALYSIS AND OUTLIER DETECTION
# ===============================================================================

print("\n" + "="*60)
print("5. ERROR ANALYSIS AND OUTLIER DETECTION")
print("="*60)

# Classification Error Analysis
def analyze_classification_errors(model, X_test, y_test, feature_names):
    """Analyze misclassified examples"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Find misclassified examples
    misclassified = X_test[y_test != y_pred]
    true_labels = y_test[y_test != y_pred]
    pred_labels = y_pred[y_test != y_pred]
    
    if y_proba is not None:
        pred_proba = y_proba[y_test != y_pred]
        confidence = np.max(pred_proba, axis=1)
    else:
        confidence = np.ones(len(misclassified))
    
    print(f"Misclassified examples: {len(misclassified)} out of {len(y_test)}")
    print(f"Error rate: {len(misclassified) / len(y_test):.3f}")
    
    # Analyze confidence of misclassified examples
    if y_proba is not None:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidence, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution of Misclassified Examples')
        plt.grid(True, alpha=0.3)
        
        # Compare with correctly classified
        correct_proba = y_proba[y_test == y_pred]
        correct_confidence = np.max(correct_proba, axis=1)
        
        plt.subplot(1, 2, 2)
        plt.hist(correct_confidence, bins=20, alpha=0.5, label='Correct', edgecolor='black')
        plt.hist(confidence, bins=20, alpha=0.5, label='Misclassified', edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence: Correct vs Misclassified')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Regression Error Analysis
def analyze_regression_errors(model, X_test, y_test, feature_names, top_errors=10):
    """Analyze largest prediction errors"""
    y_pred = model.predict(X_test)
    errors = np.abs(y_test - y_pred)
    
    # Find largest errors
    error_indices = np.argsort(errors)[::-1]
    
    print(f"Largest {top_errors} prediction errors:")
    print(f"{'Index':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 40)
    
    for i in range(min(top_errors, len(error_indices))):
        idx = error_indices[i]
        print(f"{idx:<8} {y_test.iloc[idx]:<10.3f} {y_pred[idx]:<10.3f} {errors[idx]:<10.3f}")
    
    # Plot error distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, errors, alpha=0.6)
    plt.xlabel('Predicted Values')
    plt.ylabel('Absolute Error')
    plt.title('Error vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Box plot of errors
    plt.subplot(1, 3, 3)
    plt.boxplot(errors)
    plt.ylabel('Absolute Error')
    plt.title('Error Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Outlier Detection
def detect_outliers(X, y, method='isolation_forest'):
    """Detect outliers in the dataset"""
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    
    if method == 'isolation_forest':
        detector = IsolationForest(contamination=0.1, random_state=42)
    elif method == 'elliptic_envelope':
        detector = EllipticEnvelope(contamination=0.1, random_state=42)
    else:
        raise ValueError("Method must be 'isolation_forest' or 'elliptic_envelope'")
    
    outliers = detector.fit_predict(X)
    outlier_indices = np.where(outliers == -1)[0]
    
    print(f"Detected {len(outlier_indices)} outliers using {method}")
    
    # Visualize outliers (first two dimensions)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[outliers == 1, 0], X[outliers == 1, 1], 
               c='blue', alpha=0.6, label='Normal')
    plt.scatter(X[outliers == -1, 0], X[outliers == -1, 1], 
               c='red', alpha=0.8, label='Outliers')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('Outlier Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Target distribution for outliers vs normal
    plt.subplot(1, 2, 2)
    normal_targets = y[outliers == 1]
    outlier_targets = y[outliers == -1]
    
    plt.hist(normal_targets, bins=20, alpha=0.5, label='Normal', density=True)
    if len(outlier_targets) > 0:
        plt.hist(outlier_targets, bins=20, alpha=0.5, label='Outliers', density=True)
    plt.xlabel('Target Value')
    plt.ylabel('Density')
    plt.title('Target Distribution: Normal vs Outliers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return outlier_indices

# Generate error analysis
print("5.1 Classification Error Analysis")
analyze_classification_errors(rf_classifier, X_test_class, y_test_class, feature_names_class)

print("5.2 Regression Error Analysis")
analyze_regression_errors(rf_regressor, X_test_reg, y_test_reg, feature_names_reg)

print("5.3 Outlier Detection")
outliers_class = detect_outliers(X_class, y_class, 'isolation_forest')
outliers_reg = detect_outliers(X_reg, y_reg, 'elliptic_envelope')

# ===============================================================================
# 6. CROSS-VALIDATION VISUALIZATION
# ===============================================================================

print("\n" + "="*60)
print("6. CROSS-VALIDATION VISUALIZATION")
print("="*60)

def plot_cv_scores(model, X, y, cv_strategy, title):
    """Visualize cross-validation scores"""
    cv_scores = cross_val_score(model, X, y, cv=cv_strategy)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(cv_scores) + 1), cv_scores)
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {cv_scores.mean():.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(f'Cross-Validation Scores - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(cv_scores)
    plt.ylabel('Score')
    plt.title(f'CV Score Distribution - {title}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Cross-validation visualization
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plot_cv_scores(rf_classifier, X_class, y_class, cv_strategy, 'Random Forest Classifier')

# ===============================================================================
# 7. DIAGNOSTIC SUMMARY AND INTERPRETATION GUIDE
# ===============================================================================

print("\n" + "="*60)
print("7. DIAGNOSTIC PLOTS INTERPRETATION GUIDE")
print("="*60)

interpretation_guide = {
    'ROC Curves': [
        'AUC > 0.9: Excellent discrimination',
        'AUC 0.8-0.9: Good discrimination', 
        'AUC 0.7-0.8: Fair discrimination',
        'AUC 0.6-0.7: Poor discrimination',
        'AUC < 0.6: No discrimination'
    ],
    'Precision-Recall': [
        'High precision + high recall: Ideal',
        'High precision + low recall: Conservative model',
        'Low precision + high recall: Liberal model',
        'Area under PR curve for imbalanced data'
    ],
    'Confusion Matrix': [
        'Diagonal elements: Correct predictions',
        'Off-diagonal: Misclassifications',
        'Row-wise normalization shows recall',
        'Column-wise shows precision'
    ],
    'Calibration': [
        'Close to diagonal: Well-calibrated',
        'Above diagonal: Over-confident',
        'Below diagonal: Under-confident',
        'Use calibration for probability interpretation'
    ],
    'Residual Plots': [
        'Random scatter: Good model',
        'Curved pattern: Non-linear relationship missed',
        'Funnel shape: Heteroscedasticity',
        'Q-Q plot should be linear for normal residuals'
    ],
    'Learning Curves': [
        'High bias: Both curves plateau at low performance',
        'High variance: Large gap between train/validation',
        'Good fit: Curves converge at high performance',
        'More data helps with high variance'
    ]
}

print("DIAGNOSTIC PLOT INTERPRETATION GUIDE:")
print("=" * 40)

for plot_type, guidelines in interpretation_guide.items():
    print(f"\n{plot_type}:")
    for guideline in guidelines:
        print(f"  • {guideline}")

# Model selection recommendations
print(f"\n" + "="*40)
print("MODEL SELECTION RECOMMENDATIONS")
print("="*40)

recommendations = [
    "1. Check multiple metrics, not just accuracy",
    "2. Examine confusion matrix for class-specific performance",
    "3. Use ROC for balanced data, PR for imbalanced",
    "4. Validate calibration if using probabilities",
    "5. Analyze residuals for regression models",
    "6. Check for overfitting with learning curves",
    "7. Consider error analysis for model improvement",
    "8. Use cross-validation for robust evaluation",
    "9. Detect and handle outliers appropriately",
    "10. Choose metrics aligned with business objectives"
]

for recommendation in recommendations:
    print(recommendation)

print(f"\nDiagnostic plot analysis complete!")
print(f"Use these visualizations to understand model behavior and guide improvements.")
