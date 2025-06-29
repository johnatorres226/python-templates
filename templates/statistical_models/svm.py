# ==============================================================================
# SUPPORT VECTOR MACHINE (SVM) TEMPLATE
# ==============================================================================
# Purpose: Classification and regression with Support Vector Machines
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

print("Support Vector Machine (SVM) Analysis")
print("="*50)

# ================================
# 1. DATA PREPARATION
# ================================
print("1. Data Preparation")
print("-" * 20)

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Replace 'target_column' with your target variable
target_column = 'target_column'
feature_columns = [col for col in df.columns if col != target_column]

# Handle missing values
df = df.dropna()

# Determine if this is classification or regression
is_classification = df[target_column].dtype == 'object' or df[target_column].nunique() < 10

print(f"Problem type: {'Classification' if is_classification else 'Regression'}")
print(f"Target variable: {target_column}")

# For classification, encode categorical target
if is_classification and df[target_column].dtype == 'object':
    le_target = LabelEncoder()
    df[target_column] = le_target.fit_transform(df[target_column])
    print(f"Target classes: {le_target.classes_}")

if is_classification:
    print(f"Class distribution:")
    print(df[target_column].value_counts())

# ================================
# 2. FEATURE PREPARATION
# ================================
print("\n2. Feature Preparation")
print("-" * 20)

# Separate numeric and categorical features
numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df[feature_columns].select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Handle categorical variables (one-hot encoding for SVM)
df_processed = df.copy()
for col in categorical_features:
    # One-hot encode categorical variables
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df_processed = pd.concat([df_processed, dummies], axis=1)
    df_processed = df_processed.drop(col, axis=1)

# Update feature list
feature_columns = [col for col in df_processed.columns if col != target_column]

# Prepare final datasets
X = df_processed[feature_columns]
y = df_processed[target_column]

print(f"Final feature count: {X.shape[1]}")

# ================================
# 3. TRAIN-TEST SPLIT
# ================================
print("\n3. Train-Test Split")
print("-" * 20)

# Split data
if is_classification:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ================================
# 4. FEATURE SCALING (CRITICAL FOR SVM)
# ================================
print("\n4. Feature Scaling")
print("-" * 20)

# Standardize features (essential for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized (mean=0, std=1)")
print("Note: Feature scaling is critical for SVM performance!")

# ================================
# 5. FEATURE SELECTION (Optional)
# ================================
print("\n5. Feature Selection")
print("-" * 20)

# Select best features
k_best = min(20, X.shape[1])  # Select top 20 features or all if less
if is_classification:
    selector = SelectKBest(score_func=f_classif, k=k_best)
else:
    selector = SelectKBest(score_func=f_regression, k=k_best)

X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} best features")

# ================================
# 6. MODEL TRAINING
# ================================
print("\n6. Model Training")
print("-" * 20)

# Create different SVM models
if is_classification:
    models = {
        'Linear_SVM': SVC(kernel='linear', random_state=42, probability=True),
        'RBF_SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Polynomial_SVM': SVC(kernel='poly', degree=3, random_state=42, probability=True),
        'Sigmoid_SVM': SVC(kernel='sigmoid', random_state=42, probability=True)
    }
else:
    models = {
        'Linear_SVR': SVR(kernel='linear'),
        'RBF_SVR': SVR(kernel='rbf'),
        'Polynomial_SVR': SVR(kernel='poly', degree=3),
        'Sigmoid_SVR': SVR(kernel='sigmoid')
    }

model_results = {}

for name, model in models.items():
    try:
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            model_results[name] = {
                'model': model,
                'y_pred': y_pred,
                'accuracy': accuracy,
                'score': accuracy
            }
            print(f"{name}: Accuracy = {accuracy:.3f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            model_results[name] = {
                'model': model,
                'y_pred': y_pred,
                'mse': mse,
                'r2': r2,
                'score': r2
            }
            print(f"{name}: R² = {r2:.3f}, MSE = {mse:.3f}")
            
    except Exception as e:
        print(f"{name}: Error - {str(e)}")

# ================================
# 7. BEST MODEL SELECTION
# ================================
print("\n7. Best Model Selection")
print("-" * 20)

# Select best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['score'])
best_model = model_results[best_model_name]['model']
best_y_pred = model_results[best_model_name]['y_pred']

print(f"Best model: {best_model_name}")
print(f"Best score: {model_results[best_model_name]['score']:.3f}")

# ================================
# 8. MODEL EVALUATION
# ================================
print("\n8. Model Evaluation")
print("-" * 20)

if is_classification:
    print("Classification Report:")
    print(classification_report(y_test, best_y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, best_y_pred)
    print(cm)
    
else:
    mse = mean_squared_error(y_test, best_y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, best_y_pred)
    r2 = r2_score(y_test, best_y_pred)
    
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R-squared: {r2:.3f}")

# ================================
# 9. HYPERPARAMETER TUNING
# ================================
print("\n9. Hyperparameter Tuning")
print("-" * 20)

# Define parameter grid for best kernel
if 'RBF' in best_model_name:
    if is_classification:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        tuning_model = SVC(kernel='rbf', random_state=42)
        scoring = 'accuracy'
    else:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        tuning_model = SVR(kernel='rbf')
        scoring = 'r2'
        
elif 'Linear' in best_model_name:
    if is_classification:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        tuning_model = SVC(kernel='linear', random_state=42)
        scoring = 'accuracy'
    else:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        tuning_model = SVR(kernel='linear')
        scoring = 'r2'
else:
    # Default grid for polynomial
    if is_classification:
        param_grid = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4]
        }
        tuning_model = SVC(kernel='poly', random_state=42)
        scoring = 'accuracy'
    else:
        param_grid = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'epsilon': [0.01, 0.1, 0.2]
        }
        tuning_model = SVR(kernel='poly')
        scoring = 'r2'

# Perform grid search
print("Performing hyperparameter tuning...")
grid_search = GridSearchCV(tuning_model, param_grid, cv=5, 
                          scoring=scoring, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# ================================
# 10. FINAL MODEL EVALUATION
# ================================
print("\n10. Final Model Evaluation")
print("-" * 20)

# Train final model with best parameters
final_model = grid_search.best_estimator_
final_y_pred = final_model.predict(X_test_scaled)

if is_classification:
    final_accuracy = accuracy_score(y_test, final_y_pred)
    print(f"Final model accuracy: {final_accuracy:.3f}")
    
    # Get prediction probabilities if available
    if hasattr(final_model, 'predict_proba'):
        y_pred_proba = final_model.predict_proba(X_test_scaled)
else:
    final_r2 = r2_score(y_test, final_y_pred)
    final_mse = mean_squared_error(y_test, final_y_pred)
    print(f"Final model R²: {final_r2:.3f}")
    print(f"Final model MSE: {final_mse:.3f}")

# ================================
# 11. CROSS-VALIDATION
# ================================
print("\n11. Cross-Validation")
print("-" * 20)

# Perform cross-validation with final model
cv_scores = cross_val_score(final_model, X_train_scaled, y_train, 
                           cv=5, scoring=scoring)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ================================
# 12. VISUALIZATIONS
# ================================
print("\n12. Creating Visualizations")
print("-" * 20)

# Set up the plotting area
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Support Vector Machine Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Model Comparison
model_names = list(model_results.keys())
scores = [model_results[name]['score'] for name in model_names]
axes[0, 0].bar(model_names, scores, color=['blue', 'orange', 'green', 'red'])
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('SVM Kernel Comparison')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Feature Selection Scores
if len(selected_features) <= 20:  # Only if manageable number of features
    feature_scores = selector.scores_[selector.get_support()]
    axes[0, 1].barh(range(len(selected_features)), feature_scores)
    axes[0, 1].set_yticks(range(len(selected_features)))
    axes[0, 1].set_yticklabels(selected_features)
    axes[0, 1].set_xlabel('Feature Score')
    axes[0, 1].set_title('Selected Feature Importance')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
else:
    axes[0, 1].text(0.5, 0.5, f'Too many features\nto display\n({len(selected_features)} features)', 
                   ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Feature Selection Summary')

# Plot 3: Confusion Matrix (Classification) or Actual vs Predicted (Regression)
if is_classification:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix (Best Model)')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
else:
    axes[1, 0].scatter(y_test, final_y_pred, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Actual vs Predicted Values')
    axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Residuals (Regression) or Prediction Distribution (Classification)
if is_classification:
    axes[1, 1].hist(y_test, bins=20, alpha=0.7, label='Actual', color='blue')
    axes[1, 1].hist(final_y_pred, bins=20, alpha=0.7, label='Predicted', color='orange')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Actual vs Predicted Distribution')
    axes[1, 1].legend()
else:
    residuals = y_test - final_y_pred
    axes[1, 1].scatter(final_y_pred, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# 13. MODEL INTERPRETATION
# ================================
print("\n13. Model Interpretation")
print("-" * 20)

print("Key Insights:")
if is_classification:
    print(f"• Model Accuracy: {final_accuracy:.3f}")
else:
    print(f"• Model R²: {final_r2:.3f}")

print(f"• Best Kernel: {best_model_name}")
print(f"• Number of support vectors: {final_model.n_support_}")

if hasattr(final_model, 'support_'):
    support_vector_ratio = len(final_model.support_) / len(X_train_scaled)
    print(f"• Support vector ratio: {support_vector_ratio:.3f}")

print("\nSVM Characteristics:")
print("• Effective in high-dimensional spaces")
print("• Memory efficient (uses subset of training points)")
print("• Versatile with different kernel functions")
print("• Requires feature scaling")
print("• Performance depends heavily on hyperparameter tuning")

# ================================
# 14. MAKING PREDICTIONS ON NEW DATA
# ================================
print("\n14. Making Predictions on New Data")
print("-" * 20)

# Example of how to use the model for new predictions
new_data_example = X_test.iloc[:5].copy()

# Scale the new data
new_data_scaled = scaler.transform(new_data_example)

# Make predictions
new_predictions = final_model.predict(new_data_scaled)

print("Example predictions for new data:")
for i in range(len(new_predictions)):
    print(f"Sample {i+1}: Prediction = {new_predictions[i]:.3f}")

print("\nSVM analysis completed!")
print("="*50)

# ================================
# SUMMARY & RECOMMENDATIONS
# ================================
print("\nSUMMARY & RECOMMENDATIONS:")
print("• SVMs are powerful for both classification and regression")
print("• Feature scaling is absolutely critical for SVM performance")
print("• RBF kernel often works well for non-linear relationships")
print("• Linear kernel is faster and works well for linearly separable data")
print("• Hyperparameter tuning (C, gamma) significantly impacts performance")
print("• SVMs can be slow on large datasets")

print("\nRemember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update 'target_column' and feature column names")
print("3. Always scale features before training SVM")
print("4. Tune hyperparameters for optimal performance")
print("5. Consider kernel choice based on data characteristics")
