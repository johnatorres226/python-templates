# ==============================================================================
# DECISION TREES TEMPLATE
# ==============================================================================
# Purpose: Classification and regression with decision trees
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

print("Decision Tree Analysis")
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

# Handle categorical variables
df_processed = df.copy()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

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
# 4. MODEL TRAINING
# ================================
print("\n4. Model Training")
print("-" * 20)

# Create different decision tree models
if is_classification:
    models = {
        'Default': DecisionTreeClassifier(random_state=42),
        'Max_Depth_5': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Min_Samples_Split_20': DecisionTreeClassifier(min_samples_split=20, random_state=42),
        'Pruned': DecisionTreeClassifier(max_depth=5, min_samples_split=20, 
                                       min_samples_leaf=5, random_state=42)
    }
else:
    models = {
        'Default': DecisionTreeRegressor(random_state=42),
        'Max_Depth_5': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Min_Samples_Split_20': DecisionTreeRegressor(min_samples_split=20, random_state=42),
        'Pruned': DecisionTreeRegressor(max_depth=5, min_samples_split=20, 
                                      min_samples_leaf=5, random_state=42)
    }

model_results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
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

# ================================
# 5. BEST MODEL SELECTION
# ================================
print("\n5. Best Model Selection")
print("-" * 20)

# Select best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['score'])
best_model = model_results[best_model_name]['model']
best_y_pred = model_results[best_model_name]['y_pred']

print(f"Best model: {best_model_name}")

# ================================
# 6. MODEL EVALUATION
# ================================
print("\n6. Model Evaluation")
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
# 7. FEATURE IMPORTANCE
# ================================
print("\n7. Feature Importance")
print("-" * 20)

# Get feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 most important features:")
print(importance_df.head(10))

# ================================
# 8. HYPERPARAMETER TUNING
# ================================
print("\n8. Hyperparameter Tuning")
print("-" * 20)

# Define parameter grid
if is_classification:
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    tuning_model = DecisionTreeClassifier(random_state=42)
    scoring = 'accuracy'
else:
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['squared_error', 'absolute_error']
    }
    tuning_model = DecisionTreeRegressor(random_state=42)
    scoring = 'r2'

# Perform grid search
grid_search = GridSearchCV(tuning_model, param_grid, cv=5, 
                          scoring=scoring, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# ================================
# 9. CROSS-VALIDATION
# ================================
print("\n9. Cross-Validation")
print("-" * 20)

# Perform cross-validation with best model
cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, 
                           cv=5, scoring=scoring)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ================================
# 10. VISUALIZATIONS
# ================================
print("\n10. Creating Visualizations")
print("-" * 20)

# Set up the plotting area
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Decision Tree Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Feature Importance
top_features = importance_df.head(15)
axes[0, 0].barh(range(len(top_features)), top_features['Importance'])
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features['Feature'])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Feature Importance')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Plot 2: Model Comparison
model_names = list(model_results.keys())
scores = [model_results[name]['score'] for name in model_names]
axes[0, 1].bar(model_names, scores, color=['blue', 'orange', 'green', 'red'])
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Model Comparison')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Confusion Matrix (Classification) or Actual vs Predicted (Regression)
if is_classification:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix (Best Model)')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
else:
    axes[1, 0].scatter(y_test, best_y_pred, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Actual vs Predicted Values')
    axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Tree Visualization (simplified)
# Show a small version of the tree
simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42) if is_classification else DecisionTreeRegressor(max_depth=3, random_state=42)
simple_tree.fit(X_train, y_train)

plot_tree(simple_tree, 
         feature_names=X.columns[:10],  # Show only first 10 features for readability
         filled=True, 
         rounded=True, 
         fontsize=8,
         ax=axes[1, 1])
axes[1, 1].set_title('Decision Tree Structure (Depth=3)')

plt.tight_layout()
plt.show()

# ================================
# 11. TREE RULES EXPORT
# ================================
print("\n11. Tree Rules (Text Format)")
print("-" * 20)

# Export tree rules in text format (for a small tree)
tree_rules = export_text(simple_tree, 
                        feature_names=list(X.columns))
print("Tree rules (simplified tree with max_depth=3):")
print(tree_rules[:1000] + "..." if len(tree_rules) > 1000 else tree_rules)

# ================================
# 12. MODEL INTERPRETATION
# ================================
print("\n12. Model Interpretation")
print("-" * 20)

print("Key Insights:")
if is_classification:
    print(f"• Model Accuracy: {model_results[best_model_name]['accuracy']:.3f}")
else:
    print(f"• Model R²: {model_results[best_model_name]['r2']:.3f}")

print(f"• Model Type: {best_model_name}")
print(f"• Most important feature: {importance_df.iloc[0]['Feature']}")
print(f"• Tree depth: {best_model.get_depth()}")
print(f"• Number of leaves: {best_model.get_n_leaves()}")

print("\nDecision Tree Characteristics:")
print("• Non-parametric model - no assumptions about data distribution")
print("• Handles both numerical and categorical features")
print("• Provides feature importance rankings")
print("• Interpretable rules and decision paths")
print("• Prone to overfitting without proper pruning")

# ================================
# 13. MAKING PREDICTIONS ON NEW DATA
# ================================
print("\n13. Making Predictions on New Data")
print("-" * 20)

# Example of how to use the model for new predictions
new_data_example = X_test.iloc[:5].copy()

# Make predictions
new_predictions = best_model.predict(new_data_example)

print("Example predictions for new data:")
for i in range(len(new_predictions)):
    print(f"Sample {i+1}: Prediction = {new_predictions[i]:.3f}")

print("\nDecision Tree analysis completed!")
print("="*50)

# ================================
# SUMMARY & RECOMMENDATIONS
# ================================
print("\nSUMMARY & RECOMMENDATIONS:")
print("• Decision trees are easy to interpret and visualize")
print("• No need for feature scaling or normalization")
print("• Handle missing values naturally")
print("• Can capture non-linear relationships")
print("• Use pruning to prevent overfitting")
print("• Consider ensemble methods (Random Forest, Gradient Boosting) for better performance")

print("\nRemember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update 'target_column' and feature column names")
print("3. Choose appropriate max_depth to prevent overfitting")
print("4. Validate model performance with cross-validation")
print("5. Consider feature engineering for better results")
