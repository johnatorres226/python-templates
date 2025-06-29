# ==============================================================================
# BASIC NEURAL NETWORK TEMPLATE
# ==============================================================================
# Purpose: Classification and regression with basic neural networks (MLP)
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

print("Basic Neural Network (MLP) Analysis")
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

# Handle categorical variables (one-hot encoding for neural networks)
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
# 4. FEATURE SCALING (CRITICAL FOR NEURAL NETWORKS)
# ================================
print("\n4. Feature Scaling")
print("-" * 20)

# Standardize features (essential for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized (mean=0, std=1)")
print("Note: Feature scaling is critical for neural network performance!")

# ================================
# 5. FEATURE SELECTION (Optional)
# ================================
print("\n5. Feature Selection")
print("-" * 20)

# Select best features to reduce complexity
k_best = min(50, X.shape[1])  # Select top 50 features or all if less
if is_classification:
    selector = SelectKBest(score_func=f_classif, k=k_best)
else:
    selector = SelectKBest(score_func=f_regression, k=k_best)

X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} best features for neural network")

# ================================
# 6. MODEL ARCHITECTURE DESIGN
# ================================
print("\n6. Neural Network Architecture Design")
print("-" * 20)

# Design different neural network architectures
input_size = X_train_scaled.shape[1]
output_size = len(np.unique(y)) if is_classification else 1

print(f"Input layer size: {input_size}")
print(f"Output layer size: {output_size}")

# Define different architectures
architectures = {
    'Small': (50,),  # Single hidden layer with 50 neurons
    'Medium': (100, 50),  # Two hidden layers
    'Large': (100, 100, 50),  # Three hidden layers
    'Deep': (200, 100, 50, 25)  # Four hidden layers
}

print("Testing architectures:")
for name, layers in architectures.items():
    print(f"  {name}: {layers}")

# ================================
# 7. MODEL TRAINING
# ================================
print("\n7. Model Training")
print("-" * 20)

model_results = {}

for name, hidden_layers in architectures.items():
    try:
        if is_classification:
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        
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
                'score': accuracy,
                'n_iter': model.n_iter_,
                'loss': model.loss_curve_[-1] if hasattr(model, 'loss_curve_') else None
            }
            print(f"{name}: Accuracy = {accuracy:.3f}, Iterations = {model.n_iter_}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            model_results[name] = {
                'model': model,
                'y_pred': y_pred,
                'mse': mse,
                'r2': r2,
                'score': r2,
                'n_iter': model.n_iter_,
                'loss': model.loss_curve_[-1] if hasattr(model, 'loss_curve_') else None
            }
            print(f"{name}: R² = {r2:.3f}, MSE = {mse:.3f}, Iterations = {model.n_iter_}")
            
    except Exception as e:
        print(f"{name}: Error - {str(e)}")

# ================================
# 8. BEST MODEL SELECTION
# ================================
print("\n8. Best Model Selection")
print("-" * 20)

# Select best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['score'])
best_model = model_results[best_model_name]['model']
best_y_pred = model_results[best_model_name]['y_pred']

print(f"Best model: {best_model_name}")
print(f"Best architecture: {architectures[best_model_name]}")
print(f"Best score: {model_results[best_model_name]['score']:.3f}")

# ================================
# 9. MODEL EVALUATION
# ================================
print("\n9. Model Evaluation")
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
# 10. HYPERPARAMETER TUNING
# ================================
print("\n10. Hyperparameter Tuning")
print("-" * 20)

# Define parameter grid for the best architecture
best_architecture = architectures[best_model_name]

if is_classification:
    param_grid = {
        'hidden_layer_sizes': [best_architecture, (best_architecture[0] * 2,), 
                              tuple([int(x * 1.5) for x in best_architecture])],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'activation': ['relu', 'tanh']
    }
    tuning_model = MLPClassifier(max_iter=500, random_state=42, early_stopping=True)
    scoring = 'accuracy'
else:
    param_grid = {
        'hidden_layer_sizes': [best_architecture, (best_architecture[0] * 2,), 
                              tuple([int(x * 1.5) for x in best_architecture])],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'activation': ['relu', 'tanh']
    }
    tuning_model = MLPRegressor(max_iter=500, random_state=42, early_stopping=True)
    scoring = 'r2'

# Perform grid search (reduced CV for speed)
print("Performing hyperparameter tuning...")
grid_search = GridSearchCV(tuning_model, param_grid, cv=3, 
                          scoring=scoring, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# ================================
# 11. FINAL MODEL EVALUATION
# ================================
print("\n11. Final Model Evaluation")
print("-" * 20)

# Train final model with best parameters
final_model = grid_search.best_estimator_
final_y_pred = final_model.predict(X_test_scaled)

if is_classification:
    final_accuracy = accuracy_score(y_test, final_y_pred)
    print(f"Final model accuracy: {final_accuracy:.3f}")
    print(f"Training iterations: {final_model.n_iter_}")
else:
    final_r2 = r2_score(y_test, final_y_pred)
    final_mse = mean_squared_error(y_test, final_y_pred)
    print(f"Final model R²: {final_r2:.3f}")
    print(f"Final model MSE: {final_mse:.3f}")
    print(f"Training iterations: {final_model.n_iter_}")

# ================================
# 12. CROSS-VALIDATION
# ================================
print("\n12. Cross-Validation")
print("-" * 20)

# Perform cross-validation with final model
cv_scores = cross_val_score(final_model, X_train_scaled, y_train, 
                           cv=3, scoring=scoring)  # Reduced CV for speed

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ================================
# 13. VISUALIZATIONS
# ================================
print("\n13. Creating Visualizations")
print("-" * 20)

# Set up the plotting area
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Neural Network Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Model Architecture Comparison
model_names = list(model_results.keys())
scores = [model_results[name]['score'] for name in model_names]
colors = ['blue', 'orange', 'green', 'red'][:len(model_names)]
axes[0, 0].bar(model_names, scores, color=colors)
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Architecture Comparison')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Training Iterations
iterations = [model_results[name]['n_iter'] for name in model_names]
axes[0, 1].bar(model_names, iterations, color=colors)
axes[0, 1].set_ylabel('Training Iterations')
axes[0, 1].set_title('Training Convergence')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Learning Curve (if available)
if hasattr(final_model, 'loss_curve_'):
    axes[0, 2].plot(final_model.loss_curve_, linewidth=2)
    axes[0, 2].set_xlabel('Iterations')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Learning Curve (Final Model)')
    axes[0, 2].grid(True, alpha=0.3)
else:
    axes[0, 2].text(0.5, 0.5, 'Learning curve\nnot available', 
                   ha='center', va='center', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title('Learning Curve')

# Plot 4: Confusion Matrix (Classification) or Actual vs Predicted (Regression)
if is_classification:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix (Final Model)')
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

# Plot 5: Feature Importance (if available with selected features)
if len(selected_features) <= 20:  # Only if manageable number of features
    feature_scores = selector.scores_[selector.get_support()]
    top_indices = np.argsort(feature_scores)[-10:]  # Top 10 features
    
    axes[1, 1].barh(range(len(top_indices)), feature_scores[top_indices])
    axes[1, 1].set_yticks(range(len(top_indices)))
    axes[1, 1].set_yticklabels([selected_features[i] for i in top_indices])
    axes[1, 1].set_xlabel('Feature Score')
    axes[1, 1].set_title('Top 10 Feature Importance')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
else:
    axes[1, 1].text(0.5, 0.5, f'Too many features\nto display\n({len(selected_features)} features)', 
                   ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Feature Importance')

# Plot 6: Residuals (Regression) or Prediction Distribution (Classification)
if is_classification:
    axes[1, 2].hist(y_test, bins=20, alpha=0.7, label='Actual', color='blue')
    axes[1, 2].hist(final_y_pred, bins=20, alpha=0.7, label='Predicted', color='orange')
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Actual vs Predicted Distribution')
    axes[1, 2].legend()
else:
    residuals = y_test - final_y_pred
    axes[1, 2].scatter(final_y_pred, residuals, alpha=0.6)
    axes[1, 2].axhline(y=0, color='red', linestyle='--')
    axes[1, 2].set_xlabel('Predicted Values')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].set_title('Residual Plot')
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# 14. MODEL INTERPRETATION
# ================================
print("\n14. Model Interpretation")
print("-" * 20)

print("Neural Network Architecture:")
print(f"• Best architecture: {architectures[best_model_name]}")
print(f"• Total parameters: ~{sum([a*b for a, b in zip([input_size] + list(architectures[best_model_name]), list(architectures[best_model_name]) + [output_size])])}")
print(f"• Training iterations: {final_model.n_iter_}")
print(f"• Activation function: {final_model.activation}")
print(f"• Solver: {final_model.solver}")

print("\nModel Performance:")
if is_classification:
    print(f"• Final Accuracy: {final_accuracy:.3f}")
else:
    print(f"• Final R²: {final_r2:.3f}")

print("\nNeural Network Characteristics:")
print("• Universal function approximator")
print("• Can capture complex non-linear patterns")
print("• Requires large amounts of data")
print("• Sensitive to feature scaling")
print("• Black box model (limited interpretability)")

# ================================
# 15. MAKING PREDICTIONS ON NEW DATA
# ================================
print("\n15. Making Predictions on New Data")
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

# Get prediction probabilities for classification
if is_classification and hasattr(final_model, 'predict_proba'):
    new_probabilities = final_model.predict_proba(new_data_scaled)
    print("\nPrediction probabilities:")
    for i in range(len(new_probabilities)):
        max_prob = new_probabilities[i].max()
        print(f"Sample {i+1}: Max probability = {max_prob:.3f}")

print("\nNeural Network analysis completed!")
print("="*50)

# ================================
# SUMMARY & RECOMMENDATIONS
# ================================
print("\nSUMMARY & RECOMMENDATIONS:")
print("• Neural networks are powerful for complex pattern recognition")
print("• Feature scaling is absolutely critical")
print("• Regularization (alpha) helps prevent overfitting")
print("• Early stopping prevents overtraining")
print("• Architecture design affects model complexity and performance")
print("• Consider ensemble methods for better performance")

print("\nWhen to use Neural Networks:")
print("• Large datasets with complex patterns")
print("• Non-linear relationships in data")
print("• Image, text, or sequential data")
print("• When interpretability is not critical")

print("\nRemember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update 'target_column' and feature column names")
print("3. Always scale features before training")
print("4. Use early stopping to prevent overfitting")
print("5. Consider simpler models first for interpretability")
