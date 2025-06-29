# ==============================================================================
# LOGISTIC REGRESSION TEMPLATE
# ==============================================================================
# Purpose: Binary and multiclass classification with logistic regression
# Replace 'your_data.csv' with your dataset
# Update column names to match your data
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           auc, precision_recall_curve, accuracy_score, 
                           precision_score, recall_score, f1_score)
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')

# Load your data
# Replace 'your_data.csv' with your actual dataset
df = pd.read_csv('your_data.csv')

print("Logistic Regression Analysis")
print("="*50)

# ================================
# 1. DATA PREPARATION
# ================================
print("1. Data Preparation")
print("-" * 20)

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Replace 'target_column' with your binary target variable (0/1 or categorical)
target_column = 'target_column'
feature_columns = [col for col in df.columns if col != target_column]

# Handle missing values (simple approach)
df = df.dropna()

# Encode categorical target if needed
le = LabelEncoder()
if df[target_column].dtype == 'object':
    df[target_column] = le.fit_transform(df[target_column])
    print(f"Target classes: {le.classes_}")

# Check target distribution
print(f"\nTarget distribution:")
print(df[target_column].value_counts())
print(f"Class proportions:")
print(df[target_column].value_counts(normalize=True))

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

# Handle categorical variables (one-hot encoding)
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

# Split data (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ================================
# 4. FEATURE SCALING
# ================================
print("\n4. Feature Scaling")
print("-" * 20)

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized (mean=0, std=1)")

# ================================
# 5. FEATURE SELECTION (Optional)
# ================================
print("\n5. Feature Selection")
print("-" * 20)

# Select top features using chi-square test
k_best = min(10, X.shape[1])  # Select top 10 features or all if less
selector = SelectKBest(score_func=chi2, k=k_best)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} best features:")
for feature in selected_features[:5]:  # Show top 5
    print(f"  - {feature}")

# ================================
# 6. MODEL TRAINING
# ================================
print("\n6. Model Training")
print("-" * 20)

# Train multiple logistic regression models
models = {
    'Basic': LogisticRegression(random_state=42),
    'L1_Regularized': LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
    'L2_Regularized': LogisticRegression(penalty='l2', C=0.1, random_state=42),
    'Elastic_Net': LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', 
                                     max_iter=1000, random_state=42)
}

model_results = {}

for name, model in models.items():
    try:
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        model_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"{name}: Accuracy = {model_results[name]['accuracy']:.3f}")
        
    except Exception as e:
        print(f"{name}: Error - {str(e)}")

# ================================
# 7. MODEL EVALUATION
# ================================
print("\n7. Model Evaluation")
print("-" * 20)

# Use best performing model (by accuracy)
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
best_model = model_results[best_model_name]['model']
best_y_pred = model_results[best_model_name]['y_pred']
best_y_pred_proba = model_results[best_model_name]['y_pred_proba']

print(f"Best model: {best_model_name}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, best_y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_y_pred)
print(cm)

# ================================
# 8. COEFFICIENT ANALYSIS
# ================================
print("\n8. Feature Importance (Coefficients)")
print("-" * 20)

# Get feature coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_model.coef_[0],
    'Abs_Coefficient': np.abs(best_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("Top 10 most important features:")
print(coefficients.head(10)[['Feature', 'Coefficient']])

# ================================
# 9. CROSS-VALIDATION
# ================================
print("\n9. Cross-Validation")
print("-" * 20)

# Perform stratified k-fold cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                           scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ================================
# 10. VISUALIZATIONS
# ================================
print("\n10. Creating Visualizations")
print("-" * 20)

# Set up the plotting area
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Logistic Regression Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Plot 2: ROC Curve
if len(np.unique(y_test)) == 2:  # Binary classification
    fpr, tpr, _ = roc_curve(y_test, best_y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC Curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Importance
top_features = coefficients.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['Coefficient'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['Feature'])
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 10 Feature Coefficients')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Prediction Probability Distribution
if len(np.unique(y_test)) == 2:
    axes[1, 1].hist(best_y_pred_proba[y_test == 0], bins=20, alpha=0.7, 
                   label='Class 0', color='red')
    axes[1, 1].hist(best_y_pred_proba[y_test == 1], bins=20, alpha=0.7, 
                   label='Class 1', color='blue')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# 11. MODEL INTERPRETATION
# ================================
print("\n11. Model Interpretation")
print("-" * 20)

print("Key Insights:")
print(f"• Model Accuracy: {model_results[best_model_name]['accuracy']:.3f}")
print(f"• Model Type: {best_model_name}")

if len(np.unique(y_test)) == 2:
    print(f"• ROC AUC: {roc_auc:.3f}")

print(f"• Most important positive predictor: {coefficients.iloc[0]['Feature']}")
print(f"• Most important negative predictor: {coefficients.iloc[-1]['Feature']}")

print("\nCoefficient Interpretation:")
print("• Positive coefficients increase the odds of positive class")
print("• Negative coefficients decrease the odds of positive class")
print("• Larger absolute values indicate stronger influence")

# ================================
# 12. PREDICTION ON NEW DATA
# ================================
print("\n12. Making Predictions on New Data")
print("-" * 20)

# Example of how to use the model for new predictions
# Create example new data (replace with actual new data)
new_data_example = X_test.iloc[:5].copy()  # Use first 5 test samples as example

# Scale the new data
new_data_scaled = scaler.transform(new_data_example)

# Make predictions
new_predictions = best_model.predict(new_data_scaled)
new_probabilities = best_model.predict_proba(new_data_scaled)

print("Example predictions for new data:")
for i in range(len(new_predictions)):
    print(f"Sample {i+1}: Prediction = {new_predictions[i]}, "
          f"Probability = {new_probabilities[i].max():.3f}")

print("\nLogistic Regression analysis completed!")
print("="*50)

# ================================
# SUMMARY & RECOMMENDATIONS
# ================================
print("\nSUMMARY & RECOMMENDATIONS:")
print("• Logistic regression assumes linear relationship between features and log-odds")
print("• Feature scaling is important for logistic regression")
print("• Regularization (L1/L2) helps prevent overfitting")
print("• Check for multicollinearity between features")
print("• Consider feature interactions for better performance")
print("• Validate assumptions: linearity, independence, no multicollinearity")

print("\nRemember to:")
print("1. Replace 'your_data.csv' with your dataset")
print("2. Update 'target_column' and feature column names")
print("3. Handle missing values appropriately for your data")
print("4. Consider feature engineering and interactions")
print("5. Validate model assumptions before interpreting results")
