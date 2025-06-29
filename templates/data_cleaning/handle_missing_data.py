# Replace 'your_data.csv' with your dataset
# Missing Data Handling Template - Drop, Impute (Mean/Median), and Predictive Imputation

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load your dataset
df = pd.read_csv('your_data.csv')

# Display initial info about missing data
print("=== MISSING DATA OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Total missing values: {df.isnull().sum().sum()}")
print("\nMissing values by column:")
missing_info = df.isnull().sum()
missing_percent = (missing_info / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_info,
    'Missing_Percent': missing_percent
}).sort_values('Missing_Percent', ascending=False)
print(missing_df[missing_df['Missing_Count'] > 0])

# 1. DROP STRATEGY - Remove rows/columns with too many missing values
print("\n=== DROP STRATEGY ===")

# Drop columns with >50% missing data (adjust threshold as needed)
threshold = 0.5
cols_to_drop = missing_df[missing_df['Missing_Percent'] > threshold * 100].index.tolist()
if cols_to_drop:
    print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
    df_dropped = df.drop(columns=cols_to_drop)
else:
    df_dropped = df.copy()
    print("No columns exceed missing data threshold")

# Drop rows with any missing values (if dataset is large enough)
df_complete_cases = df_dropped.dropna()
print(f"Complete cases: {len(df_complete_cases)} / {len(df_dropped)} rows ({len(df_complete_cases)/len(df_dropped)*100:.1f}%)")

# 2. SIMPLE IMPUTATION - Mean/Median for numerical, Mode for categorical
print("\n=== SIMPLE IMPUTATION ===")

df_simple_imputed = df_dropped.copy()

# Numerical columns - mean for normal distribution, median for skewed
numerical_cols = df_simple_imputed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_simple_imputed.select_dtypes(include=['object']).columns.tolist()

# For numerical: Check skewness to decide between mean/median
for col in numerical_cols:
    if df_simple_imputed[col].isnull().sum() > 0:
        skewness = df_simple_imputed[col].skew()
        if abs(skewness) > 1:  # Highly skewed - use median
            impute_value = df_simple_imputed[col].median()
            method = 'median'
        else:  # Approximately normal - use mean
            impute_value = df_simple_imputed[col].mean()
            method = 'mean'
        
        df_simple_imputed[col].fillna(impute_value, inplace=True)
        print(f"Imputed {col} with {method}: {impute_value:.2f} (skewness: {skewness:.2f})")

# For categorical - use mode (most frequent value)
for col in categorical_cols:
    if df_simple_imputed[col].isnull().sum() > 0:
        mode_value = df_simple_imputed[col].mode()[0] if not df_simple_imputed[col].mode().empty else 'Unknown'
        df_simple_imputed[col].fillna(mode_value, inplace=True)
        print(f"Imputed {col} with mode: {mode_value}")

# 3. ADVANCED IMPUTATION - KNN and Predictive
print("\n=== ADVANCED IMPUTATION ===")

df_advanced = df_dropped.copy()

# Separate numerical and categorical for different imputation strategies
if numerical_cols:
    # KNN Imputation for numerical data (considers relationships between features)
    print("Applying KNN imputation to numerical columns...")
    knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
    df_advanced[numerical_cols] = knn_imputer.fit_transform(df_advanced[numerical_cols])

if categorical_cols:
    # Simple mode imputation for categorical (or use more advanced methods if needed)
    print("Applying mode imputation to categorical columns...")
    for col in categorical_cols:
        if df_advanced[col].isnull().sum() > 0:
            mode_value = df_advanced[col].mode()[0] if not df_advanced[col].mode().empty else 'Unknown'
            df_advanced[col].fillna(mode_value, inplace=True)

# 4. PREDICTIVE IMPUTATION - Use other features to predict missing values
print("\n=== PREDICTIVE IMPUTATION EXAMPLE ===")
# Example: If 'age' has missing values, predict it using other features

# Choose target column with missing values (replace 'target_column' with actual column name)
target_column = 'target_column'  # Replace with your column name

if target_column in df.columns and df[target_column].isnull().sum() > 0:
    print(f"Predictive imputation for: {target_column}")
    
    # Prepare data for prediction
    df_pred = df_simple_imputed.copy()
    
    # Split into complete and incomplete cases for target column
    complete_mask = df_pred[target_column].notnull()
    X_complete = df_pred[complete_mask].drop(columns=[target_column])
    y_complete = df_pred[complete_mask][target_column]
    X_missing = df_pred[~complete_mask].drop(columns=[target_column])
    
    # Encode categorical variables for ML model
    X_complete_encoded = pd.get_dummies(X_complete)
    X_missing_encoded = pd.get_dummies(X_missing)
    
    # Ensure same columns in both sets
    common_cols = X_complete_encoded.columns.intersection(X_missing_encoded.columns)
    X_complete_encoded = X_complete_encoded[common_cols]
    X_missing_encoded = X_missing_encoded[common_cols]
    
    # Train predictor (choose based on target type)
    if df_pred[target_column].dtype in ['object', 'category']:
        # Classification for categorical target
        predictor = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        # Regression for numerical target
        predictor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    predictor.fit(X_complete_encoded, y_complete)
    predictions = predictor.predict(X_missing_encoded)
    
    # Fill missing values with predictions
    df_pred.loc[~complete_mask, target_column] = predictions
    print(f"Predicted {len(predictions)} missing values for {target_column}")

# SUMMARY AND RECOMMENDATIONS
print("\n=== SUMMARY ===")
print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"After dropping high-missing columns: {df_dropped.shape[0]} rows, {df_dropped.shape[1]} columns")
print(f"Complete cases only: {df_complete_cases.shape[0]} rows")
print(f"After simple imputation: {df_simple_imputed.isnull().sum().sum()} missing values")
print(f"After advanced imputation: {df_advanced.isnull().sum().sum()} missing values")

print("\n=== RECOMMENDATIONS ===")
print("1. Use complete cases if you have >10,000 rows and <20% missing")
print("2. Use simple imputation for quick analysis and missing <30%")
print("3. Use KNN/predictive imputation for critical analysis or >30% missing")
print("4. Always validate imputation quality with domain knowledge")
print("5. Consider creating 'missing indicator' columns for important features")

# Example: Create missing indicators for important columns
# df['column_name_missing'] = df['column_name'].isnull().astype(int)

# Save cleaned datasets (uncomment as needed)
# df_simple_imputed.to_csv('data_simple_imputed.csv', index=False)
# df_advanced.to_csv('data_advanced_imputed.csv', index=False)
# df_complete_cases.to_csv('data_complete_cases.csv', index=False)
