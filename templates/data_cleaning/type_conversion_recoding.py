# Replace 'your_data.csv' with your dataset
# Data Type Conversion and Recoding Template

import pandas as pd
import numpy as np
from datetime import datetime

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== DATA TYPE ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"\nCurrent data types:")
print(df.dtypes)
print(f"\nMemory usage:")
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

# Display sample of data to understand what needs conversion
print(f"\nFirst few rows:")
print(df.head())

# 1. CONVERT STRINGS TO DATETIME
print("\n=== DATETIME CONVERSION ===")

# Replace 'date_column' with your actual date column name
date_columns = ['date_column', 'created_at', 'timestamp']  # Add your date column names

for col in date_columns:
    if col in df.columns:
        print(f"\nConverting {col} to datetime...")
        
        # Handle different date formats
        try:
            # Try automatic parsing first
            df[col] = pd.to_datetime(df[col])
            print(f"Successfully converted {col} using automatic parsing")
        except:
            try:
                # Try common formats
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
                print(f"Successfully converted {col} using YYYY-MM-DD format")
            except:
                try:
                    df[col] = pd.to_datetime(df[col], format='%m/%d/%Y')
                    print(f"Successfully converted {col} using MM/DD/YYYY format")
                except:
                    print(f"Could not convert {col} - please check format manually")
        
        # Extract useful date components if conversion successful
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_weekday'] = df[col].dt.day_name()
            df[f'{col}_quarter'] = df[col].dt.quarter
            print(f"Created date components for {col}")

# 2. CONVERT STRINGS TO CATEGORICAL
print("\n=== CATEGORICAL CONVERSION ===")

# Identify potential categorical columns (strings with limited unique values)
string_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_candidates = []

for col in string_cols:
    unique_count = df[col].nunique()
    total_count = len(df)
    unique_ratio = unique_count / total_count
    
    # Consider categorical if <10% unique values or <50 unique values
    if unique_ratio < 0.1 or unique_count < 50:
        categorical_candidates.append(col)
        print(f"{col}: {unique_count} unique values ({unique_ratio:.3f} ratio) - Converting to categorical")
        df[col] = df[col].astype('category')
    else:
        print(f"{col}: {unique_count} unique values ({unique_ratio:.3f} ratio) - Keeping as string")

# 3. CONVERT NUMERICAL STRINGS TO NUMBERS
print("\n=== NUMERICAL CONVERSION ===")

for col in df.select_dtypes(include=['object']).columns:
    # Skip already processed categorical columns
    if col in categorical_candidates:
        continue
        
    # Try to convert to numeric
    sample_values = df[col].dropna().head(10).astype(str)
    
    # Check if values look numeric (handle currency, percentages, etc.)
    numeric_pattern = all(
        val.replace('$', '').replace(',', '').replace('%', '').replace('-', '').replace('.', '').isdigit()
        for val in sample_values if val.strip() != ''
    )
    
    if numeric_pattern:
        print(f"Converting {col} to numeric...")
        # Clean and convert
        df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Successfully converted {col}")

# 4. OPTIMIZE NUMERICAL DATA TYPES
print("\n=== NUMERICAL OPTIMIZATION ===")

# Optimize integer columns
int_cols = df.select_dtypes(include=['int64']).columns
for col in int_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    
    # Choose smallest integer type that fits the data
    if col_min >= 0:  # Unsigned integers
        if col_max < 255:
            df[col] = df[col].astype('uint8')
            print(f"Optimized {col}: int64 -> uint8")
        elif col_max < 65535:
            df[col] = df[col].astype('uint16')
            print(f"Optimized {col}: int64 -> uint16")
        elif col_max < 4294967295:
            df[col] = df[col].astype('uint32')
            print(f"Optimized {col}: int64 -> uint32")
    else:  # Signed integers
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype('int8')
            print(f"Optimized {col}: int64 -> int8")
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype('int16')
            print(f"Optimized {col}: int64 -> int16")
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype('int32')
            print(f"Optimized {col}: int64 -> int32")

# Optimize float columns
float_cols = df.select_dtypes(include=['float64']).columns
for col in float_cols:
    # Check if values can fit in float32 without significant precision loss
    if df[col].max() < 3.4e38 and df[col].min() > -3.4e38:
        # Test conversion
        original = df[col].copy()
        converted = df[col].astype('float32')
        
        # Check if precision loss is acceptable (within 0.1% relative error)
        relative_error = abs((original - converted) / original).max()
        if relative_error < 0.001:  # Less than 0.1% error
            df[col] = converted
            print(f"Optimized {col}: float64 -> float32 (max error: {relative_error:.6f})")

# 5. RECODING VALUES
print("\n=== VALUE RECODING ===")

# Example recodings - adjust for your data

# Binary recoding (Yes/No, True/False, etc.)
binary_mappings = {
    'Yes': 1, 'No': 0,
    'True': 1, 'False': 0,
    'Y': 1, 'N': 0,
    'Male': 1, 'Female': 0,
    'Active': 1, 'Inactive': 0
}

for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        unique_vals = set(df[col].dropna().astype(str).str.title())
        
        # Check if column matches binary pattern
        if unique_vals.issubset(set(binary_mappings.keys())):
            print(f"Binary recoding {col}: {unique_vals}")
            df[col] = df[col].astype(str).str.title().map(binary_mappings)

# Ordinal recoding (Low/Medium/High, etc.)
ordinal_mappings = {
    'education_level': {
        'High School': 1,
        'Bachelor': 2,
        'Master': 3,
        'PhD': 4
    },
    'satisfaction': {
        'Very Dissatisfied': 1,
        'Dissatisfied': 2,
        'Neutral': 3,
        'Satisfied': 4,
        'Very Satisfied': 5
    }
}

for col, mapping in ordinal_mappings.items():
    if col in df.columns:
        print(f"Ordinal recoding {col}: {mapping}")
        df[col] = df[col].map(mapping)

# 6. CREATE DERIVED VARIABLES
print("\n=== DERIVED VARIABLES ===")

# Age groups (if age column exists)
if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 18, 30, 50, 65, 100], 
                            labels=['Under 18', '18-30', '31-50', '51-65', '65+'])
    print("Created age_group variable")

# Income brackets (if income column exists)
if 'income' in df.columns:
    df['income_bracket'] = pd.qcut(df['income'], 
                                  q=5, 
                                  labels=['Low', 'Lower-Mid', 'Middle', 'Upper-Mid', 'High'])
    print("Created income_bracket variable")

# Boolean flags
for col in df.columns:
    if 'amount' in col.lower() or 'price' in col.lower() or 'cost' in col.lower():
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            df[f'{col}_is_zero'] = (df[col] == 0).astype(int)
            df[f'{col}_is_missing'] = df[col].isnull().astype(int)
            print(f"Created boolean flags for {col}")

# SUMMARY
print("\n=== FINAL SUMMARY ===")
print(f"Final data types:")
print(df.dtypes)
print(f"\nFinal memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Shape: {df.shape}")

# Verification
print(f"\nCategorical columns: {df.select_dtypes(include=['category']).columns.tolist()}")
print(f"Datetime columns: {df.select_dtypes(include=['datetime64']).columns.tolist()}")
print(f"Numerical columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")

# Save the processed dataset
# df.to_csv('data_processed_types.csv', index=False)
print("\nTemplate complete. Uncomment the save line to export processed data.")
