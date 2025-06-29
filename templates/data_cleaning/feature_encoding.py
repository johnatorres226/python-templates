# Replace 'your_data.csv' with your dataset
# Feature Encoding Template - One-Hot, Label, Target, and Binary Encoding

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import KFold

# Load your dataset
df = pd.read_csv('your_data.csv')

print("=== FEATURE ENCODING ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Analyze categorical columns
print("\n=== CATEGORICAL COLUMN ANALYSIS ===")
for col in categorical_cols:
    unique_count = df[col].nunique()
    total_count = len(df)
    unique_ratio = unique_count / total_count
    print(f"{col}: {unique_count} unique values ({unique_ratio:.3f} ratio)")
    
    # Show value counts for columns with few categories
    if unique_count <= 10:
        print(f"  Values: {df[col].value_counts().to_dict()}")
    else:
        print(f"  Top 5 values: {df[col].value_counts().head().to_dict()}")

# Create working copy
df_encoded = df.copy()

# 1. ONE-HOT ENCODING (for nominal categorical variables with few categories)
print("\n=== ONE-HOT ENCODING ===")

# Identify columns suitable for one-hot encoding (typically <10-15 unique values)
onehot_threshold = 10
onehot_candidates = [col for col in categorical_cols if df[col].nunique() <= onehot_threshold]

print(f"Columns suitable for one-hot encoding: {onehot_candidates}")

for col in onehot_candidates:
    print(f"One-hot encoding: {col}")
    
    # Create dummy variables
    dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
    
    # Add to dataframe and remove original
    df_encoded = pd.concat([df_encoded, dummies], axis=1)
    df_encoded.drop(columns=[col], inplace=True)
    
    print(f"  Created {len(dummies.columns)} dummy columns")
    print(f"  Columns: {list(dummies.columns)}")

# 2. LABEL ENCODING (for ordinal variables or high-cardinality nominal variables)
print("\n=== LABEL ENCODING ===")

# Columns not suitable for one-hot (too many categories)
label_candidates = [col for col in categorical_cols if col not in onehot_candidates]

# Manual ordinal mappings (define these based on your domain knowledge)
ordinal_mappings = {
    'education_level': ['High School', 'Bachelor', 'Master', 'PhD'],
    'satisfaction': ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
    'size': ['Small', 'Medium', 'Large', 'Extra Large'],
    'priority': ['Low', 'Medium', 'High', 'Critical']
}

for col in label_candidates:
    print(f"Label encoding: {col}")
    
    if col in ordinal_mappings:
        # Use predefined ordinal mapping
        mapping = {val: idx for idx, val in enumerate(ordinal_mappings[col])}
        df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
        print(f"  Ordinal mapping: {mapping}")
        
        # Keep original for reference
        df_encoded[f'{col}_original'] = df_encoded[col]
        df_encoded.drop(columns=[col], inplace=True)
        
    else:
        # Use automatic label encoding (alphabetical order)
        le = LabelEncoder()
        
        # Handle missing values
        df_temp = df_encoded[col].fillna('Missing')
        encoded_values = le.fit_transform(df_temp)
        
        df_encoded[f'{col}_encoded'] = encoded_values
        print(f"  Automatic encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        # Keep original for reference
        df_encoded[f'{col}_original'] = df_encoded[col]
        df_encoded.drop(columns=[col], inplace=True)

# 3. BINARY ENCODING (for high-cardinality categorical variables)
print("\n=== BINARY ENCODING (Manual Implementation) ===")

# Identify high-cardinality columns that might benefit from binary encoding
high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 50]

if high_cardinality_cols:
    print(f"High-cardinality columns for binary encoding: {high_cardinality_cols}")
    
    for col in high_cardinality_cols:
        if f'{col}_original' in df_encoded.columns:
            # Get unique values
            unique_vals = df_encoded[f'{col}_original'].dropna().unique()
            n_bits = int(np.ceil(np.log2(len(unique_vals))))
            
            print(f"Binary encoding {col}: {len(unique_vals)} categories -> {n_bits} binary columns")
            
            # Create mapping
            val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
            
            # Convert to binary representation
            for bit in range(n_bits):
                col_name = f'{col}_bin_{bit}'
                df_encoded[col_name] = df_encoded[f'{col}_original'].map(val_to_int).apply(
                    lambda x: (x >> bit) & 1 if pd.notnull(x) else 0
                )
            
            print(f"  Created binary columns: {[f'{col}_bin_{i}' for i in range(n_bits)]}")

# 4. TARGET ENCODING (for categorical variables in supervised learning)
print("\n=== TARGET ENCODING ===")

# Replace 'target_column' with your actual target variable
target_column = 'target_column'  # Replace with your target column name

if target_column in df.columns:
    print(f"Target encoding using target: {target_column}")
    
    # Check if target is binary/continuous
    is_binary_target = df[target_column].nunique() == 2
    
    # Get categorical columns from original dataframe for target encoding
    target_encode_cols = [col for col in categorical_cols if col not in onehot_candidates]
    
    for col in target_encode_cols:
        if col in df.columns:  # Make sure column still exists
            print(f"Target encoding: {col}")
            
            # Calculate mean target for each category (with smoothing for small samples)
            category_means = df.groupby(col)[target_column].agg(['mean', 'count']).reset_index()
            
            # Add smoothing for categories with few samples
            global_mean = df[target_column].mean()
            min_samples = 10  # Minimum samples for reliable estimate
            
            def smooth_mean(row):
                if row['count'] < min_samples:
                    # Weighted average between category mean and global mean
                    weight = row['count'] / min_samples
                    return weight * row['mean'] + (1 - weight) * global_mean
                return row['mean']
            
            category_means['smoothed_mean'] = category_means.apply(smooth_mean, axis=1)
            
            # Create mapping
            target_mapping = dict(zip(category_means[col], category_means['smoothed_mean']))
            
            # Apply encoding
            df_encoded[f'{col}_target_encoded'] = df[col].map(target_mapping)
            
            print(f"  Encoded {len(target_mapping)} categories")
            print(f"  Sample mappings: {dict(list(target_mapping.items())[:3])}")

# 5. FREQUENCY ENCODING
print("\n=== FREQUENCY ENCODING ===")

for col in categorical_cols:
    if col in df.columns:  # Check if original column still exists
        print(f"Frequency encoding: {col}")
        
        # Calculate frequency of each category
        freq_map = df[col].value_counts().to_dict()
        df_encoded[f'{col}_frequency'] = df[col].map(freq_map)
        
        print(f"  Frequency range: {min(freq_map.values())} to {max(freq_map.values())}")

# 6. FEATURE INTERACTIONS (creating interaction features)
print("\n=== FEATURE INTERACTIONS ===")

# Create interactions between important categorical and numerical features
# Example: interaction between category and price
interaction_examples = [
    ('category', 'price'),  # Replace with your actual column names
    ('region', 'age'),
    ('type', 'amount')
]

for cat_col, num_col in interaction_examples:
    if cat_col in df.columns and num_col in df.columns:
        print(f"Creating interaction: {cat_col} × {num_col}")
        
        # Group statistics
        interaction_stats = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'median']).reset_index()
        
        # Map back to original data
        for stat in ['mean', 'std', 'median']:
            mapping = dict(zip(interaction_stats[cat_col], interaction_stats[stat]))
            df_encoded[f'{cat_col}_{num_col}_{stat}'] = df[cat_col].map(mapping)
        
        print(f"  Created {cat_col}_{num_col}_[mean/std/median] features")

# SUMMARY
print("\n=== ENCODING SUMMARY ===")
print(f"Original shape: {df.shape}")
print(f"Encoded shape: {df_encoded.shape}")
print(f"Features added: {df_encoded.shape[1] - df.shape[1]}")

# Show new columns by encoding type
new_columns = [col for col in df_encoded.columns if col not in df.columns]
print(f"\nNew columns created ({len(new_columns)}):")

encoding_types = {
    'one-hot': [col for col in new_columns if any(orig in col for orig in onehot_candidates)],
    'label': [col for col in new_columns if '_encoded' in col and '_target_' not in col],
    'binary': [col for col in new_columns if '_bin_' in col],
    'target': [col for col in new_columns if '_target_encoded' in col],
    'frequency': [col for col in new_columns if '_frequency' in col],
    'interaction': [col for col in new_columns if any(stat in col for stat in ['_mean', '_std', '_median'])]
}

for enc_type, cols in encoding_types.items():
    if cols:
        print(f"  {enc_type}: {len(cols)} columns")

# Memory usage comparison
print(f"\nMemory usage:")
print(f"  Original: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"  Encoded: {df_encoded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# RECOMMENDATIONS
print("\n=== ENCODING RECOMMENDATIONS ===")
print("1. One-hot encoding: Use for nominal variables with <10 categories")
print("2. Label encoding: Use for ordinal variables or high-cardinality nominal")
print("3. Binary encoding: Use for very high-cardinality variables (>50 categories)")
print("4. Target encoding: Use carefully - can cause overfitting, use cross-validation")
print("5. Frequency encoding: Good for variables where frequency matters")
print("6. Consider regularization (L1/L2) when using many encoded features")

# Save encoded dataset
# df_encoded.to_csv('data_encoded.csv', index=False)
print("\nEncoding complete. Uncomment the save line to export encoded data.")
