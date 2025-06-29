"""
===============================================================================
DATA VALIDATION TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive data validation and quality assessment

This template covers:
- Data type validation and consistency checks
- Range and constraint validation
- Schema validation and data profiling
- Referential integrity checks
- Data quality scoring and reporting
- Custom validation rules
- Automated validation pipelines

Prerequisites:
- pandas, numpy, matplotlib, seaborn
- Dataset loaded as 'df' for validation
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===============================================================================
# CREATE SAMPLE DATA FOR DEMONSTRATION
# ===============================================================================

def create_sample_data_with_issues():
    """Create sample dataset with various data quality issues"""
    n_samples = 1000
    
    # Generate base data
    data = {
        'customer_id': range(1, n_samples + 1),
        'name': [f'Customer_{i}' for i in range(1, n_samples + 1)],
        'email': [f'customer{i}@email.com' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'purchase_date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'purchase_amount': np.random.exponential(100, n_samples),
        'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France'], n_samples),
        'phone': [f'+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    issues_to_introduce = int(0.1 * n_samples)  # 10% of data will have issues
    
    # 1. Missing values
    missing_indices = np.random.choice(n_samples, size=issues_to_introduce//5, replace=False)
    df.loc[missing_indices, 'email'] = None
    df.loc[missing_indices[:len(missing_indices)//2], 'phone'] = None
    
    # 2. Invalid email formats
    invalid_email_indices = np.random.choice(n_samples, size=issues_to_introduce//10, replace=False)
    df.loc[invalid_email_indices, 'email'] = 'invalid_email'
    
    # 3. Negative values where they shouldn't be
    negative_indices = np.random.choice(n_samples, size=issues_to_introduce//10, replace=False)
    df.loc[negative_indices, 'age'] = -df.loc[negative_indices, 'age']
    df.loc[negative_indices, 'income'] = -abs(df.loc[negative_indices, 'income'])
    
    # 4. Outliers
    outlier_indices = np.random.choice(n_samples, size=issues_to_introduce//10, replace=False)
    df.loc[outlier_indices, 'age'] = np.random.randint(150, 200, len(outlier_indices))
    df.loc[outlier_indices, 'income'] = np.random.randint(1000000, 5000000, len(outlier_indices))
    
    # 5. Inconsistent formats
    inconsistent_indices = np.random.choice(n_samples, size=issues_to_introduce//10, replace=False)
    df.loc[inconsistent_indices, 'country'] = df.loc[inconsistent_indices, 'country'].str.lower()
    
    # 6. Duplicate customer IDs (referential integrity issue)
    duplicate_indices = np.random.choice(n_samples, size=issues_to_introduce//20, replace=False)
    df.loc[duplicate_indices, 'customer_id'] = df.loc[duplicate_indices[0], 'customer_id']
    
    # 7. Invalid phone formats
    invalid_phone_indices = np.random.choice(n_samples, size=issues_to_introduce//10, replace=False)
    df.loc[invalid_phone_indices, 'phone'] = '123'
    
    # 8. Future dates
    future_indices = np.random.choice(n_samples, size=issues_to_introduce//20, replace=False)
    df.loc[future_indices, 'purchase_date'] = pd.Timestamp.now() + timedelta(days=30)
    
    return df

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Create sample data for demonstration
df = create_sample_data_with_issues()

print("Sample Dataset Created:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# ===============================================================================
# 1. DATA TYPE VALIDATION
# ===============================================================================

print("\n" + "="*60)
print("1. DATA TYPE VALIDATION")
print("="*60)

class DataTypeValidator:
    """Class for validating data types and formats"""
    
    def __init__(self, df):
        self.df = df
        self.validation_results = {}
    
    def validate_data_types(self, expected_types):
        """Validate column data types against expected types"""
        print("1.1 Data Type Validation")
        print("-" * 30)
        
        type_issues = {}
        
        for column, expected_type in expected_types.items():
            if column in self.df.columns:
                actual_type = str(self.df[column].dtype)
                
                # Check if types match
                if expected_type == 'numeric':
                    is_valid = pd.api.types.is_numeric_dtype(self.df[column])
                elif expected_type == 'datetime':
                    is_valid = pd.api.types.is_datetime64_any_dtype(self.df[column])
                elif expected_type == 'string':
                    is_valid = pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column])
                elif expected_type == 'categorical':
                    is_valid = pd.api.types.is_categorical_dtype(self.df[column])
                else:
                    is_valid = expected_type in actual_type
                
                if not is_valid:
                    type_issues[column] = {
                        'expected': expected_type,
                        'actual': actual_type,
                        'sample_values': self.df[column].dropna().head(3).tolist()
                    }
                
                print(f"{column:<20}: Expected {expected_type:<12}, Got {actual_type:<12} {'✓' if is_valid else '✗'}")
            else:
                print(f"{column:<20}: Column not found in dataset")
        
        self.validation_results['data_types'] = type_issues
        return type_issues
    
    def validate_numeric_ranges(self, range_constraints):
        """Validate numeric columns against expected ranges"""
        print("\n1.2 Numeric Range Validation")
        print("-" * 35)
        
        range_issues = {}
        
        for column, constraints in range_constraints.items():
            if column in self.df.columns and pd.api.types.is_numeric_dtype(self.df[column]):
                min_val, max_val = constraints['min'], constraints['max']
                
                # Check for values outside range
                out_of_range = self.df[
                    (self.df[column] < min_val) | (self.df[column] > max_val)
                ][column].dropna()
                
                if len(out_of_range) > 0:
                    range_issues[column] = {
                        'constraint': f'{min_val} <= {column} <= {max_val}',
                        'violations': len(out_of_range),
                        'violation_rate': len(out_of_range) / len(self.df),
                        'sample_violations': out_of_range.head(5).tolist()
                    }
                
                print(f"{column:<20}: Range [{min_val}, {max_val}], "
                      f"Violations: {len(out_of_range)} {'✓' if len(out_of_range) == 0 else '✗'}")
            else:
                print(f"{column:<20}: Column not found or not numeric")
        
        self.validation_results['range_validation'] = range_issues
        return range_issues
    
    def validate_email_format(self, email_column):
        """Validate email format using regex"""
        print(f"\n1.3 Email Format Validation ({email_column})")
        print("-" * 45)
        
        if email_column not in self.df.columns:
            print(f"Column '{email_column}' not found")
            return {}
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        # Check email format
        valid_emails = self.df[email_column].dropna().apply(
            lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
        )
        
        invalid_count = (~valid_emails).sum()
        total_count = len(valid_emails)
        
        email_issues = {
            'total_emails': total_count,
            'invalid_count': invalid_count,
            'invalid_rate': invalid_count / total_count if total_count > 0 else 0,
            'sample_invalid': self.df[email_column][~valid_emails].head(5).tolist()
        }
        
        print(f"Total emails: {total_count}")
        print(f"Invalid emails: {invalid_count}")
        print(f"Invalid rate: {email_issues['invalid_rate']:.2%}")
        if email_issues['sample_invalid']:
            print(f"Sample invalid: {email_issues['sample_invalid']}")
        
        self.validation_results['email_validation'] = email_issues
        return email_issues
    
    def validate_phone_format(self, phone_column):
        """Validate phone number format"""
        print(f"\n1.4 Phone Format Validation ({phone_column})")
        print("-" * 45)
        
        if phone_column not in self.df.columns:
            print(f"Column '{phone_column}' not found")
            return {}
        
        # Common phone patterns
        phone_patterns = [
            r'^\+1-\d{3}-\d{3}-\d{4}$',  # +1-555-123-4567
            r'^\(\d{3}\) \d{3}-\d{4}$',  # (555) 123-4567
            r'^\d{3}-\d{3}-\d{4}$',      # 555-123-4567
            r'^\d{10}$'                   # 5551234567
        ]
        
        def is_valid_phone(phone):
            if pd.isna(phone):
                return False
            phone_str = str(phone)
            return any(re.match(pattern, phone_str) for pattern in phone_patterns)
        
        valid_phones = self.df[phone_column].apply(is_valid_phone)
        invalid_count = (~valid_phones).sum()
        total_count = len(self.df[phone_column].dropna())
        
        phone_issues = {
            'total_phones': total_count,
            'invalid_count': invalid_count,
            'invalid_rate': invalid_count / total_count if total_count > 0 else 0,
            'sample_invalid': self.df[phone_column][~valid_phones].dropna().head(5).tolist()
        }
        
        print(f"Total phones: {total_count}")
        print(f"Invalid phones: {invalid_count}")
        print(f"Invalid rate: {phone_issues['invalid_rate']:.2%}")
        if phone_issues['sample_invalid']:
            print(f"Sample invalid: {phone_issues['sample_invalid']}")
        
        self.validation_results['phone_validation'] = phone_issues
        return phone_issues

# Initialize validator
validator = DataTypeValidator(df)

# Define expected data types
expected_types = {
    'customer_id': 'numeric',
    'name': 'string',
    'email': 'string',
    'age': 'numeric',
    'income': 'numeric',
    'purchase_date': 'datetime',
    'product_category': 'string',
    'purchase_amount': 'numeric',
    'country': 'string',
    'phone': 'string'
}

# Define range constraints
range_constraints = {
    'age': {'min': 0, 'max': 120},
    'income': {'min': 0, 'max': 1000000},
    'purchase_amount': {'min': 0, 'max': 10000}
}

# Run validations
type_issues = validator.validate_data_types(expected_types)
range_issues = validator.validate_numeric_ranges(range_constraints)
email_issues = validator.validate_email_format('email')
phone_issues = validator.validate_phone_format('phone')

# ===============================================================================
# 2. SCHEMA VALIDATION AND DATA PROFILING
# ===============================================================================

print("\n" + "="*60)
print("2. SCHEMA VALIDATION AND DATA PROFILING")
print("="*60)

class SchemaValidator:
    """Class for schema validation and data profiling"""
    
    def __init__(self, df):
        self.df = df
    
    def validate_schema(self, expected_schema):
        """Validate dataset schema against expected schema"""
        print("2.1 Schema Validation")
        print("-" * 25)
        
        schema_issues = {
            'missing_columns': [],
            'extra_columns': [],
            'column_type_mismatches': []
        }
        
        expected_columns = set(expected_schema.keys())
        actual_columns = set(self.df.columns)
        
        # Check for missing columns
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            schema_issues['missing_columns'] = list(missing_columns)
            print(f"Missing columns: {missing_columns}")
        
        # Check for extra columns
        extra_columns = actual_columns - expected_columns
        if extra_columns:
            schema_issues['extra_columns'] = list(extra_columns)
            print(f"Extra columns: {extra_columns}")
        
        # Check column types
        for column in expected_columns.intersection(actual_columns):
            expected_type = expected_schema[column]
            actual_type = str(self.df[column].dtype)
            
            if expected_type not in actual_type:
                schema_issues['column_type_mismatches'].append({
                    'column': column,
                    'expected': expected_type,
                    'actual': actual_type
                })
        
        if not any(schema_issues.values()):
            print("✓ Schema validation passed")
        else:
            print("✗ Schema validation failed")
            if schema_issues['column_type_mismatches']:
                print("Type mismatches:")
                for mismatch in schema_issues['column_type_mismatches']:
                    print(f"  {mismatch['column']}: expected {mismatch['expected']}, got {mismatch['actual']}")
        
        return schema_issues
    
    def profile_data_quality(self):
        """Generate comprehensive data quality profile"""
        print("\n2.2 Data Quality Profiling")
        print("-" * 30)
        
        profile = {}
        
        for column in self.df.columns:
            col_profile = {
                'dtype': str(self.df[column].dtype),
                'non_null_count': self.df[column].count(),
                'null_count': self.df[column].isnull().sum(),
                'null_percentage': (self.df[column].isnull().sum() / len(self.df)) * 100,
                'unique_count': self.df[column].nunique(),
                'unique_percentage': (self.df[column].nunique() / len(self.df)) * 100
            }
            
            # Add numeric statistics
            if pd.api.types.is_numeric_dtype(self.df[column]):
                col_profile.update({
                    'mean': self.df[column].mean(),
                    'std': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max(),
                    'q25': self.df[column].quantile(0.25),
                    'q50': self.df[column].quantile(0.50),
                    'q75': self.df[column].quantile(0.75)
                })
            
            # Add string statistics
            elif pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                non_null_values = self.df[column].dropna()
                if len(non_null_values) > 0:
                    col_profile.update({
                        'avg_length': non_null_values.astype(str).str.len().mean(),
                        'min_length': non_null_values.astype(str).str.len().min(),
                        'max_length': non_null_values.astype(str).str.len().max(),
                        'most_common': non_null_values.value_counts().head(3).to_dict()
                    })
            
            profile[column] = col_profile
        
        # Display profile summary
        profile_df = pd.DataFrame(profile).T
        print("Data Quality Profile Summary:")
        print(profile_df[['dtype', 'null_count', 'null_percentage', 'unique_count']].round(2))
        
        return profile
    
    def detect_data_anomalies(self):
        """Detect various data anomalies"""
        print("\n2.3 Data Anomaly Detection")
        print("-" * 32)
        
        anomalies = {}
        
        for column in self.df.columns:
            col_anomalies = []
            
            # Check for high null percentage
            null_pct = (self.df[column].isnull().sum() / len(self.df)) * 100
            if null_pct > 50:
                col_anomalies.append(f"High null percentage: {null_pct:.1f}%")
            
            # Check for single value columns
            if self.df[column].nunique() == 1:
                col_anomalies.append("Single unique value (constant column)")
            
            # Check for numeric anomalies
            if pd.api.types.is_numeric_dtype(self.df[column]):
                # Detect outliers using IQR
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)][column]
                if len(outliers) > 0:
                    outlier_pct = (len(outliers) / len(self.df)) * 100
                    col_anomalies.append(f"Outliers detected: {len(outliers)} ({outlier_pct:.1f}%)")
                
                # Check for negative values where they might not be expected
                if column in ['age', 'income', 'purchase_amount']:
                    negative_count = (self.df[column] < 0).sum()
                    if negative_count > 0:
                        col_anomalies.append(f"Negative values: {negative_count}")
            
            # Check for string anomalies
            elif pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_object_dtype(self.df[column]):
                # Check for inconsistent case
                non_null_strings = self.df[column].dropna().astype(str)
                if len(non_null_strings) > 0:
                    case_variations = len(set(non_null_strings.str.lower())) != len(set(non_null_strings))
                    if case_variations:
                        col_anomalies.append("Inconsistent case formatting")
                    
                    # Check for unusual length variations
                    lengths = non_null_strings.str.len()
                    if lengths.std() > lengths.mean():
                        col_anomalies.append("High length variation")
            
            if col_anomalies:
                anomalies[column] = col_anomalies
                print(f"\n{column}:")
                for anomaly in col_anomalies:
                    print(f"  ⚠ {anomaly}")
        
        if not anomalies:
            print("✓ No significant anomalies detected")
        
        return anomalies

# Schema validation
schema_validator = SchemaValidator(df)

# Define expected schema
expected_schema = {
    'customer_id': 'int64',
    'name': 'object',
    'email': 'object',
    'age': 'int64',
    'income': 'float64',
    'purchase_date': 'datetime64',
    'product_category': 'object',
    'purchase_amount': 'float64',
    'country': 'object',
    'phone': 'object'
}

# Run schema validation
schema_issues = schema_validator.validate_schema(expected_schema)
data_profile = schema_validator.profile_data_quality()
anomalies = schema_validator.detect_data_anomalies()

# ===============================================================================
# 3. REFERENTIAL INTEGRITY CHECKS
# ===============================================================================

print("\n" + "="*60)
print("3. REFERENTIAL INTEGRITY CHECKS")
print("="*60)

class ReferentialIntegrityValidator:
    """Class for validating referential integrity"""
    
    def __init__(self, df):
        self.df = df
    
    def check_primary_key_uniqueness(self, primary_key_columns):
        """Check if primary key columns are unique"""
        print("3.1 Primary Key Uniqueness")
        print("-" * 30)
        
        if isinstance(primary_key_columns, str):
            primary_key_columns = [primary_key_columns]
        
        # Check if all columns exist
        missing_cols = [col for col in primary_key_columns if col not in self.df.columns]
        if missing_cols:
            print(f"Missing primary key columns: {missing_cols}")
            return {}
        
        # Create composite key
        if len(primary_key_columns) == 1:
            composite_key = self.df[primary_key_columns[0]]
        else:
            composite_key = self.df[primary_key_columns].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
        
        # Check for duplicates
        duplicates = composite_key.duplicated()
        duplicate_count = duplicates.sum()
        
        integrity_results = {
            'total_records': len(self.df),
            'duplicate_count': duplicate_count,
            'duplicate_rate': duplicate_count / len(self.df),
            'unique_count': composite_key.nunique(),
            'duplicate_values': composite_key[duplicates].unique()[:10].tolist()
        }
        
        print(f"Primary key columns: {primary_key_columns}")
        print(f"Total records: {integrity_results['total_records']}")
        print(f"Unique keys: {integrity_results['unique_count']}")
        print(f"Duplicates: {duplicate_count}")
        print(f"Duplicate rate: {integrity_results['duplicate_rate']:.2%}")
        
        if duplicate_count > 0:
            print("✗ Primary key uniqueness violated")
            print(f"Sample duplicate values: {integrity_results['duplicate_values']}")
        else:
            print("✓ Primary key uniqueness maintained")
        
        return integrity_results
    
    def check_foreign_key_integrity(self, foreign_key_mapping):
        """Check foreign key integrity between tables/columns"""
        print("\n3.2 Foreign Key Integrity")
        print("-" * 30)
        
        # This is a simplified version - in practice, you'd have multiple tables
        # Here we check for valid values in categorical columns
        
        integrity_issues = {}
        
        for column, valid_values in foreign_key_mapping.items():
            if column not in self.df.columns:
                print(f"Column '{column}' not found")
                continue
            
            # Check for invalid values
            invalid_values = self.df[~self.df[column].isin(valid_values)][column].dropna()
            invalid_count = len(invalid_values)
            
            if invalid_count > 0:
                integrity_issues[column] = {
                    'invalid_count': invalid_count,
                    'invalid_rate': invalid_count / len(self.df),
                    'invalid_values': invalid_values.unique()[:10].tolist()
                }
                
                print(f"\n{column}:")
                print(f"  Valid values: {valid_values}")
                print(f"  Invalid count: {invalid_count}")
                print(f"  Invalid rate: {integrity_issues[column]['invalid_rate']:.2%}")
                print(f"  Sample invalid: {integrity_issues[column]['invalid_values']}")
                print("  ✗ Foreign key integrity violated")
            else:
                print(f"\n{column}: ✓ Foreign key integrity maintained")
        
        return integrity_issues
    
    def check_cross_column_consistency(self, consistency_rules):
        """Check consistency between related columns"""
        print("\n3.3 Cross-Column Consistency")
        print("-" * 35)
        
        consistency_issues = {}
        
        for rule_name, rule_func in consistency_rules.items():
            try:
                violations = rule_func(self.df)
                violation_count = len(violations)
                
                if violation_count > 0:
                    consistency_issues[rule_name] = {
                        'violation_count': violation_count,
                        'violation_rate': violation_count / len(self.df),
                        'sample_violations': violations.head(5).index.tolist()
                    }
                    
                    print(f"\n{rule_name}:")
                    print(f"  Violations: {violation_count}")
                    print(f"  Violation rate: {consistency_issues[rule_name]['violation_rate']:.2%}")
                    print("  ✗ Consistency rule violated")
                else:
                    print(f"\n{rule_name}: ✓ Consistency rule satisfied")
                    
            except Exception as e:
                print(f"\n{rule_name}: Error evaluating rule - {str(e)}")
        
        return consistency_issues

# Referential integrity validation
integrity_validator = ReferentialIntegrityValidator(df)

# Check primary key uniqueness
primary_key_results = integrity_validator.check_primary_key_uniqueness('customer_id')

# Define valid foreign key values
foreign_key_mapping = {
    'product_category': ['Electronics', 'Clothing', 'Books', 'Home'],
    'country': ['USA', 'Canada', 'UK', 'Germany', 'France']
}

# Check foreign key integrity
foreign_key_results = integrity_validator.check_foreign_key_integrity(foreign_key_mapping)

# Define consistency rules
def age_income_consistency(df):
    """Check if income is reasonable for age"""
    # People under 18 should have lower income
    return df[(df['age'] < 18) & (df['income'] > 30000)]

def purchase_date_consistency(df):
    """Check if purchase dates are not in the future"""
    return df[df['purchase_date'] > pd.Timestamp.now()]

def purchase_amount_consistency(df):
    """Check if purchase amount is reasonable"""
    # Very high purchase amounts might be errors
    return df[df['purchase_amount'] > 5000]

consistency_rules = {
    'age_income_consistency': age_income_consistency,
    'purchase_date_consistency': purchase_date_consistency,
    'purchase_amount_consistency': purchase_amount_consistency
}

# Check cross-column consistency
consistency_results = integrity_validator.check_cross_column_consistency(consistency_rules)

# ===============================================================================
# 4. DATA QUALITY SCORING AND REPORTING
# ===============================================================================

print("\n" + "="*60)
print("4. DATA QUALITY SCORING AND REPORTING")
print("="*60)

class DataQualityScorer:
    """Class for calculating data quality scores"""
    
    def __init__(self, df):
        self.df = df
        self.quality_score = {}
    
    def calculate_completeness_score(self):
        """Calculate completeness score based on missing values"""
        total_cells = self.df.size
        missing_cells = self.df.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells
        
        self.quality_score['completeness'] = {
            'score': completeness_score,
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': (missing_cells / total_cells) * 100
        }
        
        return completeness_score
    
    def calculate_validity_score(self, validation_results):
        """Calculate validity score based on validation results"""
        total_records = len(self.df)
        invalid_records = 0
        
        # Count validation violations
        for validation_type, results in validation_results.items():
            if isinstance(results, dict):
                if 'violations' in results:
                    invalid_records += results['violations']
                elif 'invalid_count' in results:
                    invalid_records += results['invalid_count']
                elif isinstance(results, list):
                    invalid_records += len(results)
        
        validity_score = max(0, (total_records - invalid_records) / total_records)
        
        self.quality_score['validity'] = {
            'score': validity_score,
            'total_records': total_records,
            'invalid_records': invalid_records,
            'invalid_percentage': (invalid_records / total_records) * 100
        }
        
        return validity_score
    
    def calculate_uniqueness_score(self, primary_key_results):
        """Calculate uniqueness score based on primary key duplicates"""
        if primary_key_results:
            uniqueness_score = 1 - primary_key_results.get('duplicate_rate', 0)
        else:
            uniqueness_score = 1.0
        
        self.quality_score['uniqueness'] = {
            'score': uniqueness_score,
            'duplicate_rate': primary_key_results.get('duplicate_rate', 0) if primary_key_results else 0
        }
        
        return uniqueness_score
    
    def calculate_consistency_score(self, consistency_results):
        """Calculate consistency score based on cross-column validation"""
        total_records = len(self.df)
        inconsistent_records = sum(
            result['violation_count'] for result in consistency_results.values()
        )
        
        consistency_score = max(0, (total_records - inconsistent_records) / total_records)
        
        self.quality_score['consistency'] = {
            'score': consistency_score,
            'total_records': total_records,
            'inconsistent_records': inconsistent_records,
            'inconsistent_percentage': (inconsistent_records / total_records) * 100
        }
        
        return consistency_score
    
    def calculate_overall_score(self, weights=None):
        """Calculate overall data quality score"""
        if weights is None:
            weights = {
                'completeness': 0.3,
                'validity': 0.3,
                'uniqueness': 0.2,
                'consistency': 0.2
            }
        
        overall_score = sum(
            self.quality_score[dimension]['score'] * weight
            for dimension, weight in weights.items()
            if dimension in self.quality_score
        )
        
        self.quality_score['overall'] = {
            'score': overall_score,
            'weights': weights,
            'grade': self.get_quality_grade(overall_score)
        }
        
        return overall_score
    
    def get_quality_grade(self, score):
        """Convert quality score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def generate_quality_report(self):
        """Generate comprehensive data quality report"""
        print("4.1 Data Quality Score Report")
        print("-" * 35)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Columns: {len(self.df.columns)}")
        
        print(f"\nDATA QUALITY DIMENSIONS:")
        print(f"{'Dimension':<15} {'Score':<8} {'Grade':<6} {'Details'}")
        print("-" * 55)
        
        for dimension, metrics in self.quality_score.items():
            if dimension != 'overall':
                score = metrics['score']
                grade = self.get_quality_grade(score)
                
                if dimension == 'completeness':
                    details = f"{metrics['missing_percentage']:.1f}% missing"
                elif dimension == 'validity':
                    details = f"{metrics['invalid_percentage']:.1f}% invalid"
                elif dimension == 'uniqueness':
                    details = f"{metrics['duplicate_rate']:.1%} duplicates"
                elif dimension == 'consistency':
                    details = f"{metrics['inconsistent_percentage']:.1f}% inconsistent"
                else:
                    details = ""
                
                print(f"{dimension.capitalize():<15} {score:<8.3f} {grade:<6} {details}")
        
        if 'overall' in self.quality_score:
            overall_metrics = self.quality_score['overall']
            print("-" * 55)
            print(f"{'OVERALL':<15} {overall_metrics['score']:<8.3f} {overall_metrics['grade']:<6}")
        
        return self.quality_score

# Calculate data quality scores
quality_scorer = DataQualityScorer(df)

# Calculate individual dimension scores
completeness_score = quality_scorer.calculate_completeness_score()
validity_score = quality_scorer.calculate_validity_score({
    'email': email_issues,
    'phone': phone_issues,
    'ranges': range_issues
})
uniqueness_score = quality_scorer.calculate_uniqueness_score(primary_key_results)
consistency_score = quality_scorer.calculate_consistency_score(consistency_results)

# Calculate overall score
overall_score = quality_scorer.calculate_overall_score()

# Generate quality report
quality_report = quality_scorer.generate_quality_report()

# ===============================================================================
# 5. DATA QUALITY VISUALIZATION
# ===============================================================================

print("\n" + "="*60)
print("5. DATA QUALITY VISUALIZATION")
print("="*60)

def plot_data_quality_dashboard(df, quality_scorer):
    """Create comprehensive data quality dashboard"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Quality Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Missing Values Heatmap
    ax1 = axes[0, 0]
    missing_data = df.isnull()
    if missing_data.any().any():
        sns.heatmap(missing_data, cbar=True, ax=ax1, cmap='viridis', 
                   yticklabels=False, xticklabels=True)
        ax1.set_title('Missing Values Pattern')
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Missing Values Pattern')
    
    # 2. Data Quality Scores
    ax2 = axes[0, 1]
    dimensions = []
    scores = []
    colors = []
    
    for dimension, metrics in quality_scorer.quality_score.items():
        if dimension != 'overall':
            dimensions.append(dimension.capitalize())
            scores.append(metrics['score'])
            # Color based on score
            if metrics['score'] >= 0.8:
                colors.append('green')
            elif metrics['score'] >= 0.6:
                colors.append('orange')
            else:
                colors.append('red')
    
    bars = ax2.bar(dimensions, scores, color=colors, alpha=0.7)
    ax2.set_title('Data Quality Scores by Dimension')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Missing Values by Column
    ax3 = axes[0, 2]
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
    
    if len(missing_counts) > 0:
        missing_counts.plot(kind='barh', ax=ax3, color='red', alpha=0.7)
        ax3.set_title('Missing Values by Column')
        ax3.set_xlabel('Count of Missing Values')
    else:
        ax3.text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Missing Values by Column')
    
    # 4. Data Type Distribution
    ax4 = axes[1, 0]
    dtype_counts = df.dtypes.value_counts()
    ax4.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
           startangle=90)
    ax4.set_title('Data Type Distribution')
    
    # 5. Unique Values Distribution
    ax5 = axes[1, 1]
    unique_counts = df.nunique()
    ax5.hist(unique_counts, bins=min(20, len(unique_counts)), alpha=0.7, edgecolor='black')
    ax5.set_title('Distribution of Unique Values per Column')
    ax5.set_xlabel('Number of Unique Values')
    ax5.set_ylabel('Number of Columns')
    
    # 6. Overall Quality Score Gauge
    ax6 = axes[1, 2]
    overall_score = quality_scorer.quality_score.get('overall', {}).get('score', 0)
    
    # Create a gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Background semicircle
    ax6.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
    
    # Score indicator
    score_theta = np.pi * (1 - overall_score)
    ax6.plot([score_theta, score_theta], [0, 1], 'r-', linewidth=5)
    
    # Add score text
    ax6.text(np.pi/2, 0.5, f'{overall_score:.2f}', ha='center', va='center',
            fontsize=20, fontweight='bold')
    ax6.text(np.pi/2, 0.3, f'Grade: {quality_scorer.get_quality_grade(overall_score)}',
            ha='center', va='center', fontsize=14)
    
    ax6.set_xlim(0, np.pi)
    ax6.set_ylim(0, 1.2)
    ax6.set_title('Overall Data Quality Score')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.show()

# Generate data quality dashboard
plot_data_quality_dashboard(df, quality_scorer)

# ===============================================================================
# 6. AUTOMATED VALIDATION PIPELINE
# ===============================================================================

print("\n" + "="*60)
print("6. AUTOMATED VALIDATION PIPELINE")
print("="*60)

class AutomatedValidationPipeline:
    """Automated data validation pipeline"""
    
    def __init__(self):
        self.validation_steps = []
        self.results = {}
    
    def add_validation_step(self, name, validator_func, **kwargs):
        """Add a validation step to the pipeline"""
        self.validation_steps.append({
            'name': name,
            'validator': validator_func,
            'kwargs': kwargs
        })
    
    def run_pipeline(self, df):
        """Run the complete validation pipeline"""
        print("6.1 Running Automated Validation Pipeline")
        print("-" * 45)
        
        pipeline_start = datetime.now()
        
        for step in self.validation_steps:
            step_start = datetime.now()
            
            try:
                print(f"\nExecuting: {step['name']}")
                result = step['validator'](df, **step['kwargs'])
                step_duration = datetime.now() - step_start
                
                self.results[step['name']] = {
                    'status': 'success',
                    'result': result,
                    'duration': step_duration.total_seconds(),
                    'timestamp': step_start
                }
                
                print(f"✓ Completed in {step_duration.total_seconds():.2f}s")
                
            except Exception as e:
                step_duration = datetime.now() - step_start
                
                self.results[step['name']] = {
                    'status': 'error',
                    'error': str(e),
                    'duration': step_duration.total_seconds(),
                    'timestamp': step_start
                }
                
                print(f"✗ Failed: {str(e)}")
        
        pipeline_duration = datetime.now() - pipeline_start
        print(f"\nPipeline completed in {pipeline_duration.total_seconds():.2f}s")
        
        return self.results
    
    def generate_pipeline_report(self):
        """Generate pipeline execution report"""
        print("\n6.2 Pipeline Execution Report")
        print("-" * 35)
        
        total_steps = len(self.validation_steps)
        successful_steps = sum(1 for result in self.results.values() if result['status'] == 'success')
        failed_steps = total_steps - successful_steps
        
        print(f"Total Validation Steps: {total_steps}")
        print(f"Successful Steps: {successful_steps}")
        print(f"Failed Steps: {failed_steps}")
        print(f"Success Rate: {(successful_steps / total_steps) * 100:.1f}%")
        
        print(f"\nStep Details:")
        print(f"{'Step':<30} {'Status':<10} {'Duration (s)':<12}")
        print("-" * 55)
        
        for step_name, result in self.results.items():
            status = result['status']
            duration = result['duration']
            print(f"{step_name[:29]:<30} {status:<10} {duration:<12.2f}")
        
        return self.results

# Define validation functions for pipeline
def validate_missing_data(df, threshold=0.1):
    """Validate missing data doesn't exceed threshold"""
    missing_rates = df.isnull().sum() / len(df)
    violations = missing_rates[missing_rates > threshold]
    return {'violations': len(violations), 'details': violations.to_dict()}

def validate_data_types(df, expected_types):
    """Validate data types match expectations"""
    type_violations = []
    for column, expected_type in expected_types.items():
        if column in df.columns:
            actual_type = str(df[column].dtype)
            if expected_type not in actual_type:
                type_violations.append({'column': column, 'expected': expected_type, 'actual': actual_type})
    return {'violations': len(type_violations), 'details': type_violations}

def validate_range_constraints(df, constraints):
    """Validate numeric ranges"""
    violations = []
    for column, constraint in constraints.items():
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            out_of_range = df[(df[column] < constraint['min']) | (df[column] > constraint['max'])]
            if len(out_of_range) > 0:
                violations.append({'column': column, 'count': len(out_of_range)})
    return {'violations': len(violations), 'details': violations}

# Create and run validation pipeline
pipeline = AutomatedValidationPipeline()

# Add validation steps
pipeline.add_validation_step(
    'Missing Data Validation',
    validate_missing_data,
    threshold=0.2
)

pipeline.add_validation_step(
    'Data Type Validation',
    validate_data_types,
    expected_types=expected_types
)

pipeline.add_validation_step(
    'Range Constraint Validation',
    validate_range_constraints,
    constraints=range_constraints
)

# Run pipeline
pipeline_results = pipeline.run_pipeline(df)

# Generate pipeline report
pipeline_report = pipeline.generate_pipeline_report()

# ===============================================================================
# 7. DATA VALIDATION SUMMARY AND RECOMMENDATIONS
# ===============================================================================

print("\n" + "="*60)
print("7. DATA VALIDATION SUMMARY AND RECOMMENDATIONS")
print("="*60)

print("7.1 Validation Summary")
print("-" * 25)

# Summarize all validation results
validation_summary = {
    'Dataset Info': {
        'Total Records': len(df),
        'Total Columns': len(df.columns),
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    },
    'Data Quality Scores': {
        'Overall Score': f"{overall_score:.3f}",
        'Overall Grade': quality_scorer.get_quality_grade(overall_score),
        'Completeness': f"{completeness_score:.3f}",
        'Validity': f"{validity_score:.3f}",
        'Uniqueness': f"{uniqueness_score:.3f}",
        'Consistency': f"{consistency_score:.3f}"
    },
    'Key Issues Found': {
        'Email Format Issues': email_issues.get('invalid_count', 0),
        'Phone Format Issues': phone_issues.get('invalid_count', 0),
        'Range Violations': sum(len(issues.get('sample_violations', [])) for issues in range_issues.values()),
        'Primary Key Duplicates': primary_key_results.get('duplicate_count', 0),
        'Consistency Violations': sum(result['violation_count'] for result in consistency_results.values())
    }
}

for category, metrics in validation_summary.items():
    print(f"\n{category}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

print("\n7.2 Recommendations")
print("-" * 25)

recommendations = []

# Generate recommendations based on findings
if completeness_score < 0.9:
    recommendations.append("• Address missing data issues - consider imputation or data collection improvements")

if validity_score < 0.8:
    recommendations.append("• Implement data validation at source to prevent invalid data entry")

if uniqueness_score < 0.95:
    recommendations.append("• Investigate and resolve duplicate records in primary key columns")

if consistency_score < 0.9:
    recommendations.append("• Review and fix cross-column consistency issues")

if email_issues.get('invalid_count', 0) > 0:
    recommendations.append("• Implement email validation in data entry systems")

if phone_issues.get('invalid_count', 0) > 0:
    recommendations.append("• Standardize phone number format and validation")

if len(range_issues) > 0:
    recommendations.append("• Review and correct out-of-range values in numeric columns")

# General recommendations
recommendations.extend([
    "• Establish data quality monitoring and alerting",
    "• Implement automated data validation in ETL pipelines",
    "• Create data quality dashboards for ongoing monitoring",
    "• Develop data quality standards and documentation",
    "• Train data entry personnel on quality standards"
])

print("Based on the validation results, here are the key recommendations:")
for rec in recommendations:
    print(rec)

print(f"\nData validation analysis complete!")
print(f"Overall data quality grade: {quality_scorer.get_quality_grade(overall_score)}")
print(f"Priority: {'High' if overall_score < 0.7 else 'Medium' if overall_score < 0.9 else 'Low'} - Focus on improving data quality processes")
