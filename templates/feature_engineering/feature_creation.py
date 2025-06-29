"""
===============================================================================
FEATURE CREATION TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive template for creating new features from existing data

This template covers:
- Mathematical transformations
- Binning and discretization
- Interaction features
- Domain-specific feature creation
- Text and categorical feature engineering
- Time-based feature extraction

Prerequisites:
- pandas, numpy, matplotlib, seaborn, scikit-learn
- Dataset with various feature types loaded as 'df'
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# LOAD AND EXAMINE DATA
# ===============================================================================

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Sample dataset creation for demonstration
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'years_experience': np.random.randint(0, 40, 1000),
    'education_years': np.random.randint(12, 20, 1000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'text_field': ['sample text ' + str(i) for i in range(1000)],
    'date_field': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'binary_flag': np.random.choice([0, 1], 1000),
    'score1': np.random.normal(100, 15, 1000),
    'score2': np.random.normal(85, 10, 1000)
})

print("Original Dataset Shape:", df.shape)
print("\nOriginal Features:")
print(df.dtypes)

# ===============================================================================
# 1. MATHEMATICAL TRANSFORMATIONS
# ===============================================================================

print("\n" + "="*50)
print("1. MATHEMATICAL TRANSFORMATIONS")
print("="*50)

# Log transformations (for skewed data)
df['income_log'] = np.log1p(df['income'])  # log1p for handling zeros

# Square root transformation
df['age_sqrt'] = np.sqrt(df['age'])

# Square transformation
df['experience_squared'] = df['years_experience'] ** 2

# Reciprocal transformation
df['income_reciprocal'] = 1 / (df['income'] + 1)  # Adding 1 to avoid division by zero

# Power transformations
df['age_cubed'] = df['age'] ** 3

# Normalize/standardize features
df['income_normalized'] = (df['income'] - df['income'].mean()) / df['income'].std()

print("Mathematical transformations created:")
print("- income_log, age_sqrt, experience_squared")
print("- income_reciprocal, age_cubed, income_normalized")

# ===============================================================================
# 2. BINNING AND DISCRETIZATION
# ===============================================================================

print("\n" + "="*50)
print("2. BINNING AND DISCRETIZATION")
print("="*50)

# Equal-width binning
df['age_bins_equal'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Mature', 'Senior'])

# Equal-frequency binning (quantile-based)
df['income_quartiles'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Custom binning with specific thresholds
age_bins = [0, 25, 35, 50, 65, 100]
df['age_groups'] = pd.cut(df['age'], bins=age_bins, labels=['Young Adult', 'Early Career', 'Mid Career', 'Late Career', 'Senior'])

# KBins discretization from sklearn
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['experience_bins'] = discretizer.fit_transform(df[['years_experience']]).flatten()

print("Binning features created:")
print("- age_bins_equal, income_quartiles, age_groups, experience_bins")

# ===============================================================================
# 3. INTERACTION FEATURES
# ===============================================================================

print("\n" + "="*50)
print("3. INTERACTION FEATURES")
print("="*50)

# Simple multiplication interactions
df['age_income_interaction'] = df['age'] * df['income']
df['experience_education_interaction'] = df['years_experience'] * df['education_years']

# Division interactions (ratios)
df['income_per_year_experience'] = df['income'] / (df['years_experience'] + 1)
df['education_age_ratio'] = df['education_years'] / df['age']

# Polynomial features using sklearn
poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
numerical_cols = ['age', 'income', 'years_experience']
poly_array = poly_features.fit_transform(df[numerical_cols])
poly_feature_names = poly_features.get_feature_names_out(numerical_cols)

# Add polynomial features to dataframe
for i, name in enumerate(poly_feature_names):
    if name not in numerical_cols:  # Skip original features
        df[f'poly_{name}'] = poly_array[:, i]

print("Interaction features created:")
print("- age_income_interaction, experience_education_interaction")
print("- income_per_year_experience, education_age_ratio")
print("- Polynomial interaction features")

# ===============================================================================
# 4. AGGREGATION AND ROLLING FEATURES
# ===============================================================================

print("\n" + "="*50)
print("4. AGGREGATION AND ROLLING FEATURES")
print("="*50)

# Group-based aggregations
category_stats = df.groupby('category')['income'].agg(['mean', 'std', 'median']).reset_index()
category_stats.columns = ['category', 'category_income_mean', 'category_income_std', 'category_income_median']
df = df.merge(category_stats, on='category', how='left')

# Create deviation from group mean
df['income_deviation_from_category'] = df['income'] - df['category_income_mean']

# Ranking within groups
df['income_rank_within_category'] = df.groupby('category')['income'].rank(pct=True)

# Moving averages (for time series or ordered data)
df = df.sort_values('date_field')
df['income_ma_7'] = df['income'].rolling(window=7, min_periods=1).mean()
df['income_std_7'] = df['income'].rolling(window=7, min_periods=1).std()

print("Aggregation features created:")
print("- category_income_mean/std/median")
print("- income_deviation_from_category, income_rank_within_category")
print("- income_ma_7, income_std_7")

# ===============================================================================
# 5. TIME-BASED FEATURES
# ===============================================================================

print("\n" + "="*50)
print("5. TIME-BASED FEATURES")
print("="*50)

# Extract date components
df['year'] = df['date_field'].dt.year
df['month'] = df['date_field'].dt.month
df['day'] = df['date_field'].dt.day
df['dayofweek'] = df['date_field'].dt.dayofweek
df['quarter'] = df['date_field'].dt.quarter
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Create cyclical features for better ML performance
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

# Time since reference date
reference_date = df['date_field'].min()
df['days_since_start'] = (df['date_field'] - reference_date).dt.days

print("Time-based features created:")
print("- year, month, day, dayofweek, quarter, is_weekend")
print("- month_sin/cos, day_sin/cos (cyclical)")
print("- days_since_start")

# ===============================================================================
# 6. TEXT FEATURE ENGINEERING
# ===============================================================================

print("\n" + "="*50)
print("6. TEXT FEATURE ENGINEERING")
print("="*50)

# Basic text statistics
df['text_length'] = df['text_field'].str.len()
df['text_word_count'] = df['text_field'].str.split().str.len()
df['text_unique_words'] = df['text_field'].apply(lambda x: len(set(x.split())))

# Character-based features
df['text_digit_count'] = df['text_field'].str.count(r'\d')
df['text_upper_count'] = df['text_field'].str.count(r'[A-Z]')
df['text_special_char_count'] = df['text_field'].str.count(r'[!@#$%^&*(),.?":{}|<>]')

# TF-IDF features (example with top 5 features)
tfidf = TfidfVectorizer(max_features=5, stop_words='english')
tfidf_features = tfidf.fit_transform(df['text_field'])
tfidf_feature_names = [f'tfidf_{name}' for name in tfidf.get_feature_names_out()]

for i, name in enumerate(tfidf_feature_names):
    df[name] = tfidf_features[:, i].toarray().flatten()

print("Text features created:")
print("- text_length, text_word_count, text_unique_words")
print("- text_digit/upper/special_char_count")
print("- TF-IDF features")

# ===============================================================================
# 7. CATEGORICAL FEATURE ENGINEERING
# ===============================================================================

print("\n" + "="*50)
print("7. CATEGORICAL FEATURE ENGINEERING")
print("="*50)

# Frequency encoding
category_counts = df['category'].value_counts().to_dict()
df['category_frequency'] = df['category'].map(category_counts)

# Target encoding (using income as target for demonstration)
target_mean = df.groupby('category')['income'].mean().to_dict()
df['category_target_encoded'] = df['category'].map(target_mean)

# One-hot encoding
category_dummies = pd.get_dummies(df['category'], prefix='category')
df = pd.concat([df, category_dummies], axis=1)

# Label encoding (ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_label_encoded'] = le.fit_transform(df['category'])

print("Categorical features created:")
print("- category_frequency, category_target_encoded")
print("- One-hot encoded categories")
print("- category_label_encoded")

# ===============================================================================
# 8. COMPOSITE AND DOMAIN-SPECIFIC FEATURES
# ===============================================================================

print("\n" + "="*50)
print("8. COMPOSITE AND DOMAIN-SPECIFIC FEATURES")
print("="*50)

# Create composite scores
df['total_score'] = df['score1'] + df['score2']
df['score_difference'] = df['score1'] - df['score2']
df['score_ratio'] = df['score1'] / (df['score2'] + 1)
df['average_score'] = (df['score1'] + df['score2']) / 2

# Domain-specific features (example: employment/career related)
df['experience_per_age'] = df['years_experience'] / df['age']
df['education_premium'] = df['education_years'] - 12  # Years beyond high school
df['career_progression'] = df['income'] / (df['years_experience'] + 1)

# Create flags/indicators
df['high_performer'] = ((df['score1'] > df['score1'].quantile(0.75)) & 
                       (df['score2'] > df['score2'].quantile(0.75))).astype(int)
df['experienced_worker'] = (df['years_experience'] > df['years_experience'].median()).astype(int)
df['high_earner'] = (df['income'] > df['income'].quantile(0.8)).astype(int)

print("Composite features created:")
print("- total_score, score_difference, score_ratio, average_score")
print("- experience_per_age, education_premium, career_progression")
print("- high_performer, experienced_worker, high_earner flags")

# ===============================================================================
# 9. FEATURE SELECTION AND VALIDATION
# ===============================================================================

print("\n" + "="*50)
print("9. FEATURE SUMMARY AND VALIDATION")
print("="*50)

# Get all numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Total features after engineering: {len(df.columns)}")
print(f"Original features: {len(['age', 'income', 'years_experience', 'education_years', 'category', 'text_field', 'date_field', 'binary_flag', 'score1', 'score2'])}")
print(f"New features created: {len(df.columns) - 10}")

# Check for missing values in new features
missing_summary = df.isnull().sum()
missing_features = missing_summary[missing_summary > 0]
if len(missing_features) > 0:
    print("\nFeatures with missing values:")
    print(missing_features)
else:
    print("\nNo missing values in engineered features ✓")

# Display correlation with original features
if len(numeric_cols) > 0:
    correlation_matrix = df[numeric_cols[:15]].corr()  # Limit to first 15 for display
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Original and Engineered Features')
    plt.tight_layout()
    plt.show()

# ===============================================================================
# 10. FEATURE IMPORTANCE AND SELECTION
# ===============================================================================

print("\n" + "="*50)
print("10. FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Example using Random Forest for feature importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare data for feature importance analysis
feature_cols = [col for col in numeric_cols if col != 'income']  # Use income as target
X = df[feature_cols].fillna(0)  # Fill any NaN values
y = df['income']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importance Scores')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ===============================================================================
# 11. EXPORT ENGINEERED FEATURES
# ===============================================================================

print("\n" + "="*50)
print("11. EXPORT RESULTS")
print("="*50)

# Save engineered dataset
# df.to_csv('engineered_features_dataset.csv', index=False)

# Create feature documentation
feature_documentation = {
    'Mathematical Transformations': ['income_log', 'age_sqrt', 'experience_squared', 'income_reciprocal', 'age_cubed', 'income_normalized'],
    'Binning Features': ['age_bins_equal', 'income_quartiles', 'age_groups', 'experience_bins'],
    'Interaction Features': ['age_income_interaction', 'experience_education_interaction', 'income_per_year_experience', 'education_age_ratio'],
    'Aggregation Features': ['category_income_mean', 'category_income_std', 'income_deviation_from_category', 'income_rank_within_category'],
    'Time Features': ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'month_sin', 'month_cos', 'days_since_start'],
    'Text Features': ['text_length', 'text_word_count', 'text_unique_words', 'text_digit_count'],
    'Categorical Features': ['category_frequency', 'category_target_encoded', 'category_label_encoded'],
    'Composite Features': ['total_score', 'score_difference', 'experience_per_age', 'high_performer']
}

print("Feature Engineering Complete!")
print("\nFeature Categories Created:")
for category, features in feature_documentation.items():
    print(f"- {category}: {len(features)} features")

print(f"\nFinal dataset shape: {df.shape}")
print("Dataset ready for modeling!")
