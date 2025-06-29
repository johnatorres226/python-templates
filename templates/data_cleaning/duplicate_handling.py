"""
===============================================================================
DUPLICATE HANDLING TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive duplicate detection and handling strategies

This template covers:
- Duplicate detection methods and strategies
- Exact and fuzzy duplicate matching
- Similarity-based duplicate identification
- Duplicate resolution strategies
- Record linkage and deduplication
- Quality assessment of duplicates
- Automated deduplication pipelines

Prerequisites:
- pandas, numpy, matplotlib, seaborn
- Optional: fuzzywuzzy, recordlinkage (pip install fuzzywuzzy recordlinkage)
- Dataset loaded as 'df' for duplicate analysis
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Advanced duplicate detection libraries (install if needed)
try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("fuzzywuzzy not available. Install with: pip install fuzzywuzzy")

try:
    import recordlinkage as rl
    RECORDLINKAGE_AVAILABLE = True
except ImportError:
    RECORDLINKAGE_AVAILABLE = False
    print("recordlinkage not available. Install with: pip install recordlinkage")

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===============================================================================
# CREATE SAMPLE DATA WITH DUPLICATES
# ===============================================================================

def create_sample_data_with_duplicates():
    """Create sample dataset with various types of duplicates"""
    n_samples = 1000
    
    # Generate base data
    companies = ['TechCorp', 'DataSys', 'InnovateLLC', 'GlobalTech', 'SmartSolutions']
    cities = ['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle']
    domains = ['gmail.com', 'yahoo.com', 'company.com', 'outlook.com', 'tech.com']
    
    data = {
        'id': range(1, n_samples + 1),
        'first_name': np.random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Tom', 'Anna'], n_samples),
        'last_name': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis'], n_samples),
        'company': np.random.choice(companies, n_samples),
        'city': np.random.choice(cities, n_samples),
        'age': np.random.randint(25, 65, n_samples),
        'salary': np.random.randint(40000, 120000, n_samples),
        'phone': [f'({np.random.randint(200, 999)}) {np.random.randint(200, 999)}-{np.random.randint(1000, 9999)}' 
                 for _ in range(n_samples)]
    }
    
    # Generate emails
    emails = []
    for i in range(n_samples):
        first = data['first_name'][i].lower()
        last = data['last_name'][i].lower()
        domain = np.random.choice(domains)
        email = f"{first}.{last}@{domain}"
        emails.append(email)
    
    data['email'] = emails
    
    df = pd.DataFrame(data)
    
    # Introduce different types of duplicates
    
    # 1. Exact duplicates (5% of data)
    exact_duplicate_count = int(0.05 * n_samples)
    exact_indices = np.random.choice(n_samples, size=exact_duplicate_count, replace=False)
    
    for idx in exact_indices:
        # Create exact duplicate with new ID
        duplicate_row = df.iloc[idx].copy()
        duplicate_row['id'] = n_samples + idx + 1
        df = pd.concat([df, duplicate_row.to_frame().T], ignore_index=True)
    
    # 2. Near duplicates with minor variations (3% of data)
    near_duplicate_count = int(0.03 * n_samples)
    near_indices = np.random.choice(n_samples, size=near_duplicate_count, replace=False)
    
    for idx in near_indices:
        duplicate_row = df.iloc[idx].copy()
        duplicate_row['id'] = n_samples + exact_duplicate_count + idx + 1
        
        # Introduce minor variations
        variations = np.random.choice([
            'name_variation', 'email_variation', 'phone_variation', 'company_variation'
        ], size=1)[0]
        
        if variations == 'name_variation':
            # Add/remove middle initial, change case
            if np.random.random() > 0.5:
                duplicate_row['first_name'] = duplicate_row['first_name'].upper()
            else:
                duplicate_row['last_name'] = duplicate_row['last_name'] + 'son'
        
        elif variations == 'email_variation':
            # Change email format slightly
            original_email = duplicate_row['email']
            name_part = original_email.split('@')[0]
            domain_part = original_email.split('@')[1]
            
            if '.' in name_part:
                # Change separator
                new_name_part = name_part.replace('.', '_')
            else:
                new_name_part = name_part + '123'
            
            duplicate_row['email'] = f"{new_name_part}@{domain_part}"
        
        elif variations == 'phone_variation':
            # Change phone format
            phone = duplicate_row['phone']
            # Remove formatting
            digits = re.sub(r'[^\d]', '', phone)
            # Add different formatting
            duplicate_row['phone'] = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        
        elif variations == 'company_variation':
            # Add/remove company suffixes
            company = duplicate_row['company']
            if 'LLC' in company:
                duplicate_row['company'] = company.replace('LLC', 'Inc')
            else:
                duplicate_row['company'] = company + ' Inc'
        
        df = pd.concat([df, duplicate_row.to_frame().T], ignore_index=True)
    
    # 3. Fuzzy duplicates with more significant variations (2% of data)
    fuzzy_duplicate_count = int(0.02 * n_samples)
    fuzzy_indices = np.random.choice(n_samples, size=fuzzy_duplicate_count, replace=False)
    
    for idx in fuzzy_indices:
        duplicate_row = df.iloc[idx].copy()
        duplicate_row['id'] = n_samples + exact_duplicate_count + near_duplicate_count + idx + 1
        
        # Introduce typos and variations
        if np.random.random() > 0.5:
            # Add typo to name
            name = duplicate_row['first_name']
            if len(name) > 3:
                pos = np.random.randint(1, len(name) - 1)
                name_list = list(name)
                name_list[pos] = chr(ord(name_list[pos]) + 1)  # Change character
                duplicate_row['first_name'] = ''.join(name_list)
        
        # Change age slightly (data entry error)
        duplicate_row['age'] = duplicate_row['age'] + np.random.randint(-2, 3)
        
        # Change salary slightly
        duplicate_row['salary'] = duplicate_row['salary'] + np.random.randint(-5000, 5000)
        
        df = pd.concat([df, duplicate_row.to_frame().T], ignore_index=True)
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Create sample data for demonstration
df = create_sample_data_with_duplicates()

print("Sample Dataset with Duplicates Created:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# ===============================================================================
# 1. EXACT DUPLICATE DETECTION
# ===============================================================================

print("\n" + "="*60)
print("1. EXACT DUPLICATE DETECTION")
print("="*60)

class ExactDuplicateDetector:
    """Class for detecting exact duplicates"""
    
    def __init__(self, df):
        self.df = df
        self.duplicate_results = {}
    
    def detect_complete_duplicates(self, exclude_columns=None):
        """Detect completely identical rows"""
        print("1.1 Complete Row Duplicates")
        print("-" * 30)
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Select columns for duplicate detection
        columns_to_check = [col for col in self.df.columns if col not in exclude_columns]
        
        # Find duplicates
        duplicate_mask = self.df.duplicated(subset=columns_to_check, keep=False)
        duplicates = self.df[duplicate_mask]
        
        duplicate_count = len(duplicates)
        unique_duplicate_groups = len(duplicates.drop_duplicates(subset=columns_to_check))
        
        self.duplicate_results['complete_duplicates'] = {
            'total_duplicates': duplicate_count,
            'unique_groups': unique_duplicate_groups,
            'duplicate_rate': duplicate_count / len(self.df),
            'duplicates_df': duplicates
        }
        
        print(f"Total duplicate rows: {duplicate_count}")
        print(f"Unique duplicate groups: {unique_duplicate_groups}")
        print(f"Duplicate rate: {(duplicate_count / len(self.df)):.2%}")
        
        if duplicate_count > 0:
            print("\nSample duplicate groups:")
            # Show first few duplicate groups
            for i, (_, group) in enumerate(duplicates.groupby(columns_to_check)):
                if i >= 3:  # Show only first 3 groups
                    break
                print(f"\nGroup {i+1} ({len(group)} records):")
                print(group[['first_name', 'last_name', 'email', 'company']].head(2))
        
        return duplicates
    
    def detect_column_specific_duplicates(self, columns):
        """Detect duplicates based on specific columns"""
        print(f"\n1.2 Column-Specific Duplicates: {columns}")
        print("-" * 50)
        
        if isinstance(columns, str):
            columns = [columns]
        
        # Check if columns exist
        missing_columns = [col for col in columns if col not in self.df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return pd.DataFrame()
        
        # Find duplicates
        duplicate_mask = self.df.duplicated(subset=columns, keep=False)
        duplicates = self.df[duplicate_mask]
        
        duplicate_count = len(duplicates)
        unique_groups = self.df[columns].duplicated(keep=False).sum()
        
        print(f"Duplicate rows based on {columns}: {duplicate_count}")
        print(f"Duplicate rate: {(duplicate_count / len(self.df)):.2%}")
        
        # Show value counts for duplicate columns
        if duplicate_count > 0:
            print(f"\nMost common duplicate values:")
            duplicate_values = self.df[duplicate_mask].groupby(columns).size().sort_values(ascending=False)
            print(duplicate_values.head(10))
        
        self.duplicate_results[f"duplicates_{'+'.join(columns)}"] = {
            'total_duplicates': duplicate_count,
            'duplicate_rate': duplicate_count / len(self.df),
            'duplicates_df': duplicates
        }
        
        return duplicates
    
    def analyze_duplicate_patterns(self):
        """Analyze patterns in duplicate data"""
        print("\n1.3 Duplicate Pattern Analysis")
        print("-" * 35)
        
        # Check duplicates by different column combinations
        column_combinations = [
            ['email'],
            ['first_name', 'last_name'],
            ['phone'],
            ['first_name', 'last_name', 'company'],
            ['email', 'phone']
        ]
        
        pattern_results = {}
        
        for columns in column_combinations:
            if all(col in self.df.columns for col in columns):
                duplicate_count = self.df.duplicated(subset=columns, keep=False).sum()
                pattern_results['+'.join(columns)] = {
                    'count': duplicate_count,
                    'rate': duplicate_count / len(self.df)
                }
        
        # Display results
        print("Duplicate patterns by column combination:")
        print(f"{'Columns':<25} {'Count':<8} {'Rate':<8}")
        print("-" * 45)
        
        for pattern, stats in sorted(pattern_results.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"{pattern:<25} {stats['count']:<8} {stats['rate']:<8.2%}")
        
        return pattern_results

# Initialize exact duplicate detector
exact_detector = ExactDuplicateDetector(df)

# Detect complete duplicates (excluding ID column)
complete_duplicates = exact_detector.detect_complete_duplicates(exclude_columns=['id'])

# Detect duplicates by specific columns
email_duplicates = exact_detector.detect_column_specific_duplicates(['email'])
name_duplicates = exact_detector.detect_column_specific_duplicates(['first_name', 'last_name'])
phone_duplicates = exact_detector.detect_column_specific_duplicates(['phone'])

# Analyze duplicate patterns
duplicate_patterns = exact_detector.analyze_duplicate_patterns()

# ===============================================================================
# 2. FUZZY DUPLICATE DETECTION
# ===============================================================================

if FUZZYWUZZY_AVAILABLE:
    print("\n" + "="*60)
    print("2. FUZZY DUPLICATE DETECTION")
    print("="*60)
    
    class FuzzyDuplicateDetector:
        """Class for detecting fuzzy/similar duplicates"""
        
        def __init__(self, df):
            self.df = df
            self.fuzzy_results = {}
        
        def detect_fuzzy_name_duplicates(self, threshold=85):
            """Detect similar names using fuzzy matching"""
            print(f"2.1 Fuzzy Name Matching (threshold: {threshold})")
            print("-" * 45)
            
            # Combine first and last names
            full_names = (self.df['first_name'] + ' ' + self.df['last_name']).dropna()
            
            fuzzy_matches = []
            processed_names = set()
            
            for i, name1 in enumerate(full_names):
                if name1 in processed_names:
                    continue
                
                matches = []
                for j, name2 in enumerate(full_names):
                    if i != j and name2 not in processed_names:
                        similarity = fuzz.ratio(name1.lower(), name2.lower())
                        if similarity >= threshold:
                            matches.append({
                                'index1': i,
                                'index2': j,
                                'name1': name1,
                                'name2': name2,
                                'similarity': similarity
                            })
                
                if matches:
                    fuzzy_matches.extend(matches)
                    processed_names.add(name1)
                    processed_names.update([match['name2'] for match in matches])
            
            print(f"Found {len(fuzzy_matches)} fuzzy name matches")
            
            if fuzzy_matches:
                print("\nSample fuzzy matches:")
                for match in fuzzy_matches[:5]:
                    print(f"  '{match['name1']}' ~ '{match['name2']}' (similarity: {match['similarity']})")
            
            self.fuzzy_results['name_matches'] = fuzzy_matches
            return fuzzy_matches
        
        def detect_fuzzy_email_duplicates(self, threshold=80):
            """Detect similar email addresses"""
            print(f"\n2.2 Fuzzy Email Matching (threshold: {threshold})")
            print("-" * 45)
            
            emails = self.df['email'].dropna().unique()
            
            fuzzy_email_matches = []
            processed_emails = set()
            
            for i, email1 in enumerate(emails):
                if email1 in processed_emails:
                    continue
                
                for j, email2 in enumerate(emails):
                    if i != j and email2 not in processed_emails:
                        # Compare email local parts (before @)
                        local1 = email1.split('@')[0]
                        local2 = email2.split('@')[0]
                        domain1 = email1.split('@')[1]
                        domain2 = email2.split('@')[1]
                        
                        # Only compare if domains are the same
                        if domain1 == domain2:
                            similarity = fuzz.ratio(local1.lower(), local2.lower())
                            if similarity >= threshold:
                                fuzzy_email_matches.append({
                                    'email1': email1,
                                    'email2': email2,
                                    'similarity': similarity
                                })
                                processed_emails.add(email2)
                
                processed_emails.add(email1)
            
            print(f"Found {len(fuzzy_email_matches)} fuzzy email matches")
            
            if fuzzy_email_matches:
                print("\nSample fuzzy email matches:")
                for match in fuzzy_email_matches[:5]:
                    print(f"  '{match['email1']}' ~ '{match['email2']}' (similarity: {match['similarity']})")
            
            self.fuzzy_results['email_matches'] = fuzzy_email_matches
            return fuzzy_email_matches
        
        def detect_fuzzy_company_duplicates(self, threshold=75):
            """Detect similar company names"""
            print(f"\n2.3 Fuzzy Company Matching (threshold: {threshold})")
            print("-" * 48)
            
            companies = self.df['company'].dropna().unique()
            
            fuzzy_company_matches = []
            processed_companies = set()
            
            for i, company1 in enumerate(companies):
                if company1 in processed_companies:
                    continue
                
                for j, company2 in enumerate(companies):
                    if i != j and company2 not in processed_companies:
                        similarity = fuzz.ratio(company1.lower(), company2.lower())
                        if similarity >= threshold:
                            fuzzy_company_matches.append({
                                'company1': company1,
                                'company2': company2,
                                'similarity': similarity
                            })
                            processed_companies.add(company2)
                
                processed_companies.add(company1)
            
            print(f"Found {len(fuzzy_company_matches)} fuzzy company matches")
            
            if fuzzy_company_matches:
                print("\nSample fuzzy company matches:")
                for match in fuzzy_company_matches[:5]:
                    print(f"  '{match['company1']}' ~ '{match['company2']}' (similarity: {match['similarity']})")
            
            self.fuzzy_results['company_matches'] = fuzzy_company_matches
            return fuzzy_company_matches
        
        def comprehensive_fuzzy_matching(self, name_threshold=85, email_threshold=80, company_threshold=75):
            """Perform comprehensive fuzzy matching across all fields"""
            print(f"\n2.4 Comprehensive Fuzzy Matching")
            print("-" * 40)
            
            # Combine all fuzzy matching results
            potential_duplicates = []
            
            # Create a scoring system for potential matches
            for i in range(len(self.df)):
                for j in range(i + 1, len(self.df)):
                    row1 = self.df.iloc[i]
                    row2 = self.df.iloc[j]
                    
                    match_score = 0
                    match_details = {}
                    
                    # Name similarity
                    name1 = f"{row1['first_name']} {row1['last_name']}"
                    name2 = f"{row2['first_name']} {row2['last_name']}"
                    name_sim = fuzz.ratio(name1.lower(), name2.lower())
                    if name_sim >= name_threshold:
                        match_score += name_sim * 0.4
                        match_details['name_similarity'] = name_sim
                    
                    # Email similarity
                    if pd.notna(row1['email']) and pd.notna(row2['email']):
                        email_sim = fuzz.ratio(row1['email'].lower(), row2['email'].lower())
                        if email_sim >= email_threshold:
                            match_score += email_sim * 0.3
                            match_details['email_similarity'] = email_sim
                    
                    # Company similarity
                    if pd.notna(row1['company']) and pd.notna(row2['company']):
                        company_sim = fuzz.ratio(row1['company'].lower(), row2['company'].lower())
                        if company_sim >= company_threshold:
                            match_score += company_sim * 0.2
                            match_details['company_similarity'] = company_sim
                    
                    # Age and salary proximity
                    if abs(row1['age'] - row2['age']) <= 2:
                        match_score += 10
                        match_details['age_match'] = True
                    
                    if abs(row1['salary'] - row2['salary']) <= 5000:
                        match_score += 10
                        match_details['salary_match'] = True
                    
                    # If overall match score is high enough, consider it a potential duplicate
                    if match_score >= 50:
                        potential_duplicates.append({
                            'index1': i,
                            'index2': j,
                            'record1': row1,
                            'record2': row2,
                            'match_score': match_score,
                            'match_details': match_details
                        })
            
            # Sort by match score
            potential_duplicates.sort(key=lambda x: x['match_score'], reverse=True)
            
            print(f"Found {len(potential_duplicates)} potential fuzzy duplicates")
            
            if potential_duplicates:
                print(f"\nTop 5 potential duplicates:")
                for i, match in enumerate(potential_duplicates[:5]):
                    print(f"\nMatch {i+1} (Score: {match['match_score']:.1f}):")
                    print(f"  Record 1: {match['record1']['first_name']} {match['record1']['last_name']}, {match['record1']['email']}")
                    print(f"  Record 2: {match['record2']['first_name']} {match['record2']['last_name']}, {match['record2']['email']}")
                    print(f"  Details: {match['match_details']}")
            
            self.fuzzy_results['comprehensive_matches'] = potential_duplicates
            return potential_duplicates
    
    # Initialize fuzzy duplicate detector
    fuzzy_detector = FuzzyDuplicateDetector(df)
    
    # Detect fuzzy duplicates
    fuzzy_name_matches = fuzzy_detector.detect_fuzzy_name_duplicates(threshold=85)
    fuzzy_email_matches = fuzzy_detector.detect_fuzzy_email_duplicates(threshold=80)
    fuzzy_company_matches = fuzzy_detector.detect_fuzzy_company_duplicates(threshold=75)
    comprehensive_matches = fuzzy_detector.comprehensive_fuzzy_matching()

else:
    print("\n" + "="*60)
    print("2. FUZZY DUPLICATE DETECTION (UNAVAILABLE)")
    print("="*60)
    print("fuzzywuzzy library not available. Install with: pip install fuzzywuzzy")

# ===============================================================================
# 3. STATISTICAL DUPLICATE DETECTION
# ===============================================================================

print("\n" + "="*60)
print("3. STATISTICAL DUPLICATE DETECTION")
print("="*60)

class StatisticalDuplicateDetector:
    """Class for statistical duplicate detection methods"""
    
    def __init__(self, df):
        self.df = df
        self.statistical_results = {}
    
    def detect_outlier_based_duplicates(self):
        """Detect duplicates based on statistical outliers"""
        print("3.1 Outlier-Based Duplicate Detection")
        print("-" * 40)
        
        # Calculate frequency of each combination of categorical variables
        categorical_cols = ['first_name', 'last_name', 'company', 'city']
        existing_cols = [col for col in categorical_cols if col in self.df.columns]
        
        if not existing_cols:
            print("No categorical columns found for analysis")
            return pd.DataFrame()
        
        # Count occurrences of each combination
        combination_counts = self.df.groupby(existing_cols).size().reset_index(name='count')
        
        # Statistical analysis of counts
        mean_count = combination_counts['count'].mean()
        std_count = combination_counts['count'].std()
        threshold = mean_count + 2 * std_count  # 2 standard deviations above mean
        
        # Find statistical outliers (unusually high frequencies)
        outlier_combinations = combination_counts[combination_counts['count'] > threshold]
        
        print(f"Mean occurrence count: {mean_count:.2f}")
        print(f"Standard deviation: {std_count:.2f}")
        print(f"Outlier threshold (mean + 2*std): {threshold:.2f}")
        print(f"Number of outlier combinations: {len(outlier_combinations)}")
        
        if len(outlier_combinations) > 0:
            print(f"\nTop outlier combinations:")
            print(outlier_combinations.sort_values('count', ascending=False).head())
            
            # Get the actual records for these outlier combinations
            outlier_records = []
            for _, row in outlier_combinations.iterrows():
                condition = True
                for col in existing_cols:
                    condition &= (self.df[col] == row[col])
                outlier_records.append(self.df[condition])
            
            if outlier_records:
                all_outlier_records = pd.concat(outlier_records, ignore_index=True)
                self.statistical_results['outlier_duplicates'] = all_outlier_records
                return all_outlier_records
        
        return pd.DataFrame()
    
    def detect_distribution_anomalies(self):
        """Detect duplicates based on distribution anomalies"""
        print("\n3.2 Distribution Anomaly Detection")
        print("-" * 40)
        
        anomalies = {}
        
        # Analyze numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'id':  # Skip ID column
                col_data = self.df[col].dropna()
                
                # Calculate z-scores
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                
                # Find records with extreme z-scores (potential data entry errors/duplicates)
                extreme_indices = z_scores[z_scores > 3].index
                
                if len(extreme_indices) > 0:
                    anomalies[col] = {
                        'count': len(extreme_indices),
                        'percentage': len(extreme_indices) / len(col_data) * 100,
                        'indices': extreme_indices.tolist()
                    }
        
        print("Distribution anomalies found:")
        for col, stats in anomalies.items():
            print(f"  {col}: {stats['count']} anomalies ({stats['percentage']:.2f}%)")
        
        self.statistical_results['distribution_anomalies'] = anomalies
        return anomalies
    
    def detect_pattern_based_duplicates(self):
        """Detect duplicates based on common patterns"""
        print("\n3.3 Pattern-Based Duplicate Detection")
        print("-" * 42)
        
        pattern_duplicates = {}
        
        # Email domain analysis
        if 'email' in self.df.columns:
            email_domains = self.df['email'].dropna().str.split('@').str[1]
            domain_counts = email_domains.value_counts()
            
            # Look for unusually high concentrations of single domains
            mean_domain_count = domain_counts.mean()
            std_domain_count = domain_counts.std()
            suspicious_domains = domain_counts[domain_counts > mean_domain_count + 2 * std_domain_count]
            
            if len(suspicious_domains) > 0:
                pattern_duplicates['suspicious_email_domains'] = suspicious_domains.to_dict()
                print(f"Suspicious email domains (high frequency):")
                for domain, count in suspicious_domains.head().items():
                    print(f"  {domain}: {count} occurrences")
        
        # Phone number pattern analysis
        if 'phone' in self.df.columns:
            # Extract area codes
            phone_patterns = self.df['phone'].dropna().str.extract(r'(\d{3})')[0]
            pattern_counts = phone_patterns.value_counts()
            
            # Look for unusual concentrations
            mean_pattern_count = pattern_counts.mean()
            suspicious_patterns = pattern_counts[pattern_counts > mean_pattern_count + 2 * pattern_counts.std()]
            
            if len(suspicious_patterns) > 0:
                pattern_duplicates['suspicious_phone_patterns'] = suspicious_patterns.to_dict()
                print(f"\nSuspicious phone area codes (high frequency):")
                for pattern, count in suspicious_patterns.head().items():
                    print(f"  {pattern}: {count} occurrences")
        
        # Name pattern analysis
        if 'first_name' in self.df.columns and 'last_name' in self.df.columns:
            # Look for unusual name combinations
            name_combinations = self.df['first_name'] + ' ' + self.df['last_name']
            name_counts = name_combinations.value_counts()
            
            # Names appearing more than expected
            duplicate_names = name_counts[name_counts > 1]
            if len(duplicate_names) > 0:
                pattern_duplicates['duplicate_names'] = duplicate_names.to_dict()
                print(f"\nDuplicate name combinations:")
                for name, count in duplicate_names.head().items():
                    print(f"  {name}: {count} occurrences")
        
        self.statistical_results['pattern_duplicates'] = pattern_duplicates
        return pattern_duplicates

# Initialize statistical duplicate detector
statistical_detector = StatisticalDuplicateDetector(df)

# Detect statistical duplicates
outlier_duplicates = statistical_detector.detect_outlier_based_duplicates()
distribution_anomalies = statistical_detector.detect_distribution_anomalies()
pattern_duplicates = statistical_detector.detect_pattern_based_duplicates()

# ===============================================================================
# 4. DUPLICATE RESOLUTION STRATEGIES
# ===============================================================================

print("\n" + "="*60)
print("4. DUPLICATE RESOLUTION STRATEGIES")
print("="*60)

class DuplicateResolver:
    """Class for resolving duplicates using various strategies"""
    
    def __init__(self, df):
        self.df = df
        self.resolution_results = {}
    
    def resolve_exact_duplicates(self, subset=None, keep='first'):
        """Resolve exact duplicates using pandas built-in methods"""
        print(f"4.1 Exact Duplicate Resolution (keep='{keep}')")
        print("-" * 45)
        
        original_count = len(self.df)
        
        # Remove exact duplicates
        if subset is None:
            resolved_df = self.df.drop_duplicates(keep=keep)
        else:
            resolved_df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        removed_count = original_count - len(resolved_df)
        
        print(f"Original records: {original_count}")
        print(f"After deduplication: {len(resolved_df)}")
        print(f"Removed duplicates: {removed_count}")
        print(f"Reduction rate: {(removed_count / original_count):.2%}")
        
        self.resolution_results['exact_resolution'] = {
            'original_count': original_count,
            'final_count': len(resolved_df),
            'removed_count': removed_count,
            'method': f'drop_duplicates(keep={keep})'
        }
        
        return resolved_df
    
    def resolve_with_data_quality_priority(self, quality_columns):
        """Resolve duplicates prioritizing records with better data quality"""
        print("\n4.2 Data Quality Priority Resolution")
        print("-" * 42)
        
        # Create a copy for resolution
        df_copy = self.df.copy()
        
        # Calculate data quality score for each record
        def calculate_quality_score(row):
            score = 0
            for col in quality_columns:
                if col in row.index:
                    if pd.notna(row[col]):
                        score += 1
                        # Additional scoring for data completeness
                        if isinstance(row[col], str) and len(str(row[col]).strip()) > 0:
                            score += 0.5
            return score
        
        df_copy['quality_score'] = df_copy.apply(calculate_quality_score, axis=1)
        
        # Group by potential duplicate key and keep highest quality record
        # Using name + email as duplicate key for this example
        if 'first_name' in df_copy.columns and 'last_name' in df_copy.columns and 'email' in df_copy.columns:
            duplicate_key = df_copy['first_name'] + '_' + df_copy['last_name'] + '_' + df_copy['email'].fillna('')
            
            # Keep record with highest quality score in each group
            resolved_df = df_copy.loc[df_copy.groupby(duplicate_key)['quality_score'].idxmax()]
            
            original_count = len(self.df)
            removed_count = original_count - len(resolved_df)
            
            print(f"Original records: {original_count}")
            print(f"After quality-based resolution: {len(resolved_df)}")
            print(f"Removed duplicates: {removed_count}")
            print(f"Quality columns used: {quality_columns}")
            
            # Drop the temporary quality score column
            resolved_df = resolved_df.drop('quality_score', axis=1)
            
            self.resolution_results['quality_resolution'] = {
                'original_count': original_count,
                'final_count': len(resolved_df),
                'removed_count': removed_count,
                'method': 'data_quality_priority'
            }
            
            return resolved_df
        else:
            print("Required columns not found for quality-based resolution")
            return self.df
    
    def resolve_with_record_merging(self, merge_strategy='most_complete'):
        """Resolve duplicates by merging information from duplicate records"""
        print(f"\n4.3 Record Merging Resolution (strategy: {merge_strategy})")
        print("-" * 55)
        
        # Group potential duplicates by name
        if 'first_name' in self.df.columns and 'last_name' in self.df.columns:
            name_groups = self.df.groupby(['first_name', 'last_name'])
            
            merged_records = []
            total_groups = len(name_groups)
            merged_groups = 0
            
            for (first_name, last_name), group in name_groups:
                if len(group) > 1:  # Only process groups with duplicates
                    merged_groups += 1
                    
                    if merge_strategy == 'most_complete':
                        # Create merged record with most complete information
                        merged_record = {}
                        
                        for col in group.columns:
                            if col == 'id':
                                # Keep the first ID
                                merged_record[col] = group[col].iloc[0]
                            else:
                                # Take the first non-null value
                                non_null_values = group[col].dropna()
                                if len(non_null_values) > 0:
                                    merged_record[col] = non_null_values.iloc[0]
                                else:
                                    merged_record[col] = np.nan
                        
                        merged_records.append(merged_record)
                    
                    elif merge_strategy == 'newest':
                        # Keep the record with the highest ID (assuming newer)
                        if 'id' in group.columns:
                            newest_record = group.loc[group['id'].idxmax()]
                            merged_records.append(newest_record.to_dict())
                        else:
                            merged_records.append(group.iloc[-1].to_dict())
                
                else:
                    # Keep single records as is
                    merged_records.append(group.iloc[0].to_dict())
            
            resolved_df = pd.DataFrame(merged_records)
            
            original_count = len(self.df)
            removed_count = original_count - len(resolved_df)
            
            print(f"Original records: {original_count}")
            print(f"Total name groups: {total_groups}")
            print(f"Groups with duplicates: {merged_groups}")
            print(f"After merging resolution: {len(resolved_df)}")
            print(f"Removed duplicates: {removed_count}")
            
            self.resolution_results['merge_resolution'] = {
                'original_count': original_count,
                'final_count': len(resolved_df),
                'removed_count': removed_count,
                'merged_groups': merged_groups,
                'method': f'record_merging_{merge_strategy}'
            }
            
            return resolved_df
        
        else:
            print("Required columns not found for merge resolution")
            return self.df
    
    def resolve_with_manual_review(self, duplicate_groups, resolution_decisions):
        """Apply manual resolution decisions to duplicate groups"""
        print("\n4.4 Manual Review Resolution")
        print("-" * 35)
        
        # This would typically integrate with a manual review system
        # For demonstration, we'll simulate some manual decisions
        
        resolved_indices = set()
        
        for group_id, decision in resolution_decisions.items():
            if group_id < len(duplicate_groups):
                group = duplicate_groups[group_id]
                
                if decision == 'keep_first':
                    # Keep first record in group
                    resolved_indices.add(group.index[0])
                elif decision == 'keep_last':
                    # Keep last record in group
                    resolved_indices.add(group.index[-1])
                elif decision == 'keep_all':
                    # Keep all records (not duplicates)
                    resolved_indices.update(group.index)
                elif isinstance(decision, int):
                    # Keep specific record by index
                    if decision < len(group):
                        resolved_indices.add(group.index[decision])
        
        # Create resolved dataframe
        resolved_df = self.df.loc[list(resolved_indices)]
        
        original_count = len(self.df)
        removed_count = original_count - len(resolved_df)
        
        print(f"Original records: {original_count}")
        print(f"After manual resolution: {len(resolved_df)}")
        print(f"Removed duplicates: {removed_count}")
        print(f"Manual decisions applied: {len(resolution_decisions)}")
        
        self.resolution_results['manual_resolution'] = {
            'original_count': original_count,
            'final_count': len(resolved_df),
            'removed_count': removed_count,
            'decisions_count': len(resolution_decisions),
            'method': 'manual_review'
        }
        
        return resolved_df

# Initialize duplicate resolver
resolver = DuplicateResolver(df)

# Apply different resolution strategies
print("Applying different resolution strategies:")

# 1. Exact duplicate resolution
exact_resolved = resolver.resolve_exact_duplicates(keep='first')

# 2. Data quality priority resolution
quality_columns = ['first_name', 'last_name', 'email', 'phone', 'company']
quality_resolved = resolver.resolve_with_data_quality_priority(quality_columns)

# 3. Record merging resolution
merge_resolved = resolver.resolve_with_record_merging(merge_strategy='most_complete')

# 4. Simulate manual review (for demonstration)
# In practice, this would involve human review interface
sample_decisions = {
    0: 'keep_first',
    1: 'keep_last',
    2: 0,  # Keep first record (index 0)
    3: 'keep_all'
}

# Create some sample duplicate groups for manual review
sample_groups = []
if len(exact_detector.duplicate_results.get('complete_duplicates', {}).get('duplicates_df', pd.DataFrame())) > 0:
    duplicates_df = exact_detector.duplicate_results['complete_duplicates']['duplicates_df']
    for name, group in duplicates_df.groupby(['first_name', 'last_name']):
        if len(group) > 1:
            sample_groups.append(group)
            if len(sample_groups) >= 4:  # Just take first 4 for demo
                break

if sample_groups:
    manual_resolved = resolver.resolve_with_manual_review(sample_groups, sample_decisions)

# ===============================================================================
# 5. DUPLICATE ANALYSIS VISUALIZATION
# ===============================================================================

print("\n" + "="*60)
print("5. DUPLICATE ANALYSIS VISUALIZATION")
print("="*60)

def create_duplicate_analysis_dashboard(df, exact_detector, resolver):
    """Create comprehensive duplicate analysis dashboard"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Duplicate Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Duplicate Detection Summary
    ax1 = axes[0, 0]
    
    # Get duplicate counts from different methods
    duplicate_counts = {
        'Complete Duplicates': exact_detector.duplicate_results.get('complete_duplicates', {}).get('total_duplicates', 0),
        'Email Duplicates': exact_detector.duplicate_results.get('duplicates_email', {}).get('total_duplicates', 0),
        'Name Duplicates': exact_detector.duplicate_results.get('duplicates_first_name+last_name', {}).get('total_duplicates', 0),
        'Phone Duplicates': exact_detector.duplicate_results.get('duplicates_phone', {}).get('total_duplicates', 0)
    }
    
    methods = list(duplicate_counts.keys())
    counts = list(duplicate_counts.values())
    
    bars = ax1.bar(methods, counts, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax1.set_title('Duplicate Counts by Detection Method')
    ax1.set_ylabel('Number of Duplicates')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # 2. Duplicate Resolution Comparison
    ax2 = axes[0, 1]
    
    resolution_methods = []
    records_removed = []
    
    for method, results in resolver.resolution_results.items():
        resolution_methods.append(method.replace('_', ' ').title())
        records_removed.append(results['removed_count'])
    
    if resolution_methods:
        ax2.bar(resolution_methods, records_removed, color='steelblue', alpha=0.7)
        ax2.set_title('Records Removed by Resolution Method')
        ax2.set_ylabel('Records Removed')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No resolution results', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title('Records Removed by Resolution Method')
    
    # 3. Duplicate Distribution by Column
    ax3 = axes[0, 2]
    
    # Calculate duplicate rates by column
    column_duplicate_rates = {}
    for col in df.columns:
        if col != 'id':
            duplicate_count = df[col].duplicated().sum()
            total_count = df[col].count()
            if total_count > 0:
                column_duplicate_rates[col] = duplicate_count / total_count
    
    if column_duplicate_rates:
        columns = list(column_duplicate_rates.keys())
        rates = list(column_duplicate_rates.values())
        
        ax3.barh(columns, rates, color='lightcoral', alpha=0.7)
        ax3.set_title('Duplicate Rate by Column')
        ax3.set_xlabel('Duplicate Rate')
    else:
        ax3.text(0.5, 0.5, 'No duplicate data', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Duplicate Rate by Column')
    
    # 4. Dataset Size Reduction
    ax4 = axes[1, 0]
    
    original_size = len(df)
    resolution_sizes = [original_size]
    resolution_labels = ['Original']
    
    for method, results in resolver.resolution_results.items():
        resolution_sizes.append(results['final_count'])
        resolution_labels.append(method.replace('_', ' ').title())
    
    ax4.plot(resolution_labels, resolution_sizes, 'o-', linewidth=2, markersize=8)
    ax4.set_title('Dataset Size After Resolution')
    ax4.set_ylabel('Number of Records')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Duplicate Pattern Analysis
    ax5 = axes[1, 1]
    
    # Analyze most common duplicate patterns
    if 'first_name' in df.columns and 'last_name' in df.columns:
        name_counts = (df['first_name'] + ' ' + df['last_name']).value_counts()
        duplicate_names = name_counts[name_counts > 1]
        
        if len(duplicate_names) > 0:
            top_duplicates = duplicate_names.head(10)
            ax5.barh(range(len(top_duplicates)), top_duplicates.values, color='lightblue', alpha=0.7)
            ax5.set_yticks(range(len(top_duplicates)))
            ax5.set_yticklabels(top_duplicates.index, fontsize=8)
            ax5.set_title('Top Duplicate Name Patterns')
            ax5.set_xlabel('Frequency')
        else:
            ax5.text(0.5, 0.5, 'No duplicate names found', ha='center', va='center',
                    transform=ax5.transAxes)
            ax5.set_title('Top Duplicate Name Patterns')
    
    # 6. Resolution Effectiveness
    ax6 = axes[1, 2]
    
    if resolver.resolution_results:
        # Create pie chart of resolution effectiveness
        total_original = len(df)
        
        # Calculate how many duplicates were removed by each method
        method_effectiveness = {}
        for method, results in resolver.resolution_results.items():
            effectiveness = (results['removed_count'] / total_original) * 100
            method_effectiveness[method.replace('_', ' ').title()] = effectiveness
        
        if method_effectiveness:
            labels = list(method_effectiveness.keys())
            sizes = list(method_effectiveness.values())
            
            ax6.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax6.set_title('Resolution Method Effectiveness')
        else:
            ax6.text(0.5, 0.5, 'No resolution data', ha='center', va='center',
                    transform=ax6.transAxes)
            ax6.set_title('Resolution Method Effectiveness')
    
    plt.tight_layout()
    plt.show()

# Create duplicate analysis dashboard
create_duplicate_analysis_dashboard(df, exact_detector, resolver)

# ===============================================================================
# 6. DEDUPLICATION RECOMMENDATIONS AND SUMMARY
# ===============================================================================

print("\n" + "="*60)
print("6. DEDUPLICATION RECOMMENDATIONS AND SUMMARY")
print("="*60)

print("6.1 Duplicate Analysis Summary")
print("-" * 35)

# Summarize findings
total_records = len(df)
complete_duplicates = exact_detector.duplicate_results.get('complete_duplicates', {}).get('total_duplicates', 0)
email_duplicates = exact_detector.duplicate_results.get('duplicates_email', {}).get('total_duplicates', 0)
name_duplicates = exact_detector.duplicate_results.get('duplicates_first_name+last_name', {}).get('total_duplicates', 0)

print(f"Dataset Overview:")
print(f"  Total records: {total_records:,}")
print(f"  Complete duplicates: {complete_duplicates:,} ({(complete_duplicates/total_records)*100:.1f}%)")
print(f"  Email duplicates: {email_duplicates:,} ({(email_duplicates/total_records)*100:.1f}%)")
print(f"  Name duplicates: {name_duplicates:,} ({(name_duplicates/total_records)*100:.1f}%)")

print(f"\nResolution Results:")
for method, results in resolver.resolution_results.items():
    reduction_rate = (results['removed_count'] / results['original_count']) * 100
    print(f"  {method.replace('_', ' ').title()}: {results['removed_count']:,} records removed ({reduction_rate:.1f}%)")

print("\n6.2 Recommendations")
print("-" * 25)

recommendations = []

# Generate recommendations based on findings
if complete_duplicates > 0:
    recommendations.append(f"• Remove {complete_duplicates:,} exact duplicate records immediately")

if email_duplicates > complete_duplicates:
    recommendations.append("• Investigate email-based duplicates - may indicate data entry issues")

if name_duplicates > complete_duplicates:
    recommendations.append("• Review name-based duplicates for potential fuzzy matches")

# Resolution strategy recommendations
if FUZZYWUZZY_AVAILABLE and 'comprehensive_matches' in locals():
    fuzzy_count = len(comprehensive_matches) if comprehensive_matches else 0
    if fuzzy_count > 0:
        recommendations.append(f"• Review {fuzzy_count} potential fuzzy matches for resolution")

# General recommendations
recommendations.extend([
    "• Implement data validation at source to prevent duplicates",
    "• Establish unique identifiers for all records",
    "• Create automated duplicate detection in ETL pipelines",
    "• Develop data quality monitoring dashboards",
    "• Train staff on data entry best practices",
    "• Regular deduplication maintenance schedules"
])

print("Based on the duplicate analysis, here are the key recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i:2d}. {rec}")

print("\n6.3 Implementation Strategy")
print("-" * 32)

strategy_steps = [
    "1. **Immediate Actions:**",
    "   - Remove exact duplicates using automated methods",
    "   - Flag potential fuzzy duplicates for manual review",
    "",
    "2. **Short-term (1-3 months):**",
    "   - Implement data validation rules",
    "   - Create duplicate detection workflows",
    "   - Train team on duplicate identification",
    "",
    "3. **Long-term (3-12 months):**",
    "   - Establish data governance policies",
    "   - Implement automated monitoring systems",
    "   - Regular data quality assessments"
]

for step in strategy_steps:
    print(step)

print(f"\nDuplicate handling analysis complete!")
print(f"Priority: {'High' if complete_duplicates > total_records * 0.05 else 'Medium' if complete_duplicates > 0 else 'Low'}")
print(f"Focus on implementing exact duplicate removal and fuzzy matching for comprehensive deduplication.")
