"""
===============================================================================
DATA PIPELINE TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive data pipeline for automated data processing workflows

This template covers:
- Data ingestion from multiple sources
- Data validation and quality checks
- Data transformation pipelines
- Error handling and logging
- Pipeline monitoring and alerts
- Data export and versioning

Prerequisites:
- pandas, numpy, matplotlib, seaborn
- Additional: sqlalchemy, requests (for advanced features)
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# SETUP LOGGING AND CONFIGURATION
# ===============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Pipeline configuration
PIPELINE_CONFIG = {
    'data_sources': {
        'csv_files': ['data1.csv', 'data2.csv'],
        'excel_files': ['data.xlsx'],
        'json_files': ['data.json'],
        'parquet_files': ['data.parquet']
    },
    'output_path': 'processed_data/',
    'backup_path': 'backup/',
    'validation_rules': {
        'required_columns': ['id', 'timestamp', 'value'],
        'data_types': {'id': 'int64', 'value': 'float64'},
        'value_ranges': {'value': {'min': 0, 'max': 1000}}
    },
    'quality_thresholds': {
        'missing_data_max': 0.1,  # 10% max missing data
        'duplicate_max': 0.05,    # 5% max duplicates
        'outlier_threshold': 3    # 3 standard deviations
    }
}

print("Data Pipeline Configuration:")
print(json.dumps(PIPELINE_CONFIG, indent=2))

# ===============================================================================
# 1. DATA INGESTION CLASS
# ===============================================================================

print("\n" + "="*60)
print("1. DATA INGESTION MODULE")
print("="*60)

class DataIngestion:
    """Handle data ingestion from multiple sources"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def ingest_csv(self, file_path, **kwargs):
        """Ingest data from CSV file"""
        try:
            self.logger.info(f"Ingesting CSV file: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV {file_path}: {str(e)}")
            return None
    
    def ingest_excel(self, file_path, sheet_name=None, **kwargs):
        """Ingest data from Excel file"""
        try:
            self.logger.info(f"Ingesting Excel file: {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Excel {file_path}: {str(e)}")
            return None
    
    def ingest_json(self, file_path, orient='records', **kwargs):
        """Ingest data from JSON file"""
        try:
            self.logger.info(f"Ingesting JSON file: {file_path}")
            df = pd.read_json(file_path, orient=orient, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading JSON {file_path}: {str(e)}")
            return None
    
    def ingest_parquet(self, file_path, **kwargs):
        """Ingest data from Parquet file"""
        try:
            self.logger.info(f"Ingesting Parquet file: {file_path}")
            df = pd.read_parquet(file_path, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Parquet {file_path}: {str(e)}")
            return None
    
    def ingest_multiple_files(self, file_list, file_type='csv'):
        """Ingest and combine multiple files"""
        dfs = []
        for file_path in file_list:
            if file_type == 'csv':
                df = self.ingest_csv(file_path)
            elif file_type == 'excel':
                df = self.ingest_excel(file_path)
            elif file_type == 'json':
                df = self.ingest_json(file_path)
            elif file_type == 'parquet':
                df = self.ingest_parquet(file_path)
            else:
                self.logger.error(f"Unsupported file type: {file_type}")
                continue
                
            if df is not None:
                df['source_file'] = file_path
                dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            self.logger.info(f"Combined {len(dfs)} files into {len(combined_df)} rows")
            return combined_df
        else:
            self.logger.error("No files were successfully loaded")
            return None

# Initialize ingestion module
ingestion = DataIngestion(PIPELINE_CONFIG)

# Create sample data for demonstration
sample_data = pd.DataFrame({
    'id': range(1, 1001),
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
    'value': np.random.normal(100, 20, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'status': np.random.choice(['active', 'inactive'], 1000)
})

# Introduce some data quality issues for demonstration
sample_data.loc[10:20, 'value'] = np.nan  # Missing values
sample_data.loc[50:60, 'value'] = np.random.normal(500, 50, 11)  # Outliers
sample_data = pd.concat([sample_data, sample_data.iloc[:10]])  # Duplicates

print(f"Sample data created: {sample_data.shape}")
print("Data quality issues introduced for demonstration")

# ===============================================================================
# 2. DATA VALIDATION CLASS
# ===============================================================================

print("\n" + "="*60)
print("2. DATA VALIDATION MODULE")
print("="*60)

class DataValidator:
    """Validate data quality and integrity"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_schema(self, df):
        """Validate data schema and structure"""
        self.logger.info("Validating data schema...")
        
        # Check required columns
        required_cols = self.config['validation_rules']['required_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.validation_results['missing_columns'] = missing_cols
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check data types
        if 'data_types' in self.config['validation_rules']:
            expected_types = self.config['validation_rules']['data_types']
            type_issues = []
            
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if actual_type != expected_type:
                        type_issues.append({
                            'column': col,
                            'expected': expected_type,
                            'actual': actual_type
                        })
            
            if type_issues:
                self.validation_results['type_issues'] = type_issues
                self.logger.warning(f"Data type issues found: {type_issues}")
        
        self.logger.info("Schema validation completed")
        return len(missing_cols) == 0
    
    def validate_data_quality(self, df):
        """Validate data quality metrics"""
        self.logger.info("Validating data quality...")
        
        quality_issues = {}
        
        # Check missing data
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > self.config['quality_thresholds']['missing_data_max']]
        
        if not high_missing.empty:
            quality_issues['high_missing_data'] = high_missing.to_dict()
            self.logger.warning(f"Columns with high missing data: {high_missing.index.tolist()}")
        
        # Check duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_pct = duplicate_count / len(df)
        
        if duplicate_pct > self.config['quality_thresholds']['duplicate_max']:
            quality_issues['high_duplicates'] = {
                'count': duplicate_count,
                'percentage': duplicate_pct
            }
            self.logger.warning(f"High duplicate rate: {duplicate_pct:.2%}")
        
        # Check outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_count = (z_scores > self.config['quality_thresholds']['outlier_threshold']).sum()
                
                if outlier_count > 0:
                    outlier_info[col] = {
                        'count': outlier_count,
                        'percentage': outlier_count / len(df)
                    }
        
        if outlier_info:
            quality_issues['outliers'] = outlier_info
            self.logger.info(f"Outliers detected in columns: {list(outlier_info.keys())}")
        
        self.validation_results['quality_issues'] = quality_issues
        self.logger.info("Data quality validation completed")
        
        return quality_issues
    
    def validate_value_ranges(self, df):
        """Validate value ranges"""
        self.logger.info("Validating value ranges...")
        
        range_issues = {}
        
        if 'value_ranges' in self.config['validation_rules']:
            for col, ranges in self.config['validation_rules']['value_ranges'].items():
                if col in df.columns:
                    min_val, max_val = ranges.get('min'), ranges.get('max')
                    
                    if min_val is not None:
                        below_min = (df[col] < min_val).sum()
                        if below_min > 0:
                            range_issues[f'{col}_below_min'] = below_min
                    
                    if max_val is not None:
                        above_max = (df[col] > max_val).sum()
                        if above_max > 0:
                            range_issues[f'{col}_above_max'] = above_max
        
        if range_issues:
            self.validation_results['range_issues'] = range_issues
            self.logger.warning(f"Value range issues: {range_issues}")
        
        return range_issues
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        self.logger.info("Generating validation report...")
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'results': self.validation_results
        }
        
        # Save report
        with open('validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

# Initialize validator
validator = DataValidator(PIPELINE_CONFIG)

# Run validation on sample data
schema_valid = validator.validate_schema(sample_data)
quality_issues = validator.validate_data_quality(sample_data)
range_issues = validator.validate_value_ranges(sample_data)
validation_report = validator.generate_validation_report()

print(f"Schema validation: {'PASSED' if schema_valid else 'FAILED'}")
print(f"Quality issues found: {len(quality_issues)}")
print(f"Range issues found: {len(range_issues)}")

# ===============================================================================
# 3. DATA TRANSFORMATION CLASS
# ===============================================================================

print("\n" + "="*60)
print("3. DATA TRANSFORMATION MODULE")
print("="*60)

class DataTransformer:
    """Handle data transformation and cleaning"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transformation_log = []
    
    def handle_missing_data(self, df, strategy='mean', columns=None):
        """Handle missing data with various strategies"""
        self.logger.info(f"Handling missing data with strategy: {strategy}")
        
        if columns is None:
            columns = df.columns
        
        original_missing = df[columns].isnull().sum().sum()
        
        if strategy == 'mean':
            numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mode':
            for col in columns:
                if col in df.columns:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
        elif strategy == 'forward_fill':
            df[columns] = df[columns].fillna(method='ffill')
        elif strategy == 'backward_fill':
            df[columns] = df[columns].fillna(method='bfill')
        elif strategy == 'drop':
            df = df.dropna(subset=columns)
        
        final_missing = df[columns].isnull().sum().sum()
        
        self.transformation_log.append({
            'operation': 'handle_missing_data',
            'strategy': strategy,
            'columns': columns,
            'original_missing': original_missing,
            'final_missing': final_missing
        })
        
        self.logger.info(f"Missing data handled: {original_missing} -> {final_missing}")
        return df
    
    def remove_duplicates(self, df, subset=None, keep='first'):
        """Remove duplicate records"""
        self.logger.info("Removing duplicate records...")
        
        original_count = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df)
        
        self.transformation_log.append({
            'operation': 'remove_duplicates',
            'subset': subset,
            'keep': keep,
            'original_count': original_count,
            'final_count': final_count,
            'removed': original_count - final_count
        })
        
        self.logger.info(f"Duplicates removed: {original_count - final_count}")
        return df
    
    def handle_outliers(self, df, columns=None, method='clip', threshold=3):
        """Handle outliers in numeric data"""
        self.logger.info(f"Handling outliers with method: {method}")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        
        for col in columns:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # Identify outliers
                z_scores = np.abs((df[col] - mean_val) / std_val)
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    if method == 'clip':
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        df[col] = df[col].clip(lower_bound, upper_bound)
                    elif method == 'remove':
                        df = df[~outliers]
                    elif method == 'cap':
                        percentile_99 = df[col].quantile(0.99)
                        percentile_01 = df[col].quantile(0.01)
                        df[col] = df[col].clip(percentile_01, percentile_99)
                    
                    outlier_info[col] = outlier_count
        
        self.transformation_log.append({
            'operation': 'handle_outliers',
            'method': method,
            'threshold': threshold,
            'columns': columns,
            'outliers_found': outlier_info
        })
        
        self.logger.info(f"Outliers handled in columns: {list(outlier_info.keys())}")
        return df
    
    def standardize_formats(self, df):
        """Standardize data formats"""
        self.logger.info("Standardizing data formats...")
        
        # Standardize date columns
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_columns.append(col)
                except:
                    pass
        
        # Standardize text columns
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col not in date_columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        self.transformation_log.append({
            'operation': 'standardize_formats',
            'date_columns': date_columns,
            'text_columns': text_columns.tolist()
        })
        
        self.logger.info(f"Formats standardized for {len(date_columns)} date and {len(text_columns)} text columns")
        return df
    
    def create_derived_features(self, df):
        """Create derived features"""
        self.logger.info("Creating derived features...")
        
        derived_features = []
        
        # Time-based features
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                
                derived_features.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day', 
                    f'{col}_hour', f'{col}_dayofweek'
                ])
        
        # Numeric feature interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:3]):  # Limit for demonstration
                for col2 in numeric_cols[i+1:4]:
                    if col1 != col2:
                        # Ratio feature
                        ratio_col = f'{col1}_{col2}_ratio'
                        df[ratio_col] = df[col1] / (df[col2] + 1e-8)  # Avoid division by zero
                        derived_features.append(ratio_col)
        
        self.transformation_log.append({
            'operation': 'create_derived_features',
            'features_created': derived_features
        })
        
        self.logger.info(f"Created {len(derived_features)} derived features")
        return df
    
    def get_transformation_summary(self):
        """Get summary of all transformations"""
        return {
            'total_operations': len(self.transformation_log),
            'operations': self.transformation_log,
            'timestamp': datetime.now().isoformat()
        }

# Initialize transformer
transformer = DataTransformer(PIPELINE_CONFIG)

# Apply transformations to sample data
print("Applying data transformations...")
cleaned_data = sample_data.copy()

# Handle missing data
cleaned_data = transformer.handle_missing_data(cleaned_data, strategy='mean')

# Remove duplicates
cleaned_data = transformer.remove_duplicates(cleaned_data)

# Handle outliers
cleaned_data = transformer.handle_outliers(cleaned_data, method='clip')

# Standardize formats
cleaned_data = transformer.standardize_formats(cleaned_data)

# Create derived features
cleaned_data = transformer.create_derived_features(cleaned_data)

# Get transformation summary
transformation_summary = transformer.get_transformation_summary()
print(f"Transformations completed: {transformation_summary['total_operations']} operations")
print(f"Final data shape: {cleaned_data.shape}")

# ===============================================================================
# 4. PIPELINE ORCHESTRATOR
# ===============================================================================

print("\n" + "="*60)
print("4. PIPELINE ORCHESTRATOR")
print("="*60)

class DataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ingestion = DataIngestion(config)
        self.validator = DataValidator(config)
        self.transformer = DataTransformer(config)
        self.pipeline_log = []
    
    def run_pipeline(self, data_source=None, transformation_steps=None):
        """Run complete data pipeline"""
        self.logger.info("Starting data pipeline execution...")
        
        pipeline_start = datetime.now()
        
        try:
            # Step 1: Data Ingestion
            self.logger.info("Step 1: Data Ingestion")
            if data_source is not None:
                df = data_source  # Use provided data
            else:
                # Default: try to load from configured sources
                df = self._load_default_data()
            
            if df is None:
                raise ValueError("No data could be loaded")
            
            self.logger.info(f"Data loaded: {df.shape}")
            
            # Step 2: Data Validation
            self.logger.info("Step 2: Data Validation")
            schema_valid = self.validator.validate_schema(df)
            quality_issues = self.validator.validate_data_quality(df)
            range_issues = self.validator.validate_value_ranges(df)
            
            if not schema_valid:
                raise ValueError("Schema validation failed")
            
            # Step 3: Data Transformation
            self.logger.info("Step 3: Data Transformation")
            
            # Default transformation steps
            if transformation_steps is None:
                transformation_steps = [
                    ('handle_missing_data', {'strategy': 'mean'}),
                    ('remove_duplicates', {}),
                    ('handle_outliers', {'method': 'clip'}),
                    ('standardize_formats', {}),
                    ('create_derived_features', {})
                ]
            
            processed_df = df.copy()
            for step_name, step_params in transformation_steps:
                if hasattr(self.transformer, step_name):
                    processed_df = getattr(self.transformer, step_name)(processed_df, **step_params)
            
            # Step 4: Final Validation
            self.logger.info("Step 4: Final Validation")
            final_quality = self.validator.validate_data_quality(processed_df)
            
            # Step 5: Export Results
            self.logger.info("Step 5: Export Results")
            self._export_results(processed_df)
            
            pipeline_end = datetime.now()
            execution_time = (pipeline_end - pipeline_start).total_seconds()
            
            # Log pipeline execution
            self.pipeline_log.append({
                'execution_time': execution_time,
                'start_time': pipeline_start.isoformat(),
                'end_time': pipeline_end.isoformat(),
                'input_shape': df.shape,
                'output_shape': processed_df.shape,
                'quality_issues': len(quality_issues),
                'transformations': len(self.transformer.transformation_log),
                'status': 'SUCCESS'
            })
            
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            
            return {
                'status': 'SUCCESS',
                'data': processed_df,
                'execution_time': execution_time,
                'quality_report': {
                    'initial_issues': quality_issues,
                    'final_issues': final_quality
                },
                'transformation_summary': self.transformer.get_transformation_summary()
            }
            
        except Exception as e:
            pipeline_end = datetime.now()
            execution_time = (pipeline_end - pipeline_start).total_seconds()
            
            self.pipeline_log.append({
                'execution_time': execution_time,
                'start_time': pipeline_start.isoformat(),
                'end_time': pipeline_end.isoformat(),
                'status': 'FAILED',
                'error': str(e)
            })
            
            self.logger.error(f"Pipeline failed: {str(e)}")
            
            return {
                'status': 'FAILED',
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _load_default_data(self):
        """Load data from default configured sources"""
        # This would implement loading from configured sources
        # For demonstration, return None to use provided data
        return None
    
    def _export_results(self, df):
        """Export processed data"""
        # Create output directory
        output_path = Path(self.config['output_path'])
        output_path.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export in multiple formats
        df.to_csv(output_path / f'processed_data_{timestamp}.csv', index=False)
        df.to_parquet(output_path / f'processed_data_{timestamp}.parquet', index=False)
        
        # Export metadata
        metadata = {
            'timestamp': timestamp,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        with open(output_path / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Results exported to {output_path}")
    
    def get_pipeline_stats(self):
        """Get pipeline execution statistics"""
        if not self.pipeline_log:
            return "No pipeline executions recorded"
        
        successful_runs = [run for run in self.pipeline_log if run['status'] == 'SUCCESS']
        failed_runs = [run for run in self.pipeline_log if run['status'] == 'FAILED']
        
        if successful_runs:
            avg_execution_time = sum(run['execution_time'] for run in successful_runs) / len(successful_runs)
        else:
            avg_execution_time = 0
        
        return {
            'total_runs': len(self.pipeline_log),
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs),
            'success_rate': len(successful_runs) / len(self.pipeline_log) if self.pipeline_log else 0,
            'average_execution_time': avg_execution_time,
            'last_execution': self.pipeline_log[-1] if self.pipeline_log else None
        }

# Initialize and run pipeline
pipeline = DataPipeline(PIPELINE_CONFIG)

# Run pipeline with sample data
result = pipeline.run_pipeline(data_source=sample_data)

print(f"Pipeline Status: {result['status']}")
if result['status'] == 'SUCCESS':
    print(f"Execution Time: {result['execution_time']:.2f} seconds")
    print(f"Output Shape: {result['data'].shape}")
    print(f"Transformations Applied: {result['transformation_summary']['total_operations']}")

# Get pipeline statistics
stats = pipeline.get_pipeline_stats()
print(f"\nPipeline Statistics:")
print(f"Success Rate: {stats['success_rate']:.2%}")
print(f"Average Execution Time: {stats['average_execution_time']:.2f} seconds")

# ===============================================================================
# 5. MONITORING AND ALERTING
# ===============================================================================

print("\n" + "="*60)
print("5. MONITORING AND ALERTING")
print("="*60)

class PipelineMonitor:
    """Monitor pipeline performance and send alerts"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
    
    def collect_metrics(self, pipeline_result):
        """Collect performance metrics"""
        timestamp = datetime.now()
        
        metrics = {
            'timestamp': timestamp,
            'execution_time': pipeline_result.get('execution_time', 0),
            'status': pipeline_result.get('status', 'UNKNOWN'),
            'data_quality_score': self._calculate_quality_score(pipeline_result),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage()
        }
        
        # Store metrics
        self.metrics[timestamp] = metrics
        self.logger.info(f"Metrics collected: {metrics}")
        
        return metrics
    
    def _calculate_quality_score(self, pipeline_result):
        """Calculate data quality score"""
        if 'quality_report' not in pipeline_result:
            return 0.5
        
        initial_issues = len(pipeline_result['quality_report']['initial_issues'])
        final_issues = len(pipeline_result['quality_report']['final_issues'])
        
        # Simple quality score based on issue reduction
        if initial_issues == 0:
            return 1.0
        
        improvement = (initial_issues - final_issues) / initial_issues
        return max(0.0, min(1.0, improvement))
    
    def _get_memory_usage(self):
        """Get current memory usage (simplified)"""
        # In a real implementation, you'd use psutil or similar
        return 0.75  # Placeholder
    
    def _get_cpu_usage(self):
        """Get current CPU usage (simplified)"""
        # In a real implementation, you'd use psutil or similar
        return 0.45  # Placeholder
    
    def check_alerts(self, metrics):
        """Check if any alerts should be triggered"""
        alerts = []
        
        # Execution time alert
        if metrics['execution_time'] > 300:  # 5 minutes
            alerts.append({
                'type': 'PERFORMANCE',
                'message': f"Pipeline execution time ({metrics['execution_time']:.2f}s) exceeded threshold",
                'severity': 'WARNING'
            })
        
        # Failure alert
        if metrics['status'] == 'FAILED':
            alerts.append({
                'type': 'FAILURE',
                'message': "Pipeline execution failed",
                'severity': 'CRITICAL'
            })
        
        # Quality alert
        if metrics['data_quality_score'] < 0.5:
            alerts.append({
                'type': 'QUALITY',
                'message': f"Data quality score ({metrics['data_quality_score']:.2f}) below threshold",
                'severity': 'WARNING'
            })
        
        # Resource alerts
        if metrics['memory_usage'] > 0.9:
            alerts.append({
                'type': 'RESOURCE',
                'message': f"High memory usage ({metrics['memory_usage']:.2%})",
                'severity': 'WARNING'
            })
        
        if alerts:
            self.logger.warning(f"Alerts triggered: {len(alerts)}")
            for alert in alerts:
                self.logger.warning(f"{alert['severity']}: {alert['message']}")
        
        return alerts
    
    def generate_dashboard_data(self):
        """Generate data for monitoring dashboard"""
        if not self.metrics:
            return {}
        
        metrics_df = pd.DataFrame(list(self.metrics.values()))
        
        dashboard_data = {
            'execution_times': metrics_df['execution_time'].tolist(),
            'timestamps': [m.isoformat() for m in metrics_df['timestamp']],
            'success_rate': (metrics_df['status'] == 'SUCCESS').mean(),
            'average_execution_time': metrics_df['execution_time'].mean(),
            'quality_scores': metrics_df['data_quality_score'].tolist(),
            'resource_usage': {
                'memory': metrics_df['memory_usage'].tolist(),
                'cpu': metrics_df['cpu_usage'].tolist()
            }
        }
        
        return dashboard_data

# Initialize monitor
monitor = PipelineMonitor(PIPELINE_CONFIG)

# Collect metrics for the pipeline run
metrics = monitor.collect_metrics(result)
alerts = monitor.check_alerts(metrics)

print(f"Metrics collected: Quality Score = {metrics['data_quality_score']:.2f}")
print(f"Alerts triggered: {len(alerts)}")

# ===============================================================================
# 6. PIPELINE SUMMARY AND VISUALIZATION
# ===============================================================================

print("\n" + "="*60)
print("6. PIPELINE SUMMARY AND VISUALIZATION")
print("="*60)

# Create summary visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Data quality comparison
if result['status'] == 'SUCCESS':
    original_missing = sample_data.isnull().sum().sum()
    processed_missing = result['data'].isnull().sum().sum()
    
    axes[0, 0].bar(['Original', 'Processed'], [original_missing, processed_missing])
    axes[0, 0].set_title('Missing Values: Before vs After')
    axes[0, 0].set_ylabel('Count')
    
    # Data shape comparison
    axes[0, 1].bar(['Original', 'Processed'], [sample_data.shape[0], result['data'].shape[0]])
    axes[0, 1].set_title('Row Count: Before vs After')
    axes[0, 1].set_ylabel('Rows')
    
    # Feature count comparison
    axes[1, 0].bar(['Original', 'Processed'], [sample_data.shape[1], result['data'].shape[1]])
    axes[1, 0].set_title('Feature Count: Before vs After')
    axes[1, 0].set_ylabel('Features')
    
    # Transformation summary
    if result['transformation_summary']['operations']:
        operations = [op['operation'] for op in result['transformation_summary']['operations']]
        operation_counts = pd.Series(operations).value_counts()
        
        axes[1, 1].pie(operation_counts.values, labels=operation_counts.index, autopct='%1.0f%%')
        axes[1, 1].set_title('Transformation Operations')

plt.tight_layout()
plt.show()

# Final summary
print("\n" + "="*60)
print("PIPELINE EXECUTION SUMMARY")
print("="*60)

summary = {
    'Pipeline Status': result['status'],
    'Execution Time': f"{result.get('execution_time', 0):.2f} seconds",
    'Data Quality Score': f"{metrics['data_quality_score']:.2f}",
    'Alerts Generated': len(alerts),
    'Original Data Shape': sample_data.shape,
    'Processed Data Shape': result['data'].shape if result['status'] == 'SUCCESS' else 'N/A',
    'Transformations Applied': result['transformation_summary']['total_operations'] if result['status'] == 'SUCCESS' else 0,
    'Files Generated': 'CSV, Parquet, Metadata JSON' if result['status'] == 'SUCCESS' else 'None'
}

print("Pipeline Execution Summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\nPipeline components ready for production deployment!")
print("Key features implemented:")
print("  ✓ Multi-source data ingestion")
print("  ✓ Comprehensive data validation")
print("  ✓ Automated data transformation")
print("  ✓ Error handling and logging")
print("  ✓ Performance monitoring")
print("  ✓ Automated alerting")
print("  ✓ Export in multiple formats")
print("  ✓ Pipeline orchestration")
print("  ✓ Execution tracking and statistics")
