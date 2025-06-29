# 🐍 Python Templates for Data Analysis & Data Science

This repository contains a comprehensive, folder-based set of **Python templates** for universal data analysis and data science projects. Each template is designed as a plug-and-play guide with reusable `.py` scripts organized by core task.

## 📋 Repository Overview

- **No functions or classes** - These are workflow templates with real data-handling code 
- **Standard libraries focus** - Uses `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `plotly`
- **Copy-paste ready** - Each file is modular and meant to be adapted directly into your projects
- **Comprehensive comments** - Detailed inline explanations and best practices
- **Report-ready visualizations** - All plots include proper labels, titles, and formatting
- **Advanced workflows** - Includes time series, feature engineering, model evaluation, and pipeline automation

## 📁 Folder Structure & Templates

### 🧹 `data_cleaning/`
**Handle missing values, outliers, data types, feature encoding, and data quality**

| Template | Description |
|----------|-------------|
| `handle_missing_data.py` | Drop, mean/median impute, KNN impute, and predictive imputation methods |
| `type_conversion_recoding.py` | Convert data types, recode categorical variables, create derived features |
| `outlier_detection.py` | Z-score, IQR, and Isolation Forest methods with visualization and treatment options |
| `feature_encoding.py` | One-hot, label, binary, target, and frequency encoding with multicollinearity detection |
| `data_validation.py` | Comprehensive data validation, quality assessment, schema validation, and automated pipelines |
| `duplicate_handling.py` | Exact, fuzzy, and statistical duplicate detection with multiple resolution strategies |

**Key Features:**
- Memory-efficient pandas operations
- Handles edge cases and provides recommendations
- Includes validation and quality checks
- Multiple imputation strategies with pros/cons
- **Advanced data quality scoring and monitoring**
- **Fuzzy matching and record linkage capabilities**

---

### 📊 `exploratory_data_analysis/`
**Comprehensive EDA with statistical analysis and visualizations**

| Template | Description |
|----------|-------------|
| `variable_distributions.py` | Distribution analysis, normality tests, shape statistics, Q-Q plots |
| `group_analysis.py` | Compare distributions across groups, statistical tests, effect sizes |
| `correlation_analysis.py` | Pearson, Spearman, Kendall correlations with significance testing and clustering |

**Key Features:**
- Automated distribution shape analysis
- Statistical significance testing
- Effect size calculations  
- Multicollinearity detection
- Publication-ready visualizations

---

### 📈 `essential_statistics/`
**Descriptive and inferential statistics with proper assumptions checking**

| Template | Description |
|----------|-------------|
| `descriptive_stats.py` | Mean, std, skewness, kurtosis, robust statistics, entropy measures |
| `inferential_stats.py` | T-tests, Chi-square, ANOVA, confidence intervals, power analysis |

**Key Features:**
- Comprehensive statistical summaries
- Assumption checking (normality, equal variances)
- Effect sizes (Cohen's d, Cramér's V, eta squared)
- Power analysis and sample size considerations
- Both parametric and non-parametric methods

---

### 📉 `data_visuals/`
**Static, report-ready visualizations with comprehensive customization**

| Template | Description |
|----------|-------------|
| `bar_plots.py` | Simple, grouped, stacked bars with error bars and custom styling |
| `scatter_plots.py` | Basic, with regression lines, hue, size, polynomial fits, annotations |
| `line_plots.py` | Time series, multiple series, confidence intervals, trend analysis |
| `histograms_kde.py` | Distribution plots, overlays, faceting, and density estimation |
| `boxplots_violins.py` | Box plots, violin plots, outlier highlighting, group comparisons |
| `heatmaps_pairplots.py` | Correlation matrices, pairplots, clustermap, missing data patterns |

**Key Features:**
- Report-ready formatting
- Proper axis labels, titles, legends
- Color-blind friendly palettes
- Multiple subplot arrangements
- Statistical overlays (regression lines, confidence intervals)

---

### 🤖 `statistical_models/`
**Machine learning and statistical models with full diagnostics**

| Template | Description |
|----------|-------------|
| `linear_regression.py` | Full linear regression with assumptions testing, diagnostics, interpretation |
| `logistic_regression.py` | Binary/multinomial classification with ROC curves, calibration |
| `decision_trees.py` | Decision trees with pruning, feature importance, visualization |
| `random_forest.py` | Ensemble methods with hyperparameter tuning, feature selection |
| `svm.py` | Support Vector Machines with kernel selection and optimization |
| `knn.py` | K-Nearest Neighbors with optimal k selection, distance metrics, scaling |
| `pca.py` | Principal Component Analysis with scree plots, biplot, interpretation |
| `kmeans.py` | K-means clustering with elbow method, silhouette analysis |
| `neural_network.py` | Basic neural networks with MLPClassifier/Regressor |

**Key Features:**
- Comprehensive model evaluation metrics
- Hyperparameter tuning with cross-validation
- Model interpretation and diagnostics
- Assumption checking and validation
- Feature importance analysis
- Visualization of model performance

---

### ⚡ `feature_engineering/`
**Advanced feature creation and selection techniques**

| Template | Description |
|----------|-------------|
| `feature_creation.py` | Mathematical transformations, binning, interactions, aggregations, time-based features, text processing |
| `feature_selection.py` | Statistical selection, model-based selection, RFE, correlation filtering, performance evaluation |

**Key Features:**
- Mathematical and polynomial transformations
- Categorical and text feature engineering  
- Time-based and cyclical features
- Interaction and aggregation features
- Multiple selection algorithms with comparison
- Performance validation and ranking

---

### 🧠 `model_evaluation/`
**Comprehensive model comparison, evaluation, and diagnostics**

| Template | Description |
|----------|-------------|
| `model_comparison.py` | Cross-validation comparison, statistical testing, ensemble methods, performance visualization |
| `cross_validation_advanced.py` | Advanced CV strategies: stratified, time series, nested, group-based, custom CV methods |
| `hyperparameter_optimization.py` | Grid search, random search, Bayesian optimization, Optuna, successive halving |
| `diagnostic_plots.py` | ROC curves, calibration plots, residual analysis, feature importance, error analysis |

**Key Features:**
- Multiple model types (classification & regression)
- Cross-validation with statistical significance
- Model complexity vs performance analysis
- Ensemble voting and stacking
- Comprehensive performance metrics
- Residual analysis and diagnostics
- **Advanced cross-validation techniques and custom strategies**
- **State-of-the-art hyperparameter optimization methods**
- **Complete diagnostic visualization suite with interpretation guides**

---

### 📈 `advanced_visualizations/`
**Interactive and publication-ready visualizations**

| Template | Description |
|----------|-------------|
| `interactive_plots.py` | Plotly-based interactive charts, 3D plots, animations, dashboards, geographic maps |

**Key Features:**
- Interactive scatter, line, and time series plots
- 3D visualizations and animations
- Geographic mapping and heatmaps
- Dashboard-style layouts
- Network and hierarchical visualizations
- Statistical modeling plots
- Export capabilities (HTML, PNG)

---

### 🔄 `data_pipeline/`
**Automated data processing and pipeline management**

| Template | Description |
|----------|-------------|
| `automated_pipeline.py` | Complete ETL pipeline with validation, transformation, monitoring, and alerting |

**Key Features:**
- Multi-source data ingestion (CSV, Excel, JSON, Parquet)
- Comprehensive data validation and quality checks
- Automated transformation workflows
- Error handling and logging
- Performance monitoring and alerting
- Export in multiple formats
- Pipeline orchestration and statistics

---

### ⏰ `time_series/`
**Time series analysis and forecasting**

| Template | Description |
|----------|-------------|
| `time_series_analysis.py` | Trend analysis, seasonality, stationarity tests, decomposition, autocorrelation, anomaly detection |

**Key Features:**
- Comprehensive trend and seasonality analysis
- Stationarity testing (ADF, KPSS, PP)
- Time series decomposition
- Autocorrelation and partial autocorrelation
- Anomaly detection algorithms
- Periodicity and frequency analysis
- Forecasting model preparation

---

## 🚀 Getting Started

### 1. **Choose Your Template**
Navigate to the appropriate folder based on your task:
- Need to clean data? → `data_cleaning/`
- Want to explore patterns? → `exploratory_data_analysis/`
- Need statistical tests? → `essential_statistics/`
- Creating visualizations? → `data_visuals/`
- Building models? → `statistical_models/`
- Feature engineering? → `feature_engineering/`
- Model evaluation & optimization? → `model_evaluation/`
- Advanced visualizations? → `advanced_visualizations/`
- Data pipeline automation? → `data_pipeline/`
- Time series analysis? → `time_series/`

### 2. **Replace Data Source**
In each template, replace the first line:
```python
df = pd.read_csv('your_data.csv')
```
with your actual data source.

### 3. **Update Column Names**
Replace placeholder column names like:
- `'target_column'` → Your actual target variable
- `'group_column'` → Your grouping variable
- `'date_column'` → Your date column

### 4. **Run and Adapt**
Each template is designed to work out-of-the-box and provide immediate insights. Adapt the parameters and sections to your specific needs.

## 📚 Template Features

### ✅ **What's Included**
- **Real code** that handles actual data processing
- **Statistical best practices** and assumption checking
- **Comprehensive visualizations** with proper formatting
- **Error handling** and edge case management
- **Interpretation guidance** and recommendations
- **Performance metrics** and validation techniques

### ✅ **Quality Standards**
- **No placeholder functions** - Everything is working code
- **Intermediate-to-advanced level** - Assumes basic Python knowledge
- **Industry best practices** - Follows data science standards
- **Reproducible results** - Includes random seeds where applicable
- **Memory efficient** - Optimized for large datasets

## 🔧 Required Libraries

All templates use only standard data science libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

## 📖 Usage Examples

### Example 1: Quick Data Cleaning
```python
# 1. Copy data_cleaning/handle_missing_data.py
# 2. Update: df = pd.read_csv('my_data.csv')
# 3. Run the script
# 4. Get comprehensive missing data analysis and multiple imputation options
```

### Example 2: Statistical Analysis
```python
# 1. Copy essential_statistics/inferential_stats.py  
# 2. Update column names for your grouping variables
# 3. Run for complete t-tests, ANOVA, chi-square analysis
# 4. Get publication-ready statistical results
```

### Example 3: Model Building
```python
# 1. Copy statistical_models/linear_regression.py
# 2. Set your target variable name
# 3. Run for complete regression analysis with diagnostics
# 4. Get model interpretation and validation results
```

## 🎯 Use Cases

### **Data Scientists**
- **Rapid prototyping** - Skip boilerplate code
- **Best practices** - Follow industry standards
- **Comprehensive analysis** - Don't miss important steps

### **Analysts**  
- **Statistical testing** - Proper procedures with interpretation
- **Visualization** - Report-ready charts and plots
- **Data quality** - Thorough cleaning and validation

### **Students**
- **Learning templates** - See how professionals structure analysis
- **Complete workflows** - End-to-end examples
- **Best practices** - Proper statistical procedures

### **Consultants**
- **Client deliverables** - Professional-quality outputs
- **Time efficiency** - Focus on insights, not coding
- **Standardization** - Consistent approach across projects

## ⚡ Quick Reference

| Task | Template | Key Output |
|------|----------|------------|
| Missing data | `data_cleaning/handle_missing_data.py` | Multiple imputation strategies |
| Data validation | `data_cleaning/data_validation.py` | **NEW** - Quality scores, validation reports |
| Duplicate handling | `data_cleaning/duplicate_handling.py` | **NEW** - Exact/fuzzy duplicate resolution |
| Outlier detection | `data_cleaning/outlier_detection.py` | Z-score, IQR, Isolation Forest results |
| Distribution analysis | `exploratory_data_analysis/variable_distributions.py` | Normality tests, shape statistics |
| Group comparisons | `exploratory_data_analysis/group_analysis.py` | T-tests, ANOVA, effect sizes |
| Correlations | `exploratory_data_analysis/correlation_analysis.py` | Correlation matrices, significance tests |
| Regression modeling | `statistical_models/linear_regression.py` | Full regression diagnostics |
| Classification | `statistical_models/logistic_regression.py` | ROC curves, confusion matrices |
| Clustering | `statistical_models/kmeans.py` | Optimal clusters, silhouette analysis |
| Feature creation | `feature_engineering/feature_creation.py` | New features from transformations, interactions |
| Feature selection | `feature_engineering/feature_selection.py` | Selected features, performance ranking |
| Model comparison | `model_evaluation/model_comparison.py` | Comparative model performance results |
| Advanced CV | `model_evaluation/cross_validation_advanced.py` | **NEW** - Stratified, nested, time series CV |
| Hyperparameter tuning | `model_evaluation/hyperparameter_optimization.py` | **NEW** - Grid/Bayesian/Optuna optimization |
| Model diagnostics | `model_evaluation/diagnostic_plots.py` | **NEW** - ROC, calibration, residual plots |
| Interactive visuals | `advanced_visualizations/interactive_plots.py` | HTML/PNG interactive plots, dashboards |
| Data pipeline | `data_pipeline/automated_pipeline.py` | Automated ETL pipeline with stats |
| Time series analysis | `time_series/time_series_analysis.py` | Trend, seasonality, and forecasting results |

## 📝 Notes

- **File format**: All templates are `.py` files (not Jupyter notebooks)
- **Dependencies**: Only standard libraries - no custom installations
- **Modularity**: Each file is self-contained and independent
- **Flexibility**: Templates are starting points - adapt to your needs
- **Documentation**: Comprehensive inline comments explain each step

## 🤝 Best Practices

1. **Always explore your data first** - Start with EDA templates
2. **Check assumptions** - Use diagnostic sections in model templates  
3. **Validate results** - Cross-validation and holdout testing included
4. **Document findings** - Templates include interpretation sections
5. **Iterate and improve** - Use diagnostics to refine your approach

---

**Ready to analyze your data? Pick a template and start coding! 🚀**

> **Note**: These templates provide comprehensive starting points for data analysis. Always validate results with domain knowledge and adapt code to your specific requirements.
