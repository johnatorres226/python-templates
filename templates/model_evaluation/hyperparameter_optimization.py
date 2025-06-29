"""
===============================================================================
HYPERPARAMETER OPTIMIZATION TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive hyperparameter optimization strategies

This template covers:
- Grid Search with cross-validation
- Random Search optimization
- Bayesian optimization with scikit-optimize
- Optuna framework for advanced optimization
- Hyperband and successive halving
- Multi-objective optimization
- Custom optimization strategies

Prerequisites:
- pandas, numpy, scikit-learn, matplotlib, seaborn
- pip install scikit-optimize optuna
- Dataset loaded as 'df' with features and target
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, train_test_split, validation_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
import warnings
warnings.filterwarnings('ignore')

# Advanced optimization libraries (install if needed)
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import gp_minimize
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not available. Install with: pip install scikit-optimize")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===============================================================================
# LOAD AND PREPARE DATA
# ===============================================================================

# Load your dataset
# df = pd.read_csv('your_data.csv')

# Create comprehensive synthetic dataset for demonstration
n_samples = 2000
n_features = 20

# Generate features with different scales and distributions
X = np.random.randn(n_samples, n_features)
X[:, :5] *= 10  # Some features with larger scale
X[:, 5:10] = np.random.exponential(2, (n_samples, 5))  # Exponential features
X[:, 10:15] = np.random.uniform(-5, 5, (n_samples, 5))  # Uniform features

# Create target with non-linear relationships
y = (X[:, 0] > 0).astype(int)  # Basic split
y += (X[:, 1] * X[:, 2] > 2).astype(int)  # Interaction effect
y += (np.sin(X[:, 3]) > 0.5).astype(int)  # Non-linear effect
y = np.clip(y, 0, 1)  # Binary classification

# Add noise and create multi-class
noise = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])
y = (y + noise) % 3  # 3-class problem

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Dataset Shape:", df.shape)
print("Class Distribution:")
print(df['target'].value_counts().sort_index())

# Split the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ===============================================================================
# 1. GRID SEARCH OPTIMIZATION
# ===============================================================================

print("\n" + "="*60)
print("1. GRID SEARCH OPTIMIZATION")
print("="*60)

# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search for Random Forest
print("1.1 Random Forest Grid Search")
print("-" * 40)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Perform grid search
start_time = time.time()
rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train, y_train)
rf_grid_time = time.time() - start_time

print(f"Grid Search completed in {rf_grid_time:.2f} seconds")
print(f"Best parameters: {rf_grid_search.best_params_}")
print(f"Best cross-validation score: {rf_grid_search.best_score_:.4f}")

# Test set performance
rf_best_pred = rf_grid_search.predict(X_test)
rf_test_accuracy = accuracy_score(y_test, rf_best_pred)
print(f"Test set accuracy: {rf_test_accuracy:.4f}")

# Grid Search for SVM
print("\n1.2 SVM Grid Search")
print("-" * 40)

svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

start_time = time.time()
svm_grid_search = GridSearchCV(
    SVC(random_state=42),
    svm_param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

svm_grid_search.fit(X_train, y_train)
svm_grid_time = time.time() - start_time

print(f"SVM Grid Search completed in {svm_grid_time:.2f} seconds")
print(f"Best parameters: {svm_grid_search.best_params_}")
print(f"Best cross-validation score: {svm_grid_search.best_score_:.4f}")

# Visualize grid search results
def plot_grid_search_results(grid_search, param1, param2, title):
    """Plot heatmap of grid search results for two parameters"""
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Extract parameter values and scores
    param1_values = results_df[f'param_{param1}'].values
    param2_values = results_df[f'param_{param2}'].values
    scores = results_df['mean_test_score'].values
    
    # Create pivot table for heatmap
    pivot_table = pd.DataFrame({
        param1: param1_values,
        param2: param2_values,
        'score': scores
    }).pivot(index=param2, columns=param1, values='score')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'{title}: {param1} vs {param2}')
    plt.tight_layout()
    plt.show()

# Plot Random Forest grid search results (subset)
rf_results_subset = pd.DataFrame(rf_grid_search.cv_results_)
rf_subset = rf_results_subset[rf_results_subset['param_max_features'] == 'sqrt']

if len(rf_subset) > 0:
    pivot_rf = rf_subset.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_n_estimators',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_rf, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Random Forest Grid Search: Max Depth vs N Estimators (max_features=sqrt)')
    plt.tight_layout()
    plt.show()

# ===============================================================================
# 2. RANDOM SEARCH OPTIMIZATION
# ===============================================================================

print("\n" + "="*60)
print("2. RANDOM SEARCH OPTIMIZATION")
print("="*60)

from scipy.stats import randint, uniform

# Random search parameter distributions
rf_random_params = {
    'n_estimators': randint(50, 500),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Random search for Random Forest
print("2.1 Random Forest Random Search")
print("-" * 40)

start_time = time.time()
rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_random_params,
    n_iter=100,  # Number of parameter combinations to try
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_random_search.fit(X_train, y_train)
rf_random_time = time.time() - start_time

print(f"Random Search completed in {rf_random_time:.2f} seconds")
print(f"Best parameters: {rf_random_search.best_params_}")
print(f"Best cross-validation score: {rf_random_search.best_score_:.4f}")

# Compare with grid search
print(f"\nComparison (Random Forest):")
print(f"Grid Search:   Time: {rf_grid_time:.2f}s, Score: {rf_grid_search.best_score_:.4f}")
print(f"Random Search: Time: {rf_random_time:.2f}s, Score: {rf_random_search.best_score_:.4f}")

# Random search for Gradient Boosting
print("\n2.2 Gradient Boosting Random Search")
print("-" * 40)

gb_random_params = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'subsample': uniform(0.7, 0.3)
}

start_time = time.time()
gb_random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_random_params,
    n_iter=50,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

gb_random_search.fit(X_train, y_train)
gb_random_time = time.time() - start_time

print(f"Gradient Boosting Random Search completed in {gb_random_time:.2f} seconds")
print(f"Best parameters: {gb_random_search.best_params_}")
print(f"Best cross-validation score: {gb_random_search.best_score_:.4f}")

# Analyze random search exploration
def analyze_random_search(random_search, param_name):
    """Analyze how random search explored parameter space"""
    results_df = pd.DataFrame(random_search.cv_results_)
    param_values = results_df[f'param_{param_name}']
    scores = results_df['mean_test_score']
    
    plt.figure(figsize=(12, 6))
    plt.scatter(param_values, scores, alpha=0.6)
    plt.xlabel(param_name)
    plt.ylabel('Cross-validation Score')
    plt.title(f'Random Search Exploration: {param_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Analyze learning rate exploration for Gradient Boosting
analyze_random_search(gb_random_search, 'learning_rate')

# ===============================================================================
# 3. BAYESIAN OPTIMIZATION
# ===============================================================================

if SKOPT_AVAILABLE:
    print("\n" + "="*60)
    print("3. BAYESIAN OPTIMIZATION")
    print("="*60)
    
    # Define search space for Bayesian optimization
    search_space = [
        Integer(50, 300, name='n_estimators'),
        Real(0.01, 0.3, name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Real(0.7, 1.0, name='subsample')
    ]
    
    # Bayesian search for Gradient Boosting
    print("3.1 Bayesian Optimization with scikit-optimize")
    print("-" * 50)
    
    start_time = time.time()
    bayes_search = BayesSearchCV(
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 0.3),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 20),
            'subsample': Real(0.7, 1.0)
        },
        n_iter=50,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    bayes_search.fit(X_train, y_train)
    bayes_time = time.time() - start_time
    
    print(f"Bayesian Optimization completed in {bayes_time:.2f} seconds")
    print(f"Best parameters: {bayes_search.best_params_}")
    print(f"Best cross-validation score: {bayes_search.best_score_:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(12, 6))
    scores = [-score for score in bayes_search.optimizer_results_[0].func_vals]
    plt.plot(scores, 'b.-')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-validation Score')
    plt.title('Bayesian Optimization Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Compare optimization methods
    print(f"\nOptimization Method Comparison:")
    print(f"Random Search: Time: {gb_random_time:.2f}s, Score: {gb_random_search.best_score_:.4f}")
    print(f"Bayesian Opt:  Time: {bayes_time:.2f}s, Score: {bayes_search.best_score_:.4f}")

# ===============================================================================
# 4. OPTUNA OPTIMIZATION
# ===============================================================================

if OPTUNA_AVAILABLE:
    print("\n" + "="*60)
    print("4. OPTUNA OPTIMIZATION")
    print("="*60)
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        """Objective function for Optuna optimization"""
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        subsample = trial.suggest_float('subsample', 0.7, 1.0)
        
        # Create model
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            random_state=42
        )
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        return scores.mean()
    
    print("4.1 Basic Optuna Optimization")
    print("-" * 40)
    
    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    optuna_time = time.time() - start_time
    
    print(f"Optuna optimization completed in {optuna_time:.2f} seconds")
    print(f"Best parameters: {study.best_params}")
    print(f"Best cross-validation score: {study.best_value:.4f}")
    
    # Plot optimization history
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0])
    axes[0].set_title('Optimization History')
    
    # Parameter importance
    try:
        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[1])
        axes[1].set_title('Parameter Importance')
    except:
        axes[1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Multi-objective optimization
    print("\n4.2 Multi-objective Optimization")
    print("-" * 40)
    
    def multi_objective(trial):
        """Multi-objective: maximize accuracy, minimize complexity"""
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        
        # Create model
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        
        # Accuracy objective
        scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        accuracy = scores.mean()
        
        # Complexity objective (lower is better)
        complexity = n_estimators * max_depth / 1000  # Normalized complexity
        
        return accuracy, -complexity  # Maximize accuracy, minimize complexity
    
    multi_study = optuna.create_study(
        directions=['maximize', 'maximize'],  # Both objectives to maximize
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    multi_study.optimize(multi_objective, n_trials=50, show_progress_bar=True)
    
    print(f"Multi-objective optimization completed")
    print(f"Number of Pareto solutions: {len(multi_study.best_trials)}")
    
    # Show Pareto front solutions
    pareto_solutions = []
    for trial in multi_study.best_trials:
        pareto_solutions.append({
            'accuracy': trial.values[0],
            'complexity': -trial.values[1],
            'params': trial.params
        })
    
    pareto_df = pd.DataFrame(pareto_solutions)
    print("\nTop 5 Pareto Solutions:")
    print(pareto_df.head().to_string(index=False))

# ===============================================================================
# 5. ADVANCED HYPERPARAMETER STRATEGIES
# ===============================================================================

print("\n" + "="*60)
print("5. ADVANCED HYPERPARAMETER STRATEGIES")
print("="*60)

# Successive Halving (if available)
try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV
    
    print("5.1 Successive Halving")
    print("-" * 30)
    
    halving_search = HalvingRandomSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_random_params,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start_time = time.time()
    halving_search.fit(X_train, y_train)
    halving_time = time.time() - start_time
    
    print(f"Successive Halving completed in {halving_time:.2f} seconds")
    print(f"Best parameters: {halving_search.best_params_}")
    print(f"Best cross-validation score: {halving_search.best_score_:.4f}")
    
except ImportError:
    print("5.1 Successive Halving not available in this sklearn version")

# Pipeline optimization
print("\n5.2 Pipeline Hyperparameter Optimization")
print("-" * 45)

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define parameter grid for pipeline
pipeline_params = {
    'scaler': [StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

pipeline_search = GridSearchCV(
    pipeline,
    pipeline_params,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
pipeline_search.fit(X_train, y_train)
pipeline_time = time.time() - start_time

print(f"Pipeline optimization completed in {pipeline_time:.2f} seconds")
print(f"Best parameters: {pipeline_search.best_params_}")
print(f"Best cross-validation score: {pipeline_search.best_score_:.4f}")

# Model selection with hyperparameter optimization
print("\n5.3 Model Selection + Hyperparameter Optimization")
print("-" * 55)

# Define models and their parameter spaces
models_and_params = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
}

# Optimize each model
model_results = {}
for model_name, model_config in models_and_params.items():
    print(f"\nOptimizing {model_name}...")
    
    grid_search = GridSearchCV(
        model_config['model'],
        model_config['params'],
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    # Test performance
    test_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    model_results[model_name] = {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'test_accuracy': test_accuracy,
        'time': end_time - start_time,
        'best_estimator': grid_search.best_estimator_
    }
    
    print(f"{model_name}: CV Score: {grid_search.best_score_:.4f}, "
          f"Test Score: {test_accuracy:.4f}, Time: {end_time - start_time:.2f}s")

# Model comparison summary
print(f"\n{'Model':<20} {'CV Score':<12} {'Test Score':<12} {'Time (s)':<10}")
print("-" * 55)
for model_name, results in model_results.items():
    print(f"{model_name:<20} {results['best_score']:<12.4f} "
          f"{results['test_accuracy']:<12.4f} {results['time']:<10.2f}")

# ===============================================================================
# 6. HYPERPARAMETER OPTIMIZATION ANALYSIS
# ===============================================================================

print("\n" + "="*60)
print("6. HYPERPARAMETER OPTIMIZATION ANALYSIS")
print("="*60)

# Learning curve for best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_accuracy'])
best_model = model_results[best_model_name]['best_estimator']

print(f"Best performing model: {best_model_name}")
print(f"Best parameters: {model_results[best_model_name]['best_params']}")

# Validation curve for key hyperparameter
if best_model_name == 'RandomForest':
    param_name = 'n_estimators'
    param_range = [10, 50, 100, 200, 300, 500]
elif best_model_name == 'GradientBoosting':
    param_name = 'n_estimators'
    param_range = [10, 50, 100, 200, 300]
else:  # SVM
    param_name = 'C'
    param_range = [0.01, 0.1, 1, 10, 100]

# Create base model for validation curve
base_params = model_results[best_model_name]['best_params'].copy()
if param_name in base_params:
    del base_params[param_name]

if best_model_name == 'RandomForest':
    base_model = RandomForestClassifier(random_state=42, **base_params)
elif best_model_name == 'GradientBoosting':
    base_model = GradientBoostingClassifier(random_state=42, **base_params)
else:
    base_model = SVC(random_state=42, **base_params)

train_scores, test_scores = validation_curve(
    base_model, X_train, y_train,
    param_name=param_name,
    param_range=param_range,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1
)

# Plot validation curve
plt.figure(figsize=(12, 8))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 'o-', color='blue', label='Training accuracy')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(param_range, test_mean, 'o-', color='red', label='Cross-validation accuracy')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

plt.xlabel(param_name)
plt.ylabel('Accuracy')
plt.title(f'Validation Curve: {best_model_name} - {param_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================================================================
# 7. OPTIMIZATION SUMMARY AND BEST PRACTICES
# ===============================================================================

print("\n" + "="*60)
print("7. HYPERPARAMETER OPTIMIZATION SUMMARY")
print("="*60)

# Create optimization method comparison
optimization_methods = {
    'Method': ['Grid Search', 'Random Search', 'Bayesian Opt', 'Optuna', 'Successive Halving'],
    'Best For': [
        'Small parameter spaces',
        'Large parameter spaces',
        'Expensive evaluations',
        'Complex optimization',
        'Large datasets'
    ],
    'Pros': [
        'Exhaustive search',
        'Efficient exploration',
        'Smart exploration',
        'Advanced features',
        'Early stopping'
    ],
    'Cons': [
        'Computationally expensive',
        'May miss optimal',
        'Setup complexity',
        'Additional dependency',
        'Less thorough'
    ],
    'When to Use': [
        '< 100 combinations',
        '> 100 combinations',
        'Continuous parameters',
        'Multi-objective',
        'Resource constraints'
    ]
}

optimization_df = pd.DataFrame(optimization_methods)
print("Hyperparameter Optimization Method Comparison:")
print(optimization_df.to_string(index=False))

# Best practices
print("\n" + "="*50)
print("HYPERPARAMETER OPTIMIZATION BEST PRACTICES")
print("="*50)

best_practices = [
    "1. Start with default parameters as baseline",
    "2. Use random search before grid search",
    "3. Apply proper cross-validation strategy",
    "4. Consider Bayesian optimization for expensive models",
    "5. Use early stopping to save computation",
    "6. Optimize preprocessing parameters too",
    "7. Consider multi-objective optimization",
    "8. Validate on separate test set",
    "9. Monitor for overfitting",
    "10. Document optimization process"
]

for practice in best_practices:
    print(practice)

# Parameter selection guidelines
print(f"\n" + "="*40)
print("PARAMETER SELECTION GUIDELINES")
print("="*40)

guidelines = {
    'Random Forest': [
        'n_estimators: Start with 100, increase if underfitting',
        'max_depth: None for small datasets, 3-10 for large',
        'min_samples_split: 2-20, higher for noisy data',
        'max_features: sqrt for classification, 1/3 for regression'
    ],
    'Gradient Boosting': [
        'learning_rate: 0.01-0.3, lower = more estimators',
        'n_estimators: 100-1000, use early stopping',
        'max_depth: 3-8, deeper for complex patterns',
        'subsample: 0.8-1.0, <1.0 adds regularization'
    ],
    'SVM': [
        'C: 0.1-100, higher for complex boundaries',
        'kernel: rbf for non-linear, linear for high-dim',
        'gamma: auto/scale, or 0.001-1',
        'Use StandardScaler preprocessing'
    ]
}

for model, tips in guidelines.items():
    print(f"\n{model}:")
    for tip in tips:
        print(f"  • {tip}")

print(f"\nHyperparameter optimization analysis complete!")
print(f"Best model: {best_model_name} with {model_results[best_model_name]['test_accuracy']:.4f} test accuracy")
