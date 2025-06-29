"""
===============================================================================
FORECASTING TEMPLATE
===============================================================================
Author: [Your Name]
Date: [Current Date]
Project: [Project Name]
Description: Comprehensive time series forecasting with multiple algorithms

This template covers:
- ARIMA and SARIMA models
- Exponential smoothing methods
- Prophet forecasting
- Linear trend models
- Ensemble forecasting
- Forecast evaluation and validation

Prerequisites:
- pandas, numpy, matplotlib, seaborn, statsmodels
- Optional: prophet, scikit-learn
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# LOAD AND PREPARE TIME SERIES DATA
# ===============================================================================

# Load your time series dataset
# df = pd.read_csv('your_timeseries_data.csv', parse_dates=['date'], index_col='date')

# Sample time series creation for demonstration
np.random.seed(42)
dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
n_points = len(dates)

# Create synthetic time series with trend, seasonality, and noise
trend = np.linspace(100, 200, n_points)
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)  # Annual seasonality
weekly = 5 * np.sin(2 * np.pi * np.arange(n_points) / 7)  # Weekly seasonality
noise = np.random.normal(0, 5, n_points)
ts_values = trend + seasonal + weekly + noise

# Create DataFrame
df = pd.DataFrame({
    'value': ts_values,
    'date': dates
}).set_index('date')

# Add additional features
df['value_positive'] = np.maximum(df['value'], 0)  # Ensure positive values for some models

print("Time Series Dataset:")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Frequency: {df.index.freq}")
print("\nFirst few values:")
print(df.head())

# Plot the original time series
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(df.index, df['value'])
plt.title('Original Time Series')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
plt.plot(df.index[-365:], df['value'][-365:])  # Last year
plt.title('Last Year Detail')
plt.ylabel('Value')
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# ===============================================================================
# 1. DATA PREPARATION AND TRAIN/TEST SPLIT
# ===============================================================================

print("\n" + "="*60)
print("1. DATA PREPARATION AND TRAIN/TEST SPLIT")
print("="*60)

# Split data into train and test sets
test_size = 90  # Last 3 months for testing
train_data = df[:-test_size]
test_data = df[-test_size:]

print(f"Training data: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} points)")
print(f"Test data: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} points)")

# Check stationarity
def check_stationarity(timeseries, title):
    """Check stationarity using ADF and KPSS tests"""
    print(f"\n{title}:")
    
    # ADF Test
    adf_result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {adf_result[0]:.6f}')
    print(f'p-value: {adf_result[1]:.6f}')
    print(f'ADF Critical Values: {adf_result[4]}')
    
    if adf_result[1] <= 0.05:
        print("ADF Test: Series is stationary (reject null hypothesis)")
    else:
        print("ADF Test: Series is non-stationary (fail to reject null hypothesis)")
    
    # KPSS Test
    kpss_result = kpss(timeseries, regression='c', nlags="auto")
    print(f'KPSS Statistic: {kpss_result[0]:.6f}')
    print(f'p-value: {kpss_result[1]:.6f}')
    print(f'KPSS Critical Values: {kpss_result[3]}')
    
    if kpss_result[1] <= 0.05:
        print("KPSS Test: Series is non-stationary (reject null hypothesis)")
    else:
        print("KPSS Test: Series is stationary (fail to reject null hypothesis)")

# Check stationarity of original series
check_stationarity(train_data['value'].dropna(), "Original Series Stationarity")

# Make series stationary if needed
train_diff = train_data['value'].diff().dropna()
check_stationarity(train_diff, "First Difference Stationarity")

# ===============================================================================
# 2. ARIMA MODELING
# ===============================================================================

print("\n" + "="*60)
print("2. ARIMA MODELING")
print("="*60)

def find_best_arima_order(ts, max_p=5, max_d=2, max_q=5):
    """Find best ARIMA order using AIC criterion"""
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    print("Searching for best ARIMA order...")
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                        
                except Exception as e:
                    continue
    
    print(f"Best ARIMA order: {best_order}")
    print(f"Best AIC: {best_aic:.2f}")
    
    return best_order, best_model

# Find best ARIMA model
best_arima_order, best_arima_model = find_best_arima_order(train_data['value'])

# Print model summary
print(f"\nARIMA{best_arima_order} Model Summary:")
print(best_arima_model.summary())

# Generate ARIMA forecasts
arima_forecast = best_arima_model.forecast(steps=len(test_data))
arima_conf_int = best_arima_model.get_forecast(steps=len(test_data)).conf_int()

# ===============================================================================
# 3. SARIMA MODELING
# ===============================================================================

print("\n" + "="*60)
print("3. SARIMA MODELING")
print("="*60)

def find_best_sarima_order(ts, seasonal_period=7):
    """Find best SARIMA order"""
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    print(f"Searching for best SARIMA order (seasonal period = {seasonal_period})...")
    
    # Simplified search for demonstration
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in [0, 1]:
                    for D in [0, 1]:
                        for Q in [0, 1]:
                            try:
                                model = SARIMAX(ts, 
                                               order=(p, d, q),
                                               seasonal_order=(P, D, Q, seasonal_period))
                                fitted_model = model.fit(disp=False)
                                aic = fitted_model.aic
                                
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, seasonal_period)
                                    best_model = fitted_model
                                    
                            except Exception as e:
                                continue
    
    print(f"Best SARIMA order: {best_order}")
    print(f"Best seasonal order: {best_seasonal_order}")
    print(f"Best AIC: {best_aic:.2f}")
    
    return best_order, best_seasonal_order, best_model

# Find best SARIMA model (using weekly seasonality)
best_sarima_order, best_seasonal_order, best_sarima_model = find_best_sarima_order(
    train_data['value'], seasonal_period=7
)

# Generate SARIMA forecasts
sarima_forecast = best_sarima_model.forecast(steps=len(test_data))
sarima_conf_int = best_sarima_model.get_forecast(steps=len(test_data)).conf_int()

# ===============================================================================
# 4. EXPONENTIAL SMOOTHING
# ===============================================================================

print("\n" + "="*60)
print("4. EXPONENTIAL SMOOTHING")
print("="*60)

# Simple Exponential Smoothing
ses_model = ExponentialSmoothing(train_data['value'], trend=None, seasonal=None)
ses_fitted = ses_model.fit()
ses_forecast = ses_fitted.forecast(steps=len(test_data))

print("Simple Exponential Smoothing fitted")
print(f"Alpha (smoothing parameter): {ses_fitted.params['smoothing_level']:.4f}")

# Holt's Linear Trend
holt_model = ExponentialSmoothing(train_data['value'], trend='add', seasonal=None)
holt_fitted = holt_model.fit()
holt_forecast = holt_fitted.forecast(steps=len(test_data))

print(f"\nHolt's Linear Trend fitted")
print(f"Alpha: {holt_fitted.params['smoothing_level']:.4f}")
print(f"Beta: {holt_fitted.params['smoothing_trend']:.4f}")

# Holt-Winters Seasonal
hw_model = ExponentialSmoothing(train_data['value'], 
                               trend='add', 
                               seasonal='add', 
                               seasonal_periods=7)
hw_fitted = hw_model.fit()
hw_forecast = hw_fitted.forecast(steps=len(test_data))

print(f"\nHolt-Winters Seasonal fitted")
print(f"Alpha: {hw_fitted.params['smoothing_level']:.4f}")
print(f"Beta: {hw_fitted.params['smoothing_trend']:.4f}")
print(f"Gamma: {hw_fitted.params['smoothing_seasonal']:.4f}")

# ===============================================================================
# 5. MACHINE LEARNING APPROACHES
# ===============================================================================

print("\n" + "="*60)
print("5. MACHINE LEARNING APPROACHES")
print("="*60)

def create_features(ts, lags=7):
    """Create lagged features for ML models"""
    df_features = pd.DataFrame(index=ts.index)
    
    # Lagged values
    for i in range(1, lags + 1):
        df_features[f'lag_{i}'] = ts.shift(i)
    
    # Rolling statistics
    for window in [3, 7, 14]:
        df_features[f'rolling_mean_{window}'] = ts.rolling(window=window).mean()
        df_features[f'rolling_std_{window}'] = ts.rolling(window=window).std()
    
    # Time-based features
    df_features['dayofweek'] = ts.index.dayofweek
    df_features['month'] = ts.index.month
    df_features['quarter'] = ts.index.quarter
    df_features['dayofyear'] = ts.index.dayofyear
    
    # Cyclical features
    df_features['sin_dayofweek'] = np.sin(2 * np.pi * ts.index.dayofweek / 7)
    df_features['cos_dayofweek'] = np.cos(2 * np.pi * ts.index.dayofweek / 7)
    df_features['sin_month'] = np.sin(2 * np.pi * ts.index.month / 12)
    df_features['cos_month'] = np.cos(2 * np.pi * ts.index.month / 12)
    
    return df_features.dropna()

# Create features
train_features = create_features(train_data['value'])
train_target = train_data['value'].loc[train_features.index]

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(train_features, train_target)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_features, train_target)

# Generate ML forecasts (requires recursive forecasting for multi-step ahead)
def recursive_forecast(model, last_features, steps):
    """Generate recursive forecasts"""
    forecasts = []
    current_features = last_features.copy()
    
    for step in range(steps):
        # Make prediction
        pred = model.predict(current_features.values.reshape(1, -1))[0]
        forecasts.append(pred)
        
        # Update features for next prediction
        # Shift lag features
        for i in range(6, 0, -1):  # lags 7 to 2
            current_features[f'lag_{i+1}'] = current_features[f'lag_{i}']
        current_features['lag_1'] = pred
        
        # Update rolling features (simplified)
        # In practice, you'd need to maintain a rolling window
        
    return np.array(forecasts)

# Get last features from training data
last_features = train_features.iloc[-1]

# Generate ML forecasts
lr_forecast = recursive_forecast(lr_model, last_features, len(test_data))
rf_forecast = recursive_forecast(rf_model, last_features, len(test_data))

print("Machine Learning models fitted:")
print(f"Linear Regression R²: {lr_model.score(train_features, train_target):.4f}")
print(f"Random Forest R²: {rf_model.score(train_features, train_target):.4f}")

# ===============================================================================
# 6. ENSEMBLE FORECASTING
# ===============================================================================

print("\n" + "="*60)
print("6. ENSEMBLE FORECASTING")
print("="*60)

# Simple average ensemble
ensemble_forecast = (arima_forecast + sarima_forecast + hw_forecast + 
                    lr_forecast + rf_forecast) / 5

# Weighted ensemble (you can optimize weights)
weights = [0.2, 0.25, 0.25, 0.15, 0.15]  # ARIMA, SARIMA, HW, LR, RF
weighted_ensemble = (weights[0] * arima_forecast + 
                    weights[1] * sarima_forecast +
                    weights[2] * hw_forecast +
                    weights[3] * lr_forecast +
                    weights[4] * rf_forecast)

print("Ensemble forecasts created:")
print("- Simple average ensemble")
print("- Weighted ensemble")

# ===============================================================================
# 7. FORECAST EVALUATION
# ===============================================================================

print("\n" + "="*60)
print("7. FORECAST EVALUATION")
print("="*60)

def evaluate_forecast(actual, predicted, model_name):
    """Evaluate forecast accuracy"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    
    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

# Evaluate all models
results = []
results.append(evaluate_forecast(test_data['value'], arima_forecast, 'ARIMA'))
results.append(evaluate_forecast(test_data['value'], sarima_forecast, 'SARIMA'))
results.append(evaluate_forecast(test_data['value'], ses_forecast, 'Simple ES'))
results.append(evaluate_forecast(test_data['value'], holt_forecast, 'Holt Linear'))
results.append(evaluate_forecast(test_data['value'], hw_forecast, 'Holt-Winters'))
results.append(evaluate_forecast(test_data['value'], lr_forecast, 'Linear Regression'))
results.append(evaluate_forecast(test_data['value'], rf_forecast, 'Random Forest'))
results.append(evaluate_forecast(test_data['value'], ensemble_forecast, 'Simple Ensemble'))
results.append(evaluate_forecast(test_data['value'], weighted_ensemble, 'Weighted Ensemble'))

# Convert to DataFrame and sort by RMSE
results_df = pd.DataFrame(results).sort_values('RMSE')

print("Forecast Evaluation Results (sorted by RMSE):")
print(results_df.round(4))

# ===============================================================================
# 8. VISUALIZATION OF FORECASTS
# ===============================================================================

print("\n" + "="*60)
print("8. VISUALIZATION OF FORECASTS")
print("="*60)

# Plot all forecasts
plt.figure(figsize=(20, 15))

# Main forecast plot
plt.subplot(3, 2, 1)
plt.plot(train_data.index[-180:], train_data['value'][-180:], 
         label='Training Data', color='blue')
plt.plot(test_data.index, test_data['value'], 
         label='Actual', color='black', linewidth=2)
plt.plot(test_data.index, arima_forecast, 
         label='ARIMA', alpha=0.7)
plt.plot(test_data.index, sarima_forecast, 
         label='SARIMA', alpha=0.7)
plt.plot(test_data.index, hw_forecast, 
         label='Holt-Winters', alpha=0.7)
plt.fill_between(test_data.index, 
                 arima_conf_int.iloc[:, 0], 
                 arima_conf_int.iloc[:, 1], 
                 alpha=0.2, label='ARIMA CI')
plt.title('Time Series Forecasts Comparison')
plt.legend()
plt.xticks(rotation=45)

# Individual model plots
models_data = [
    ('ARIMA', arima_forecast),
    ('SARIMA', sarima_forecast),
    ('Holt-Winters', hw_forecast),
    ('Linear Regression', lr_forecast),
    ('Ensemble', ensemble_forecast)
]

for i, (name, forecast) in enumerate(models_data):
    plt.subplot(3, 2, i + 2)
    plt.plot(train_data.index[-60:], train_data['value'][-60:], 
             label='Training', color='blue', alpha=0.7)
    plt.plot(test_data.index, test_data['value'], 
             label='Actual', color='black', linewidth=2)
    plt.plot(test_data.index, forecast, 
             label=f'{name} Forecast', color='red', linewidth=2)
    plt.title(f'{name} Forecast')
    plt.legend()
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Residual analysis
plt.figure(figsize=(15, 10))

for i, (name, forecast) in enumerate(models_data):
    residuals = test_data['value'] - forecast
    
    plt.subplot(3, 2, i + 1)
    plt.plot(test_data.index, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'{name} Residuals')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Model comparison bar plot
plt.figure(figsize=(12, 8))
x_pos = np.arange(len(results_df))

plt.subplot(2, 2, 1)
plt.bar(x_pos, results_df['MAE'])
plt.xlabel('Models')
plt.ylabel('MAE')
plt.title('Mean Absolute Error')
plt.xticks(x_pos, results_df['Model'], rotation=45)

plt.subplot(2, 2, 2)
plt.bar(x_pos, results_df['RMSE'])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Root Mean Square Error')
plt.xticks(x_pos, results_df['Model'], rotation=45)

plt.subplot(2, 2, 3)
plt.bar(x_pos, results_df['MAPE'])
plt.xlabel('Models')
plt.ylabel('MAPE (%)')
plt.title('Mean Absolute Percentage Error')
plt.xticks(x_pos, results_df['Model'], rotation=45)

plt.subplot(2, 2, 4)
plt.scatter(results_df['MAE'], results_df['RMSE'])
for i, model in enumerate(results_df['Model']):
    plt.annotate(model, (results_df['MAE'].iloc[i], results_df['RMSE'].iloc[i]))
plt.xlabel('MAE')
plt.ylabel('RMSE')
plt.title('MAE vs RMSE')

plt.tight_layout()
plt.show()

# ===============================================================================
# 9. FORECAST DIAGNOSTICS
# ===============================================================================

print("\n" + "="*60)
print("9. FORECAST DIAGNOSTICS")
print("="*60)

# Residual diagnostics for best statistical model
best_statistical_model = best_sarima_model  # Use SARIMA as example

# Ljung-Box test for residual autocorrelation
residuals = best_statistical_model.resid
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)

print("Ljung-Box Test for Residual Autocorrelation:")
print(lb_test)

# Plot diagnostics
plt.figure(figsize=(15, 12))

# Residuals plot
plt.subplot(2, 3, 1)
plt.plot(train_data.index, residuals)
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residuals')

# ACF of residuals
from statsmodels.tsa.stattools import acf
residuals_acf = acf(residuals, nlags=20)
plt.subplot(2, 3, 2)
plt.bar(range(len(residuals_acf)), residuals_acf)
plt.title('ACF of Residuals')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

# Q-Q plot of residuals
from scipy import stats
plt.subplot(2, 3, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

# Histogram of residuals
plt.subplot(2, 3, 4)
plt.hist(residuals, bins=20, density=True, alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Density')

# Residuals vs fitted
fitted_values = train_data['value'] - residuals
plt.subplot(2, 3, 5)
plt.scatter(fitted_values, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Forecast accuracy over time
test_residuals = test_data['value'] - sarima_forecast
plt.subplot(2, 3, 6)
plt.plot(test_data.index, np.abs(test_residuals))
plt.title('Absolute Forecast Errors')
plt.xlabel('Date')
plt.ylabel('Absolute Error')

plt.tight_layout()
plt.show()

# ===============================================================================
# 10. FORECAST SUMMARY AND RECOMMENDATIONS
# ===============================================================================

print("\n" + "="*60)
print("10. FORECAST SUMMARY AND RECOMMENDATIONS")
print("="*60)

# Best model identification
best_model = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE']
best_mape = results_df.iloc[0]['MAPE']

print(f"FORECASTING SUMMARY:")
print(f"================")
print(f"Best Model: {best_model}")
print(f"RMSE: {best_rmse:.4f}")
print(f"MAPE: {best_mape:.2f}%")

print(f"\nMODEL RANKINGS (by RMSE):")
print(f"=========================")
for i, row in results_df.iterrows():
    print(f"{row.name + 1}. {row['Model']}: RMSE = {row['RMSE']:.4f}")

print(f"\nRECOMMENDations:")
print(f"===============")
print("✓ Use ensemble methods for robust forecasting")
print("✓ Consider seasonal patterns in model selection")
print("✓ Validate forecasts with domain knowledge")
print("✓ Monitor forecast accuracy over time")
print("✓ Retrain models with new data regularly")

# Export forecasts
forecast_df = pd.DataFrame({
    'Date': test_data.index,
    'Actual': test_data['value'].values,
    'ARIMA': arima_forecast,
    'SARIMA': sarima_forecast,
    'Holt_Winters': hw_forecast,
    'Linear_Regression': lr_forecast,
    'Random_Forest': rf_forecast,
    'Ensemble': ensemble_forecast,
    'Weighted_Ensemble': weighted_ensemble
})

# forecast_df.to_csv('time_series_forecasts.csv', index=False)
print(f"\nForecasts ready for export!")
print(f"Forecast horizon: {len(test_data)} periods")
print(f"All models trained and evaluated successfully!")
