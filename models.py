import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats


# Load and prepare data
df = pd.read_csv('data/retail_store_inventory.csv', parse_dates=['Date'])
df = df.drop(columns=['Demand Forecast'])

# Create Toys series
toys = df[df['Category'] == 'Toys'].groupby('Date')['Units Sold'].sum()
toys = toys.sort_index()

# Split 80/20
train_size = int(len(toys) * 0.8)
train = toys[:train_size]
test = toys[train_size:]

print(f'Training set size: {len(train)} observations')
print(f'Test set size: {len(test)} observations')
print(f'Training period: {train.index[0]} to {train.index[-1]}')
print(f'Test period: {test.index[0]} to {test.index[-1]}')

# test combi of p and q with d = 0 based on EDA 
p_values = [0, 1, 2]
q_values = [0, 1, 2]
d = 0  # stationary from ADF test

results = []

for p, q in itertools.product(p_values, q_values):
    try:
        model = ARIMA(train, order=(p, d, q))
        fitted = model.fit()
        results.append({
            'p': p, 'd': d, 'q': q,
            'AIC': fitted.aic,
            'BIC': fitted.bic
        })
        print(f'ARIMA({p},{d},{q}) — AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}')
    except Exception as e:
        print(f'ARIMA({p},{d},{q}) — Failed to fit: {e}')

# find best model
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['AIC'].idxmin()]
print(f'\nBest model by AIC: ARIMA({int(best.p)},{int(best.d)},{int(best.q)})')
print(f'AIC: {best.AIC:.2f}, BIC: {best.BIC:.2f}')

# fit ARIMA(0,0,0) best by AIC
model_000 = ARIMA(train, order=(0, 0, 0))
fitted_000 = model_000.fit()

# also fit ARIMA(1,0,1) for comparison
model_101 = ARIMA(train, order=(1, 0, 1))
fitted_101 = model_101.fit()

# generate forecasts for both
forecast_000 = fitted_000.get_forecast(steps=len(test))
forecast_101 = fitted_101.get_forecast(steps=len(test))

forecast_mean_000 = forecast_000.predicted_mean
forecast_mean_101 = forecast_101.predicted_mean
forecast_ci_000 = forecast_000.conf_int()
forecast_ci_101 = forecast_101.conf_int()

# Plot both
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='Actual', color='green')
plt.plot(test.index, forecast_mean_000, label='ARIMA(0,0,0)', color='red')
plt.plot(test.index, forecast_mean_101, label='ARIMA(1,0,1)', color='orange')
plt.fill_between(test.index,
                 forecast_ci_000.iloc[:, 0],
                 forecast_ci_000.iloc[:, 1],
                 alpha=0.2, color='red')
plt.title('ARIMA Forecast vs Actual — Toys Category')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.savefig('figures/arima_forecast.png', dpi=150)
plt.show()

# calculating RMSE, MAE, MAPE for ARIMA (0,0,0)
rmse = np.sqrt(mean_squared_error(test, forecast_mean_000))
mae = mean_absolute_error(test, forecast_mean_000)
mape = np.mean(np.abs((test.values - forecast_mean_000.values) / test.values)) * 100

# calculating RMSE, MAE, MAPE for ARIMA (1,0,1)
rmse_101 = np.sqrt(mean_squared_error(test, forecast_mean_101))
mae_101 = mean_absolute_error(test, forecast_mean_101)
mape_101 = np.mean(np.abs((test.values - forecast_mean_101.values) / test.values)) * 100

print('=== ARIMA (0,0,0) Model Evaluation ===')
print(f'RMSE: {rmse:.2f}')
print(f'MAE:  {mae:.2f}')
print(f'MAPE: {mape:.2f}%')

print('=== ARIMA (1,0,1) Model Evaluation ===')
print(f'ARIMA(1,0,1) — RMSE: {rmse_101:.2f}, MAE: {mae_101:.2f}, MAPE: {mape_101:.2f}%')
print(f'AIC: {fitted_101.aic:.2f}, BIC: {fitted_101.bic:.2f}')


# SARIMA ----------------------------------------------------------------------------

# Test SARIMA combinations
p_values = [0, 1, 2]
q_values = [0, 1, 2]
P_values = [0, 1]
Q_values = [0, 1]
d = 0
D = 0
m = 7  # weekly seasonality

sarima_results = []


for p, q, P, Q in itertools.product(p_values, q_values, P_values, Q_values):
    try:
        model = SARIMAX(train, 
                       order=(p, d, q),
                       seasonal_order=(P, D, Q, m))
        fitted = model.fit(disp=False)
        sarima_results.append({
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q,
            'AIC': fitted.aic,
            'BIC': fitted.bic
        })
        print(f'SARIMA({p},{d},{q})({P},{D},{Q},7) — AIC: {fitted.aic:.2f}')
    except Exception as e:
        print(f'SARIMA({p},{d},{q})({P},{D},{Q},7) — Failed to fit: {e}')

# Find best SARIMA model
sarima_df = pd.DataFrame(sarima_results)
best_sarima = sarima_df.loc[sarima_df['AIC'].idxmin()]
print(f'\nBest SARIMA model: SARIMA({int(best_sarima.p)},{int(best_sarima.d)},{int(best_sarima.q)})({int(best_sarima.P)},{int(best_sarima.D)},{int(best_sarima.Q)},7)')
print(f'AIC: {best_sarima.AIC:.2f}, BIC: {best_sarima.BIC:.2f}')

# NOTE: the lowest-AIC SARIMA above is NOT used.
# One configuration returned an AIC of ~12, while every other model scored
# ~9,300-9,500, and its forecasts were close to zero. That is a numerical
# fitting failure, not a better model, so it is rejected.
# SARIMA(1,0,1)(1,0,1,7) is chosen instead, based on the ACF/PACF evidence
# and the weekly (m = 7) cycle found in the seasonal decomposition.
sarima_model = SARIMAX(train,
                       order=(1, 0, 1),
                       seasonal_order=(1, 0, 1, 7))
sarima_fitted = sarima_model.fit(disp=False)

print(f'SARIMA(1,0,1)(1,0,1,7) — AIC: {sarima_fitted.aic:.2f}, BIC: {sarima_fitted.bic:.2f}')

# Generate forecasts
sarima_forecast = sarima_fitted.get_forecast(steps=len(test))
sarima_mean = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int()

# Check forecast values look reasonable
print(f'Forecast mean range: {sarima_mean.min():.0f} to {sarima_mean.max():.0f}')
print(f'Actual test range: {test.min():.0f} to {test.max():.0f}')

# Plot SARIMA forecast
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label='Training Data', color='steelblue', alpha=0.5)
plt.plot(test.index, test, label='Actual', color='green')
plt.plot(test.index, sarima_mean, label='SARIMA Forecast', color='purple')
plt.fill_between(test.index,
                 sarima_ci.iloc[:, 0],
                 sarima_ci.iloc[:, 1],
                 alpha=0.2, color='purple', label='95% Confidence Interval')
plt.title('SARIMA(1,0,1)(1,0,1,7) Forecast vs Actual — Toys Category')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.savefig('figures/sarima_forecast.png', dpi=150)
plt.show()

# Evaluation metrics
sarima_rmse = np.sqrt(mean_squared_error(test, sarima_mean))
sarima_mae = mean_absolute_error(test, sarima_mean)
sarima_mape = np.mean(np.abs((test.values - sarima_mean.values) / test.values)) * 100

print('=== SARIMA Model Evaluation ===')
print(f'RMSE: {sarima_rmse:.2f}')
print(f'MAE:  {sarima_mae:.2f}')
print(f'MAPE: {sarima_mape:.2f}%')


#---------------------------------------------------------------------------------------------
# SAFETY STOCK AND REORDER POINT CALCULATION


# Using ARIMA(0,0,0) forecast and confidence intervals

# Forecast values from ARIMA
forecast_values = forecast_mean_000.values
ci_lower = forecast_ci_000.iloc[:, 0].values
ci_upper = forecast_ci_000.iloc[:, 1].values

# avg daily forecast demand
avg_daily_demand = forecast_values.mean()

# forecast error: standard deviation of the forecast errors on the 147-day test set
# (an alternative would be the in-sample residuals, which avoids using test data)
forecast_errors = test.values - forecast_values
forecast_std = np.std(forecast_errors)

# Service level = 95% (standard retail assumption)
# Z-score for 95% service level = 1.645
service_level = 0.95
z_score = stats.norm.ppf(service_level)

# one week replenishment cycle
lead_time = 7

# Safety Stock Formula
# Safety Stock = Z * forecast_std * sqrt(lead_time)
safety_stock = z_score * forecast_std * np.sqrt(lead_time)

# Reorder Point Formula
# ROP = (avg_daily_demand * lead_time) + safety_stock
reorder_point = (avg_daily_demand * lead_time) + safety_stock

# Maximum stock level (upper bound of confidence interval)
max_stock = ci_upper.mean()

print('=== Inventory Planning Metrics ===')
print(f'Average Daily Forecast Demand: {avg_daily_demand:.0f} units')
print(f'Forecast Standard Deviation:   {forecast_std:.0f} units')
print(f'Service Level:                 {service_level*100:.0f}%')
print(f'Lead Time:                     {lead_time} days')
print(f'Z-Score (95%):                 {z_score:.3f}')
print(f'')
print(f'Safety Stock:                  {safety_stock:.0f} units')
print(f'Reorder Point:                 {reorder_point:.0f} units')
print(f'Maximum Stock Level:           {max_stock:.0f} units')
