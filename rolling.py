import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# load data
df = pd.read_csv('data/retail_store_inventory.csv', parse_dates=['Date'])
df = df.drop(columns=['Demand Forecast'])
toys = df[df['Category'] == 'Toys'].groupby('Date')['Units Sold'].sum().sort_index()

# Rolling origin parameters
initial_train_size = int(len(toys) * 0.8)  # 584 observations
step = 10  # move forward 10 days after each iteration
n_iterations = 10  # repeating process 10 times

arima_rmse_list = []
arima_mae_list = []
arima_mape_list = []

sarima_rmse_list = []
sarima_mae_list = []
sarima_mape_list = []

print("Running rolling origin validation...")
print(f"Initial training size: {initial_train_size}")
print(f"Step size: {step} days")
print(f"Number of iterations: {n_iterations}")
print("---")

for i in range(n_iterations):
    # Define train and test windows
    train_end = initial_train_size + (i * step)
    test_start = train_end
    test_end = test_start + step

    # Make sure not to exceed the dataset
    if test_end > len(toys):
        break

    train = toys.iloc[:train_end]
    test = toys.iloc[test_start:test_end]

    # Fit ARIMA at each step
    arima_model = ARIMA(train, order=(0, 0, 0))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.get_forecast(steps=len(test))
    arima_mean = arima_forecast.predicted_mean

    # Fit SARIMA at each step 
    sarima_model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    sarima_fitted = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fitted.get_forecast(steps=len(test))
    sarima_mean = sarima_forecast.predicted_mean

    # Calculate metrics for ARIMA
    arima_rmse = np.sqrt(mean_squared_error(test, arima_mean))
    arima_mae = mean_absolute_error(test, arima_mean)
    arima_mape = np.mean(np.abs((test.values - arima_mean.values) / test.values)) * 100

    # Calculate metrics for SARIMA
    sarima_rmse = np.sqrt(mean_squared_error(test, sarima_mean))
    sarima_mae = mean_absolute_error(test, sarima_mean)
    sarima_mape = np.mean(np.abs((test.values - sarima_mean.values) / test.values)) * 100

    # store metrics 
    arima_rmse_list.append(arima_rmse)
    arima_mae_list.append(arima_mae)
    arima_mape_list.append(arima_mape)

    sarima_rmse_list.append(sarima_rmse)
    sarima_mae_list.append(sarima_mae)
    sarima_mape_list.append(sarima_mape)

    print(f"Window {i+1}: Train={train_end} obs | ARIMA MAPE={arima_mape:.2f}% | SARIMA MAPE={sarima_mape:.2f}%")

# Summary results
print("\n=== Rolling Origin Validation Results ===")
print(f"ARIMA  — Avg RMSE: {np.mean(arima_rmse_list):.2f} | Avg MAE: {np.mean(arima_mae_list):.2f} | Avg MAPE: {np.mean(arima_mape_list):.2f}%")
print(f"SARIMA — Avg RMSE: {np.mean(sarima_rmse_list):.2f} | Avg MAE: {np.mean(sarima_mae_list):.2f} | Avg MAPE: {np.mean(sarima_mape_list):.2f}%")

# Average excluding Window 8, where SARIMA failed to converge
if len(sarima_mape_list) >= 8:
    arima_excl = [m for j, m in enumerate(arima_mape_list) if j != 7]
    sarima_excl = [m for j, m in enumerate(sarima_mape_list) if j != 7]
    print(f"Excluding Window 8 — ARIMA Avg MAPE: {np.mean(arima_excl):.2f}% | SARIMA Avg MAPE: {np.mean(sarima_excl):.2f}%")

# Chart: MAPE per window for both models, Window 8 highlighted
windows = np.arange(1, len(arima_mape_list) + 1)
width = 0.38
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(windows - width / 2, arima_mape_list, width, label='ARIMA(0,0,0)', color='steelblue')
sarima_colors = ['indianred' if w == 8 else 'mediumpurple' for w in windows]
ax.bar(windows + width / 2, sarima_mape_list, width, label='SARIMA(1,0,1)(1,0,1,7)', color=sarima_colors)
if len(windows) >= 8:
    ax.annotate('SARIMA fails to converge\n(forecast ≈ 0)',
                xy=(8 + width / 2, sarima_mape_list[7]),
                xytext=(5.2, max(sarima_mape_list) * 0.85),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
ax.set_xticks(windows)
ax.set_xlabel('Rolling-origin window')
ax.set_ylabel('MAPE (%)')
ax.set_title('Rolling-Origin Validation: MAPE per Window — Toys Category')
ax.legend()
plt.tight_layout()
plt.savefig('figures/rolling_origin_mape.png', dpi=150)
plt.show()
