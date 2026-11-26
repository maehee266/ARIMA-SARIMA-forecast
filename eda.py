import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# load dataset
df = pd.read_csv('data/retail_store_inventory.csv')

# dropping demand forecast column
df = df.drop(columns=['Demand Forecast'])

# converting date column into datetime
df['Date'] = pd.to_datetime(df['Date'])

# summary table of dataset
summary_table = pd.DataFrame({
    'Feature': [
        'Number of observations',
        'Number of columns',
        'Time period',
        'Number of stores',
        'Number of products',
        'Target variable'
    ],
    'Description': [
        df.shape[0],
        df.shape[1],
        f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        df['Store ID'].nunique(),
        df['Product ID'].nunique(),
        "Units Sold"
    ]
})

# self explanatory
# print(summary_table)


# aggregate Toys category across all of the stores 
toys = df[df['Category'] == 'Toys'].groupby('Date')['Units Sold'].sum()
toys = toys.sort_index()

# time series plot jan 2022 - jan 2024

plt.figure(figsize=(12,4))
plt.plot(toys, color='steelblue')
plt.title('Daily Units Sold - Toys Category (All Stores Combined)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.tight_layout()
plt.savefig('figures/ts_plot.png', dpi=150)
plt.show()

# moving average and SD

plt.figure(figsize=(12,4))
rolling_mean = toys.rolling(window=30).mean()
rolling_std = toys.rolling(window=30).std()

plt.plot(toys, color='steelblue', alpha=0.5, label='Original')
plt.plot(rolling_mean, color='red', label='30-day Rolling Mean')
plt.plot(rolling_std, color='black', label='30-day Rolling Std')
plt.title('Rolling Mean and Standard Deviation — Toys Category', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.savefig('figures/rolling.png', dpi=150)
plt.show()


# ADF Test

result = adfuller(toys)
print('ADF Statistic', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value:.4f}')

# seasonal decomposition 


decomp = seasonal_decompose(toys, model='additive', period=7)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig('figures/decomposition.png', dpi=150)
plt.show()

# ACF AND PACF PLOTS

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(toys, lags=40, ax=axes[0])
plot_pacf(toys, lags=40, ax=axes[1])
plt.tight_layout()
plt.savefig('figures/acf_pacf.png', dpi=150)
plt.show()

# monthly average bar chart

toys_monthly = toys.resample('ME').mean()
toys_monthly.index = toys_monthly.index.strftime('%Y-%m')
toys_monthly.plot(kind='bar', figsize=(12,4), 
                  title='Average Monthly Units Sold — Toys Category')
plt.xlabel('Month')
plt.ylabel('Average Units Sold')
plt.tight_layout()
plt.savefig('figures/monthly_avg.png', dpi=150)
plt.show()

# avg Units Sold on holidays vs non-holiday (re-uses df loaded above)
toys_df = df[df['Category'] == 'Toys']
print(toys_df.groupby('Holiday/Promotion')['Units Sold'].mean().round(2))
