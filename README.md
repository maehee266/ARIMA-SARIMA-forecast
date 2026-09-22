# ARIMA-SARIMA-forecast
# Retail Demand Forecasting with ARIMA & SARIMA

Time-series forecasting project predicting retail product demand using classical statistical models, with results translated into actionable inventory planning metrics.

## Overview

Using a dataset of 73,100 retail demand records spanning 5 stores and 20 products, this project builds and compares ARIMA and SARIMA models to forecast future demand. The goal was to move beyond simple historical averages and produce statistically grounded forecasts that could inform real inventory decisions — specifically, how much stock to hold and when to reorder.

## Methodology

**Exploratory Data Analysis**
- Stationarity testing via the Augmented Dickey-Fuller test
- Seasonal decomposition to isolate trend, seasonal, and residual components
- ACF/PACF analysis to guide model order selection

**Model Selection**
- Grid search across 9 parameter combinations
- AIC/BIC diagnostics used to balance model fit against complexity, rather than optimising purely for training error

**Validation**
- Rolling-origin cross-validation across 10 evaluation windows, testing performance on sequential out-of-sample periods rather than a single train/test split
- Identified and corrected a numerical convergence failure in the seasonal model that was silently producing a misleading 100% error rate

## Results

- **21.6% MAPE** (Mean Absolute Percentage Error) on a 147-observation holdout set, benchmarked against published retail forecasting literature
- Forecast confidence intervals converted into inventory planning metrics:
  - **Safety stock:** 2,969 units
  - **Reorder point:** 22,044 units
  - Calculated at a 95% service level

## Tech Stack

- Python
- Pandas
- Statsmodels

## Key Takeaways

Validating a model's output matters as much as building it. The convergence failure in the seasonal model would have gone unnoticed without close inspection of the cross-validation results — a reminder that a clean-looking error metric can hide a broken model underneath.

## Project Structure

```
├── data/               # Raw and processed demand data
├── notebooks/          # EDA, model selection, and validation notebooks
├── src/                # Model training and evaluation scripts
└── README.md
```

## Author

Mary Heng
