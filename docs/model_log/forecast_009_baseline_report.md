# FORECAST-009 — Leakage-Safe Forecasting Baseline Report

## 1. Objective and target definition

Forecast daily **Revenue** and **COGS** for the submission horizon (2023-01-01 to 2024-07-01) using only historical data up to 2022-12-31.

## 2. Data used

- Training time series: `data/sales.csv` (Date, Revenue, COGS)

- Submission template: `sample_submission.csv` (Date order only; Revenue/COGS ignored)

- Sales history: 2012-07-04 to 2022-12-31 (3,833 daily rows; continuous)

- Submission horizon: 2023-01-01 to 2024-07-01 (548 daily rows; continuous)

## 3. Leakage checklist

- No external data used.

- `sample_submission.csv` Revenue/COGS not used as truth or features (Date only).

- Rolling features are shifted by 1 day (no same-day target leakage).

- Validation is time-based (no random split).

- Recursive forecasting used for models with lags/rolling features.

## 4. Validation design

Primary: holdout last **548** days of sales history (matches submission horizon length).

Secondary (computed for robustness): holdout last 365 days, plus rolling-origin year splits for 2020/2021/2022.

## 5. Model candidates

- Naive last value
- Seasonal naive (364-day)
- Seasonal naive (365-day)
- Calendar profile (day-of-year/month/day-of-week averages)
- ML lag models (if sklearn available): Ridge, ElasticNet, RandomForest, ExtraTrees, HistGradientBoosting, GradientBoosting

## 6. Metrics table (primary holdout_548)

| model              | target  | mae          | rmse         | r2      | mape    |
| ------------------ | ------- | ------------ | ------------ | ------- | ------- |
| extra_trees        | COGS    | 574,777.48   | 786,175.60   | 0.6694  | 22.71%  |
| random_forest      | COGS    | 582,480.05   | 797,751.09   | 0.6596  | 23.90%  |
| seasonal_naive_365 | COGS    | 601,979.84   | 830,134.37   | 0.6314  | 24.08%  |
| hist_gbdt          | COGS    | 702,558.02   | 864,508.24   | 0.6002  | 36.52%  |
| gbr                | COGS    | 781,659.08   | 956,381.71   | 0.5107  | 39.42%  |
| seasonal_naive_364 | COGS    | 706,504.85   | 1,048,683.79 | 0.4118  | 27.93%  |
| calendar_profile   | COGS    | 1,161,705.61 | 1,435,689.92 | -0.1025 | 53.98%  |
| elasticnet         | COGS    | 1,324,677.13 | 1,514,501.04 | -0.2269 | 58.84%  |
| ridge              | COGS    | 1,393,732.62 | 1,621,767.17 | -0.4068 | 62.81%  |
| naive_last         | COGS    | 2,557,863.54 | 2,777,891.24 | -3.1276 | 150.46% |
| extra_trees        | Revenue | 670,322.88   | 907,273.69   | 0.6587  | 26.19%  |
| hist_gbdt          | Revenue | 717,882.22   | 950,580.80   | 0.6254  | 30.55%  |
| seasonal_naive_365 | Revenue | 724,009.37   | 1,022,856.30 | 0.5662  | 27.48%  |
| random_forest      | Revenue | 801,330.78   | 1,051,455.38 | 0.5416  | 33.72%  |
| gbr                | Revenue | 935,609.72   | 1,193,909.19 | 0.4090  | 45.07%  |
| seasonal_naive_364 | Revenue | 833,585.89   | 1,248,987.77 | 0.3532  | 31.54%  |
| elasticnet         | Revenue | 1,436,129.42 | 1,691,891.80 | -0.1868 | 57.60%  |
| calendar_profile   | Revenue | 1,450,819.90 | 1,767,769.15 | -0.2957 | 62.08%  |
| ridge              | Revenue | 1,609,546.68 | 1,889,697.58 | -0.4806 | 65.89%  |
| naive_last         | Revenue | 2,350,052.00 | 2,612,374.62 | -1.8296 | 129.55% |


## 7. Selected model and rationale

- Revenue model: **extra_trees** (selected by lowest RMSE then MAE on holdout_548)

- COGS model: **extra_trees** (selected by lowest RMSE then MAE on holdout_548)

Selection prioritizes Revenue error (competition focus) while producing a reasonable COGS forecast for submission schema.

## 8. Explainability summary

Feature importance is derived from model-native importance (trees), absolute coefficients (linear), or permutation importance (fallback).

### Top features — Revenue

| feature         | importance | method                     |
| --------------- | ---------- | -------------------------- |
| lag_1           | 0.364273   | model.feature_importances_ |
| lag_365         | 0.158396   | model.feature_importances_ |
| lag_364         | 0.109604   | model.feature_importances_ |
| rolling_mean_7  | 0.076051   | model.feature_importances_ |
| rolling_mean_14 | 0.051712   | model.feature_importances_ |
| rolling_mean_28 | 0.043975   | model.feature_importances_ |
| day             | 0.024147   | model.feature_importances_ |
| lag_7           | 0.020311   | model.feature_importances_ |
| lag_30          | 0.016473   | model.feature_importances_ |
| lag_28          | 0.015623   | model.feature_importances_ |


### Top features — COGS

| feature         | importance | method                     |
| --------------- | ---------- | -------------------------- |
| lag_1           | 0.324759   | model.feature_importances_ |
| lag_365         | 0.222177   | model.feature_importances_ |
| lag_364         | 0.149390   | model.feature_importances_ |
| rolling_mean_7  | 0.049760   | model.feature_importances_ |
| rolling_mean_28 | 0.034105   | model.feature_importances_ |
| rolling_mean_14 | 0.030399   | model.feature_importances_ |
| lag_30          | 0.018941   | model.feature_importances_ |
| day             | 0.018320   | model.feature_importances_ |
| lag_7           | 0.018111   | model.feature_importances_ |
| rolling_mean_56 | 0.015833   | model.feature_importances_ |


## 9. Error analysis

Error slices were computed on the primary holdout validation by month, day-of-week, year-month, and high vs normal days. See `artifacts/tables/forecast_009_error_analysis.csv`.

## 10. Submission path and format validation

- Submission written to: `artifacts/submissions/submission_forecast_009.csv`

- Schema: Date, Revenue, COGS

- Row order: Date matches sample_submission exactly (asserted in code).

- Sanity checks: Revenue>0, COGS>=0, and COGS capped to Revenue*1.4116 (historical p99 ratio).

## 11. Known limitations

- Univariate forecasting only (no exogenous regressors beyond calendar).
- Recursive multi-step forecasts can accumulate error across the horizon.
- No hyperparameter tuning beyond reasonable defaults (to reduce overfitting risk).

## 12. Next improvement ideas

- Try log-transform targets (log1p) with inverse transform to stabilize variance.
- Add robust seasonal components (e.g., STL decomposition) and model residuals.
- Evaluate simple ensembling (e.g., average of top 2–3 models) with time-based selection.
