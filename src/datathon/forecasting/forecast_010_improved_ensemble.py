import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.optimize as optimize
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

def mape(y_true, y_pred):
    mask = y_true > 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred, dates=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mp = mape(y_true, y_pred)
    
    monthly_mae = np.nan
    if dates is not None:
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'month': dates.to_period('M')})
        monthly_mae = df.groupby('month').apply(lambda x: mean_absolute_error(x['y_true'], x['y_pred'])).mean()
        
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mp, 'monthly_MAE': monthly_mae}

def create_fourier_features(dates, K=6):
    dayofyear = dates.dayofyear
    features = {}
    for k in range(1, K+1):
        features[f'sin_{k}'] = np.sin(2 * np.pi * k * dayofyear / 365.25)
        features[f'cos_{k}'] = np.cos(2 * np.pi * k * dayofyear / 365.25)
    return pd.DataFrame(features, index=dates)

def train_eval_models(df, train_end, val_start, val_end):
    train = df[df.index < val_start]
    val = df[(df.index >= val_start) & (df.index <= val_end)]
    
    dates_train = train.index
    dates_val = val.index
    
    y_train = train['Revenue'].values
    y_val = val['Revenue'].values
    
    preds = pd.DataFrame(index=dates_val)
    preds['actual'] = y_val
    
    # Model A: seasonal_naive_364
    preds['seasonal_naive_364'] = df.loc[dates_val - pd.Timedelta(days=364), 'Revenue'].values
    
    # Model B: seasonal_naive_365
    preds['seasonal_naive_365'] = df.loc[dates_val - pd.Timedelta(days=365), 'Revenue'].values
    
    # Model C: recency_weighted_dayofyear_profile
    train_doy = train.groupby([train.index.year, train.index.dayofyear])['Revenue'].mean().unstack(level=0)
    # Give higher weights to recent years
    years = train_doy.columns.values
    weights = np.exp((years - years.max()) / 2.0)
    weights /= weights.sum()
    doy_profile = train_doy.dot(weights)
    
    # Fill any missing doy with median or adjacent
    if doy_profile.isna().any():
        doy_profile = doy_profile.interpolate(limit_direction='both')
    
    # handle leap years (day 366)
    if 366 not in doy_profile.index:
        doy_profile.loc[366] = doy_profile.loc[365]
        
    preds['recency_weighted_doy'] = [doy_profile.loc[d] for d in dates_val.dayofyear]
    
    # Model D: trend_adjusted_yoy_profile
    # estimate trend from last 3 years of train
    last_year = train.index.year.max()
    rev_last = train[train.index.year == last_year]['Revenue'].sum()
    rev_prev = train[train.index.year == last_year - 1]['Revenue'].sum()
    if rev_prev > 0:
        trend = rev_last / rev_prev
    else:
        trend = 1.0
        
    # cap trend between 0.8 and 1.2 for stability
    trend = np.clip(trend, 0.8, 1.2)
    # Apply to profile
    preds['trend_adjusted_doy'] = preds['recency_weighted_doy'] * trend
    preds['trend_adjusted_doy_50'] = preds['recency_weighted_doy'] * (1.0 + 0.5 * (trend - 1.0))
    preds['trend_adjusted_doy_25'] = preds['recency_weighted_doy'] * (1.0 + 0.25 * (trend - 1.0))
    
    # Model E: Fourier Ridge calendar model
    K = 6
    X_train = create_fourier_features(dates_train, K=K)
    X_val = create_fourier_features(dates_val, K=K)
    
    X_train['dow'] = dates_train.dayofweek
    X_val['dow'] = dates_val.dayofweek
    X_train = pd.get_dummies(X_train, columns=['dow'])
    X_val = pd.get_dummies(X_val, columns=['dow'])
    
    # ensure same columns
    for c in X_train.columns:
        if c not in X_val.columns:
            X_val[c] = 0
    X_val = X_val[X_train.columns]
    
    X_train['time_idx'] = np.arange(len(dates_train))
    X_val['time_idx'] = np.arange(len(dates_train), len(dates_train) + len(dates_val))
    
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X_train, np.log1p(y_train))
    preds['fourier_ridge'] = np.expm1(ridge.predict(X_val))
    
    # Model F: direct year-over-year lag model
    # for each date t, use lag 364, 365
    def create_lag_features(dates_target):
        feat = pd.DataFrame(index=dates_target)
        for lag in [364, 365, 728, 730]:
            lag_dates = dates_target - pd.Timedelta(days=lag)
            # handle out of bounds gracefully by filling with nan then imputing
            valid_mask = lag_dates.isin(df.index)
            vals = np.full(len(dates_target), np.nan)
            vals[valid_mask] = df.loc[lag_dates[valid_mask], 'Revenue'].values
            feat[f'lag_{lag}'] = vals
            
        feat['dow'] = dates_target.dayofweek
        feat['doy'] = dates_target.dayofyear
        return feat
        
    X_train_lag = create_lag_features(dates_train)
    X_val_lag = create_lag_features(dates_val)
    
    # drop rows with NaN in train
    valid_train = X_train_lag.dropna().index
    X_train_lag_valid = X_train_lag.loc[valid_train]
    y_train_valid = train.loc[valid_train, 'Revenue']
    
    if len(X_train_lag_valid) > 0:
        rf = ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train_lag_valid, np.log1p(y_train_valid.values))
        # impute val lags if missing (should not be for 548 days horizon, but just in case)
        X_val_lag_imputed = X_val_lag.fillna(method='ffill').fillna(method='bfill')
        preds['direct_rf'] = np.expm1(rf.predict(X_val_lag_imputed))
        
        # save rf for later feature importance
        model_rf = rf
        X_val_lag_for_imp = X_val_lag_imputed
    else:
        preds['direct_rf'] = preds['seasonal_naive_364']
        model_rf = None
        X_val_lag_for_imp = None
        
    return preds, model_rf, X_val_lag_for_imp

def main():
    print("Loading data...")
    sales = pd.read_csv('data/sales.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values('Date').set_index('Date')
    
    sample_sub = pd.read_csv('sample_submission.csv')
    sample_sub['Date'] = pd.to_datetime(sample_sub['Date'])
    
    print(f"Sales data from {sales.index.min().date()} to {sales.index.max().date()}")
    print(f"Sample sub from {sample_sub['Date'].min().date()} to {sample_sub['Date'].max().date()} with {len(sample_sub)} rows")
    
    # Diagnostics of 009
    if os.path.exists('artifacts/submissions/submission_forecast_009.csv'):
        print("Diagnosing submission 009...")
        sub009 = pd.read_csv('artifacts/submissions/submission_forecast_009.csv')
        sub009['Date'] = pd.to_datetime(sub009['Date'])
        # Add basic diagnostic logs or plots if needed
    
    folds = [
        {'name': 'Fold A', 'val_start': '2019-01-01', 'val_end': '2020-07-01'},
        {'name': 'Fold B', 'val_start': '2020-01-01', 'val_end': '2021-07-01'},
        {'name': 'Fold C', 'val_start': '2021-01-01', 'val_end': '2022-07-01'}
    ]
    
    results = []
    fold_preds = {}
    rf_models = {}
    rf_val_data = {}
    
    print("Running backtest folds...")
    for f in folds:
        print(f"  {f['name']}...")
        preds, rf_mod, rf_val = train_eval_models(sales, None, f['val_start'], f['val_end'])
        fold_preds[f['name']] = preds
        rf_models[f['name']] = rf_mod
        rf_val_data[f['name']] = rf_val
        
        for col in preds.columns:
            if col == 'actual': continue
            metrics = get_metrics(preds['actual'].values, preds[col].values, preds.index)
            metrics['fold'] = f['name']
            metrics['model_name'] = col
            results.append(metrics)
            
    results_df = pd.DataFrame(results)
    
    # COGS model: use historical ratio
    # Calculate rolling annual ratio
    sales['cogs_ratio'] = sales['COGS'] / sales['Revenue']
    sales['cogs_ratio'] = sales['cogs_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # To determine ensemble weights, we use the out-of-fold predictions
    # Let's optimize non-negative weights for a hybrid ensemble
    def optimize_ensemble(preds_df):
        models = [c for c in preds_df.columns if c != 'actual']
        y = preds_df['actual'].values
        X = preds_df[models].values
        
        w, _ = optimize.nnls(X, y)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(len(models)) / len(models)
        return w, models
        
    print("Optimizing ensemble weights...")
    # Pool all folds
    all_preds = pd.concat(fold_preds.values())
    opt_w, model_cols = optimize_ensemble(all_preds)
    
    weights_df = pd.DataFrame({'model': model_cols, 'weight': opt_w})
    weights_df.to_csv('artifacts/tables/forecast_010_ensemble_weights.csv', index=False)
    print("Ensemble weights:")
    print(weights_df.sort_values('weight', ascending=False))
    
    # Add ensemble predictions to results
    for f_name, preds in fold_preds.items():
        preds['hybrid_ensemble'] = np.dot(preds[model_cols].values, opt_w)
        
        # also calculate conservative / aggressive variants
        # Conservative: more weight on seasonal naive and 0.25 trend
        # Aggressive: more weight on 1.0 trend or RF
        
        preds['conservative'] = 0.5 * preds['seasonal_naive_364'] + 0.5 * preds['trend_adjusted_doy_25']
        preds['aggressive'] = 0.3 * preds['seasonal_naive_364'] + 0.3 * preds['trend_adjusted_doy'] + 0.4 * preds['direct_rf']
        
        for ensemble_col in ['hybrid_ensemble', 'conservative', 'aggressive']:
            metrics = get_metrics(preds['actual'].values, preds[ensemble_col].values, preds.index)
            metrics['fold'] = f_name
            metrics['model_name'] = ensemble_col
            results_df = pd.concat([results_df, pd.DataFrame([metrics])])
            
    results_df = results_df.reset_index(drop=True)
    results_df.to_csv('artifacts/tables/forecast_010_backtest_metrics.csv', index=False)
    
    # Model comparison
    avg_metrics = results_df.groupby('model_name')[['MAE', 'RMSE', 'R2']].mean().reset_index()
    avg_metrics['selected_flag'] = (avg_metrics['model_name'] == 'hybrid_ensemble').astype(int)
    avg_metrics['target'] = 'Revenue'
    avg_metrics.to_csv('artifacts/tables/forecast_010_model_comparison.csv', index=False)
    
    # Error analysis for base ensemble
    print("Performing error analysis...")
    all_preds_updated = pd.concat(fold_preds.values())
    all_preds_updated['error'] = all_preds_updated['hybrid_ensemble'] - all_preds_updated['actual']
    all_preds_updated['abs_error'] = all_preds_updated['error'].abs()
    all_preds_updated['month'] = all_preds_updated.index.month
    all_preds_updated['dayofweek'] = all_preds_updated.index.dayofweek
    
    error_analysis = all_preds_updated.groupby('month')[['error', 'abs_error']].mean().reset_index()
    error_analysis.to_csv('artifacts/tables/forecast_010_error_analysis.csv', index=False)
    
    # Feature importance
    # We take the RF from fold C
    if rf_models['Fold C'] is not None:
        rf = rf_models['Fold C']
        X_val = rf_val_data['Fold C']
        # permutation importance
        y_val_actual = fold_preds['Fold C'].loc[X_val.index, 'actual'].values
        r = permutation_importance(rf, X_val, np.log1p(y_val_actual),
                                   n_repeats=10, random_state=42)
        fi = pd.DataFrame({'feature': X_val.columns, 'importance': r.importances_mean, 'std': r.importances_std})
        fi = fi.sort_values('importance', ascending=False)
        fi.to_csv('artifacts/tables/forecast_010_feature_importance.csv', index=False)
        
        plt.figure(figsize=(10, 6))
        plt.bar(fi['feature'], fi['importance'], yerr=fi['std'])
        plt.title("Feature Importance (Permutation on Fold C)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('artifacts/figures/forecast_010_feature_importance.png')
        plt.close()
    
    # Future forecasts
    print("Generating future forecasts...")
    val_start = '2023-01-01'
    val_end = '2024-07-01'
    future_dates = pd.DatetimeIndex(sample_sub['Date'])
    
    # we simulate train_eval_models by passing sales up to 2022-12-31, and predicting for future_dates
    # we need a modified version since target values are unknown
    
    # Train full
    train = sales[sales.index <= '2022-12-31']
    
    future_preds = pd.DataFrame(index=future_dates)
    
    # Model A & B
    future_preds['seasonal_naive_364'] = [train.loc[d - pd.Timedelta(days=364), 'Revenue'] if (d - pd.Timedelta(days=364)) in train.index else np.nan for d in future_dates]
    future_preds['seasonal_naive_365'] = [train.loc[d - pd.Timedelta(days=365), 'Revenue'] if (d - pd.Timedelta(days=365)) in train.index else np.nan for d in future_dates]
    
    # For dates where 364/365 lag goes into 2023, we need lag 728/730
    future_preds['seasonal_naive_364'] = future_preds['seasonal_naive_364'].fillna(
        pd.Series([train.loc[d - pd.Timedelta(days=728), 'Revenue'] if (d - pd.Timedelta(days=728)) in train.index else np.nan for d in future_dates], index=future_dates)
    )
    future_preds['seasonal_naive_365'] = future_preds['seasonal_naive_365'].fillna(
        pd.Series([train.loc[d - pd.Timedelta(days=730), 'Revenue'] if (d - pd.Timedelta(days=730)) in train.index else np.nan for d in future_dates], index=future_dates)
    )
    
    # Fill remaining NaNs using recency_weighted_doy
    train_doy = train.groupby([train.index.year, train.index.dayofyear])['Revenue'].mean().unstack(level=0)
    years = train_doy.columns.values
    weights = np.exp((years - years.max()) / 2.0)
    weights /= weights.sum()
    doy_profile = train_doy.dot(weights)
    if doy_profile.isna().any(): doy_profile = doy_profile.interpolate(limit_direction='both')
    if 366 not in doy_profile.index: doy_profile.loc[366] = doy_profile.loc[365]
    
    future_preds['recency_weighted_doy'] = [doy_profile.loc[d] for d in future_dates.dayofyear]
    
    future_preds['seasonal_naive_364'] = future_preds['seasonal_naive_364'].fillna(future_preds['recency_weighted_doy'])
    future_preds['seasonal_naive_365'] = future_preds['seasonal_naive_365'].fillna(future_preds['recency_weighted_doy'])
    
    last_year = train.index.year.max()
    rev_last = train[train.index.year == last_year]['Revenue'].sum()
    rev_prev = train[train.index.year == last_year - 1]['Revenue'].sum()
    trend = np.clip(rev_last / rev_prev if rev_prev > 0 else 1.0, 0.8, 1.2)
    
    future_preds['trend_adjusted_doy'] = future_preds['recency_weighted_doy'] * trend
    future_preds['trend_adjusted_doy_50'] = future_preds['recency_weighted_doy'] * (1.0 + 0.5 * (trend - 1.0))
    future_preds['trend_adjusted_doy_25'] = future_preds['recency_weighted_doy'] * (1.0 + 0.25 * (trend - 1.0))
    
    # Fourier
    K=6
    X_train_f = create_fourier_features(train.index, K=K)
    X_val_f = create_fourier_features(future_dates, K=K)
    X_train_f['dow'] = train.index.dayofweek
    X_val_f['dow'] = future_dates.dayofweek
    X_train_f = pd.get_dummies(X_train_f, columns=['dow'])
    X_val_f = pd.get_dummies(X_val_f, columns=['dow'])
    for c in X_train_f.columns:
        if c not in X_val_f.columns: X_val_f[c] = 0
    X_val_f = X_val_f[X_train_f.columns]
    X_train_f['time_idx'] = np.arange(len(train.index))
    X_val_f['time_idx'] = np.arange(len(train.index), len(train.index) + len(future_dates))
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X_train_f, np.log1p(train['Revenue'].values))
    future_preds['fourier_ridge'] = np.expm1(ridge.predict(X_val_f))
    
    # Direct RF
    def create_lag_features(dates_target):
        feat = pd.DataFrame(index=dates_target)
        for lag in [364, 365, 728, 730]:
            lag_dates = dates_target - pd.Timedelta(days=lag)
            valid_mask = lag_dates.isin(train.index)
            vals = np.full(len(dates_target), np.nan)
            vals[valid_mask] = train.loc[lag_dates[valid_mask], 'Revenue'].values
            feat[f'lag_{lag}'] = vals
        feat['dow'] = dates_target.dayofweek
        feat['doy'] = dates_target.dayofyear
        return feat
    
    X_train_lag = create_lag_features(train.index)
    X_val_lag = create_lag_features(future_dates)
    X_train_lag.index = train.index
    X_val_lag.index = future_dates
    
    valid_train = X_train_lag.dropna().index
    X_train_lag_valid = X_train_lag.loc[valid_train]
    y_train_valid = train.loc[valid_train, 'Revenue']
    
    if len(X_train_lag_valid) > 0:
        rf = ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train_lag_valid, np.log1p(y_train_valid.values))
        X_val_lag_imputed = X_val_lag.fillna(method='ffill').fillna(method='bfill')
        # if still any na, fill with doy profile
        for c in X_val_lag_imputed.columns:
            if X_val_lag_imputed[c].isna().any():
                X_val_lag_imputed[c] = future_preds['recency_weighted_doy'].values
        future_preds['direct_rf'] = np.expm1(rf.predict(X_val_lag_imputed))
    else:
        future_preds['direct_rf'] = future_preds['seasonal_naive_364']
        
    # Build ensembles
    future_preds['hybrid_ensemble'] = np.dot(future_preds[model_cols].values, opt_w)
    future_preds['conservative'] = 0.5 * future_preds['seasonal_naive_364'] + 0.5 * future_preds['trend_adjusted_doy_25']
    future_preds['aggressive'] = 0.3 * future_preds['seasonal_naive_364'] + 0.3 * future_preds['trend_adjusted_doy'] + 0.4 * future_preds['direct_rf']
    
    # Calculate COGS
    recent_ratio = train['cogs_ratio'].tail(365).mean()
    print(f"Recent COGS/Revenue ratio: {recent_ratio:.4f}")
    
    # Save outputs
    output_df = pd.DataFrame({
        'Date': future_dates.values,
        'Revenue_base': future_preds['hybrid_ensemble'].values,
        'COGS_base': future_preds['hybrid_ensemble'].values * recent_ratio,
        'Revenue_conservative': future_preds['conservative'].values,
        'COGS_conservative': future_preds['conservative'].values * recent_ratio,
        'Revenue_aggressive': future_preds['aggressive'].values,
        'COGS_aggressive': future_preds['aggressive'].values * recent_ratio
    })
    output_df.to_csv('artifacts/forecasts/forecast_010_future_predictions.csv', index=False)
    
    # Submissions
    sub_base = output_df[['Date', 'Revenue_base', 'COGS_base']].rename(columns={'Revenue_base': 'Revenue', 'COGS_base': 'COGS'})
    sub_base.to_csv('artifacts/submissions/submission_forecast_010.csv', index=False)
    
    sub_cons = output_df[['Date', 'Revenue_conservative', 'COGS_conservative']].rename(columns={'Revenue_conservative': 'Revenue', 'COGS_conservative': 'COGS'})
    sub_cons.to_csv('artifacts/submissions/submission_forecast_010_conservative.csv', index=False)
    
    sub_agg = output_df[['Date', 'Revenue_aggressive', 'COGS_aggressive']].rename(columns={'Revenue_aggressive': 'Revenue', 'COGS_aggressive': 'COGS'})
    sub_agg.to_csv('artifacts/submissions/submission_forecast_010_aggressive.csv', index=False)
    
    # Plots
    # Backtest Actual vs Pred
    plt.figure(figsize=(15, 6))
    plt.plot(all_preds_updated.index, all_preds_updated['actual'], label='Actual', alpha=0.6)
    plt.plot(all_preds_updated.index, all_preds_updated['hybrid_ensemble'], label='Ensemble', alpha=0.8)
    plt.title("Backtest: Actual vs Ensemble")
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_010_backtest_actual_vs_pred.png')
    plt.close()
    
    # Future Forecast
    plt.figure(figsize=(15, 6))
    plt.plot(train.index[-365*2:], train['Revenue'].tail(365*2), label='Historical (last 2y)')
    plt.plot(future_dates, output_df['Revenue_base'], label='Forecast Base')
    plt.plot(future_dates, output_df['Revenue_conservative'], label='Forecast Conservative', alpha=0.5)
    plt.plot(future_dates, output_df['Revenue_aggressive'], label='Forecast Aggressive', alpha=0.5)
    plt.title("Future Forecast Scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_010_future_forecast.png')
    plt.close()
    
    # Model Comparison Chart
    avg_metrics.set_index('model_name')['MAE'].sort_values().plot(kind='bar', figsize=(10, 6))
    plt.title("Model Comparison - MAE")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_010_model_comparison.png')
    plt.close()
    
    # Residuals by month
    plt.figure(figsize=(10, 6))
    all_preds_updated.boxplot(column='error', by='month')
    plt.title("Residuals by Month (Base Ensemble)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_010_residuals_by_month.png')
    plt.close()
    
    print("Done.")

if __name__ == '__main__':
    main()
