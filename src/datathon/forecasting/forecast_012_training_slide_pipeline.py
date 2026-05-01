import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
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
    q_mae = {'Q1': np.nan, 'Q2': np.nan, 'Q3': np.nan, 'Q4': np.nan}
    if dates is not None:
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'month': dates.to_period('M'), 'quarter': dates.quarter})
        monthly_mae = df.groupby('month').apply(lambda x: mean_absolute_error(x['y_true'], x['y_pred'])).mean()
        for q in [1, 2, 3, 4]:
            q_df = df[df['quarter'] == q]
            if len(q_df) > 0:
                q_mae[f'Q{q}_MAE'] = mean_absolute_error(q_df['y_true'], q_df['y_pred'])
        
    res = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mp, 'monthly_MAE': monthly_mae}
    for q in [1, 2, 3, 4]:
        res[f'Q{q}_MAE'] = q_mae.get(f'Q{q}_MAE', np.nan)
    return res

def create_date_features(df):
    dates = df.index
    feat = pd.DataFrame(index=dates)
    feat['year'] = dates.year
    feat['month'] = dates.month
    feat['day'] = dates.day
    feat['dayofweek'] = dates.dayofweek
    feat['dayofyear'] = dates.dayofyear
    feat['weekofyear'] = dates.isocalendar().week.astype(int)
    feat['quarter'] = dates.quarter
    feat['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    feat['time_index'] = np.arange(len(dates))
    feat['t_years'] = feat['time_index'] / 365.25
    
    # Edge of month
    feat['is_first1'] = (dates.day == 1).astype(int)
    feat['is_first2'] = (dates.day == 2).astype(int)
    feat['is_first3'] = (dates.day <= 3).astype(int)
    feat['days_to_eom'] = dates.days_in_month - dates.day
    feat['is_last1'] = (feat['days_to_eom'] == 0).astype(int)
    feat['is_last2'] = (feat['days_to_eom'] == 1).astype(int)
    feat['is_last3'] = (feat['days_to_eom'] <= 2).astype(int)
    
    feat['dom_bucket_first3'] = (dates.day <= 3).astype(int)
    feat['dom_bucket_last3'] = (feat['days_to_eom'] <= 2).astype(int)
    feat['dom_bucket_last7'] = (feat['days_to_eom'] <= 6).astype(int)
    feat['dom_bucket_mid'] = ((dates.day > 3) & (feat['days_to_eom'] > 6)).astype(int)
    
    # Fourier
    for k in range(1, 7):
        feat[f'yearly_sin_{k}'] = np.sin(2 * np.pi * k * dates.dayofyear / 365.25)
        feat[f'yearly_cos_{k}'] = np.cos(2 * np.pi * k * dates.dayofyear / 365.25)
    for k in range(1, 4):
        feat[f'weekly_sin_{k}'] = np.sin(2 * np.pi * k * dates.dayofweek / 7.0)
        feat[f'weekly_cos_{k}'] = np.cos(2 * np.pi * k * dates.dayofweek / 7.0)
    for k in range(1, 4):
        feat[f'monthly_sin_{k}'] = np.sin(2 * np.pi * k * dates.day / dates.days_in_month)
        feat[f'monthly_cos_{k}'] = np.cos(2 * np.pi * k * dates.day / dates.days_in_month)
        
    # Odd/even
    feat['is_odd_year'] = (dates.year % 2 != 0).astype(int)
    feat['is_even_year'] = (dates.year % 2 == 0).astype(int)
    feat['is_august'] = (dates.month == 8).astype(int)
    feat['is_august_even_year'] = feat['is_august'] * feat['is_even_year']
    feat['is_august_odd_year'] = feat['is_august'] * feat['is_odd_year']
    
    for m in range(1, 13):
        feat[f'month_{m}_odd'] = ((dates.month == m) & feat['is_odd_year']).astype(int)
        feat[f'month_{m}_even'] = ((dates.month == m) & feat['is_even_year']).astype(int)
        
    # Quarter
    for q in range(1, 5):
        feat[f'q{q}'] = (dates.quarter == q).astype(int)
    feat['q3_odd_year'] = feat['q3'] * feat['is_odd_year']
    feat['q3_even_year'] = feat['q3'] * feat['is_even_year']
    
    return feat

def detect_tet_proxy(df):
    tet_dates = {}
    for year in df.index.year.unique():
        year_data = df[df.index.year == year]
        jan_feb = year_data[((year_data.index.month == 1) & (year_data.index.day >= 10)) | ((year_data.index.month == 2) & (year_data.index.day <= 25))]
        if len(jan_feb) > 0:
            rolling_7 = jan_feb['Revenue'].rolling(7, center=True).mean()
            if not rolling_7.isna().all():
                min_idx = rolling_7.idxmin()
                tet_dates[year] = min_idx
    return tet_dates

def add_tet_features(feat, tet_dates):
    historical_doy = [d.dayofyear for d in tet_dates.values()]
    median_doy = int(np.median(historical_doy)) if historical_doy else 40
    
    all_years = feat.index.year.unique()
    tet_dates_full = {}
    for y in all_years:
        if y in tet_dates:
            tet_dates_full[y] = tet_dates[y]
        else:
            tet_dates_full[y] = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=median_doy - 1)
            
    feat['tet_proxy_date'] = feat.index.year.map(tet_dates_full)
    feat['tet_days_diff'] = (feat.index - feat['tet_proxy_date']).dt.days
    feat['abs_tet_days_diff'] = feat['tet_days_diff'].abs()
    feat['tet_in_3'] = (feat['abs_tet_days_diff'] <= 3).astype(int)
    feat['tet_in_7'] = (feat['abs_tet_days_diff'] <= 7).astype(int)
    feat['tet_in_14'] = (feat['abs_tet_days_diff'] <= 14).astype(int)
    feat['post_tet_14'] = ((feat['tet_days_diff'] > 0) & (feat['tet_days_diff'] <= 14)).astype(int)
    
    feat = feat.drop(columns=['tet_proxy_date'])
    return feat

def add_regime_weights(dates):
    w = np.ones(len(dates))
    w[(dates >= '2012-07-04') & (dates <= '2013-12-31')] = 0.25
    w[(dates >= '2014-01-01') & (dates <= '2018-12-31')] = 0.75
    w[(dates >= '2019-01-01') & (dates <= '2019-12-31')] = 0.80
    w[(dates >= '2020-01-01') & (dates <= '2022-12-31')] = 1.50
    return w

def train_eval_models(df_train, df_val, feat_train, feat_val):
    dates_train = df_train.index
    dates_val = df_val.index
    
    y_train = np.log1p(df_train['Revenue'].values)
    y_val = df_val['Revenue'].values
    w_train = add_regime_weights(dates_train)
    
    preds = pd.DataFrame(index=dates_val)
    preds['actual'] = y_val
    
    models_dict = {}
    
    # M1: Ridge Fourier Calendar
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(feat_train, y_train, sample_weight=w_train)
    preds['M1_Ridge'] = np.expm1(ridge.predict(feat_val))
    models_dict['M1_Ridge'] = ridge
    
    # M2: ExtraTrees Calendar
    rf = ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=12, min_samples_leaf=5)
    rf.fit(feat_train, y_train, sample_weight=w_train)
    preds['M2_ExtraTrees'] = np.expm1(rf.predict(feat_val))
    models_dict['M2_ExtraTrees'] = rf
    
    # M3: HistGradientBoosting
    hgb = HistGradientBoostingRegressor(random_state=42, max_depth=10, min_samples_leaf=5)
    hgb.fit(feat_train, y_train, sample_weight=w_train)
    preds['M3_HGB'] = np.expm1(hgb.predict(feat_val))
    models_dict['M3_HGB'] = hgb
    
    # M4: Seasonal profile with regime calibration
    shape_train = df_train[(df_train.index >= '2014-01-01') & (df_train.index <= '2018-12-31')]
    shape_doy = shape_train.groupby(shape_train.index.dayofyear)['Revenue'].mean()
    shape_doy = shape_doy / shape_doy.mean()
    
    level_train = df_train[(df_train.index >= '2020-01-01') & (df_train.index <= '2022-12-31')]
    level_mean = level_train['Revenue'].mean() if len(level_train) > 0 else df_train['Revenue'].mean()
    
    pred_m4 = []
    for d in dates_val:
        doy = d.dayofyear
        val_shape = shape_doy.get(doy, 1.0)
        if pd.isna(val_shape): val_shape = 1.0
        if d.day <= 3: val_shape *= 1.1
        if d.days_in_month - d.day <= 2: val_shape *= 1.05
        pred_m4.append(val_shape * level_mean)
    preds['M4_SeasonalProfile'] = pred_m4
    models_dict['M4_SeasonalProfile'] = None
    
    # M5: Q-specialist models
    preds['M5_QSpecialist'] = 0.0
    for q in [1, 2, 3, 4]:
        w_q = w_train.copy()
        w_q[dates_train.quarter == q] *= 2.0
        ridge_q = RidgeCV(alphas=[0.1, 1.0, 10.0])
        ridge_q.fit(feat_train, y_train, sample_weight=w_q)
        
        q_mask = dates_val.quarter == q
        if q_mask.any():
            preds.loc[q_mask, 'M5_QSpecialist'] = np.expm1(ridge_q.predict(feat_val[q_mask]))
            
    preds['M5_QSpecialist_Blend'] = 0.5 * preds['M1_Ridge'] + 0.5 * preds['M5_QSpecialist']
    models_dict['M5_QSpecialist'] = None
    
    # Simple COGS model: Ratio
    cogs_ratio = df_train['COGS'] / df_train['Revenue']
    cogs_ratio = cogs_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    cogs_ratio_recent = cogs_ratio.tail(365).mean()
    models_dict['cogs_ratio'] = cogs_ratio_recent
    
    return preds, models_dict

def main():
    print("Loading data...")
    sales = pd.read_csv('data/sales.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values('Date').set_index('Date')
    
    sample_sub = pd.read_csv('sample_submission.csv')
    sample_sub['Date'] = pd.to_datetime(sample_sub['Date'])
    
    print("Creating features...")
    tet_dates = detect_tet_proxy(sales)
    sales_feat = create_date_features(sales)
    sales_feat = add_tet_features(sales_feat, tet_dates)
    
    future_dates = pd.DatetimeIndex(sample_sub['Date'])
    future_feat = create_date_features(pd.DataFrame(index=future_dates))
    future_feat = add_tet_features(future_feat, tet_dates)
    
    folds = [
        {'name': 'Fold 1', 'train_end': '2018-12-31', 'val_start': '2019-01-01', 'val_end': '2020-07-01'},
        {'name': 'Fold 2', 'train_end': '2019-12-31', 'val_start': '2020-01-01', 'val_end': '2021-07-01'},
        {'name': 'Fold 3', 'train_end': '2020-12-31', 'val_start': '2021-01-01', 'val_end': '2022-07-01'},
        {'name': 'Fold 4', 'train_end': '2021-07-01', 'val_start': '2021-07-02', 'val_end': '2022-12-31'}
    ]
    
    results = []
    fold_preds = {}
    models_saved = {}
    
    print("Running backtest folds...")
    for f in folds:
        print(f"  {f['name']}...")
        df_train = sales[sales.index < f['val_start']]
        df_val = sales[(sales.index >= f['val_start']) & (sales.index <= f['val_end'])]
        feat_train = sales_feat.loc[df_train.index]
        feat_val = sales_feat.loc[df_val.index]
        
        preds, models_dict = train_eval_models(df_train, df_val, feat_train, feat_val)
        fold_preds[f['name']] = preds
        models_saved[f['name']] = models_dict
        
        for col in preds.columns:
            if col == 'actual': continue
            metrics = get_metrics(preds['actual'].values, preds[col].values, preds.index)
            metrics['fold'] = f['name']
            metrics['model_name'] = col
            results.append(metrics)
            
    results_df = pd.DataFrame(results)
    
    def optimize_ensemble(preds_df):
        models = [c for c in preds_df.columns if c != 'actual' and 'Ensemble' not in c and 'calibrated' not in c]
        y = preds_df['actual'].values
        X = preds_df[models].values
        
        w, _ = optimize.nnls(X, y)
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones(len(models)) / len(models)
        return w, models
        
    print("Optimizing ensemble weights...")
    all_preds = pd.concat(fold_preds.values())
    opt_w, model_cols = optimize_ensemble(all_preds)
    
    weights_df = pd.DataFrame({'model': model_cols, 'weight': opt_w})
    weights_df.to_csv('artifacts/tables/forecast_012_ensemble_weights.csv', index=False)
    
    # Calculate calibration
    # simply predict global ratio of sum(actual) / sum(pred)
    base_pred = np.dot(all_preds[model_cols].values, opt_w)
    calibration_factor = all_preds['actual'].sum() / base_pred.sum()
    print(f"Calibration factor: {calibration_factor:.4f}")
    
    # Add ensemble predictions to results
    for f_name, preds in fold_preds.items():
        preds['M6_Ensemble'] = np.dot(preds[model_cols].values, opt_w)
        preds['M6_Ensemble_calibrated'] = preds['M6_Ensemble'] * calibration_factor
        
        for ensemble_col in ['M6_Ensemble', 'M6_Ensemble_calibrated']:
            metrics = get_metrics(preds['actual'].values, preds[ensemble_col].values, preds.index)
            metrics['fold'] = f_name
            metrics['model_name'] = ensemble_col
            results_df = pd.concat([results_df, pd.DataFrame([metrics])])
            
    results_df = results_df.reset_index(drop=True)
    results_df.to_csv('artifacts/tables/forecast_012_backtest_metrics.csv', index=False)
    
    avg_metrics = results_df.groupby('model_name')[['MAE', 'RMSE', 'R2', 'MAPE']].mean().reset_index()
    avg_metrics['selected_flag'] = (avg_metrics['model_name'] == 'M6_Ensemble').astype(int)
    avg_metrics.to_csv('artifacts/tables/forecast_012_model_comparison.csv', index=False)
    
    # Error analysis
    all_preds_updated = pd.concat(fold_preds.values())
    all_preds_updated['error'] = all_preds_updated['M6_Ensemble'] - all_preds_updated['actual']
    all_preds_updated['abs_error'] = all_preds_updated['error'].abs()
    all_preds_updated['quarter'] = all_preds_updated.index.quarter
    
    error_analysis = all_preds_updated.groupby('quarter')[['error', 'abs_error']].mean().reset_index()
    error_analysis.to_csv('artifacts/tables/forecast_012_error_analysis.csv', index=False)
    
    # Feature importance
    # Use Ridge model from Fold 4
    if 'M1_Ridge' in models_saved['Fold 4']:
        ridge_mod = models_saved['Fold 4']['M1_Ridge']
        fi = pd.DataFrame({'feature': sales_feat.columns, 'importance': np.abs(ridge_mod.coef_)})
        fi = fi.sort_values('importance', ascending=False)
        fi.to_csv('artifacts/tables/forecast_012_feature_importance.csv', index=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(fi['feature'].head(20), fi['importance'].head(20))
        plt.title("Feature Importance (Absolute Ridge Coefficients)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('artifacts/figures/forecast_012_feature_importance.png')
        plt.close()
    
    # Future forecasts
    print("Generating future forecasts...")
    future_preds, future_models = train_eval_models(sales, pd.DataFrame(index=future_dates, columns=['Revenue']).fillna(0), sales_feat, future_feat)
    
    future_preds['base'] = np.dot(future_preds[model_cols].values, opt_w)
    future_preds['qspecialist'] = future_preds['M5_QSpecialist_Blend']
    future_preds['calibrated'] = future_preds['base'] * calibration_factor
    future_preds['lowrisk'] = 0.5 * future_preds['M1_Ridge'] + 0.5 * future_preds['M4_SeasonalProfile']
    
    ratio = future_models['cogs_ratio']
    
    output_df = pd.DataFrame({
        'Date': future_dates.values,
        'Revenue_base': future_preds['base'].values,
        'COGS_base': future_preds['base'].values * ratio,
        'Revenue_qspecialist': future_preds['qspecialist'].values,
        'COGS_qspecialist': future_preds['qspecialist'].values * ratio,
        'Revenue_calibrated': future_preds['calibrated'].values,
        'COGS_calibrated': future_preds['calibrated'].values * ratio,
        'Revenue_lowrisk': future_preds['lowrisk'].values,
        'COGS_lowrisk': future_preds['lowrisk'].values * ratio
    })
    output_df.to_csv('artifacts/forecasts/forecast_012_future_predictions.csv', index=False)
    
    # Submissions
    def save_sub(variant, filename):
        sub = output_df[['Date', f'Revenue_{variant}', f'COGS_{variant}']].rename(columns={f'Revenue_{variant}': 'Revenue', f'COGS_{variant}': 'COGS'})
        sub.to_csv(filename, index=False)
        
    save_sub('base', 'artifacts/submissions/submission_forecast_012_base.csv')
    save_sub('qspecialist', 'artifacts/submissions/submission_forecast_012_qspecialist.csv')
    save_sub('calibrated', 'artifacts/submissions/submission_forecast_012_calibrated.csv')
    save_sub('lowrisk', 'artifacts/submissions/submission_forecast_012_lowrisk.csv')
    
    # Candidate manifest
    manifest_data = [
        {'candidate_file': 'submission_forecast_012_base.csv', 'candidate_name': 'base', 'method': 'M6 Ensemble', 'hypothesis': 'Optimal CV blend'},
        {'candidate_file': 'submission_forecast_012_qspecialist.csv', 'candidate_name': 'qspecialist', 'method': 'M5 Q-Specialist Blend', 'hypothesis': 'Quarterly targeted weights'},
        {'candidate_file': 'submission_forecast_012_calibrated.csv', 'candidate_name': 'calibrated', 'method': 'M6 Ensemble * cal_factor', 'hypothesis': 'Fix level bias'},
        {'candidate_file': 'submission_forecast_012_lowrisk.csv', 'candidate_name': 'lowrisk', 'method': 'Ridge + Profile', 'hypothesis': 'High stability, lower variance'}
    ]
    pd.DataFrame(manifest_data).to_csv('artifacts/tables/forecast_012_candidate_manifest.csv', index=False)
    
    # Plots
    # Backtest Actual vs Pred
    plt.figure(figsize=(15, 6))
    plt.plot(all_preds_updated.index, all_preds_updated['actual'], label='Actual', alpha=0.6)
    plt.plot(all_preds_updated.index, all_preds_updated['M6_Ensemble'], label='Ensemble', alpha=0.8)
    plt.title("Backtest: Actual vs Ensemble")
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_012_backtest_actual_vs_pred.png')
    plt.close()
    
    # Future Forecast
    plt.figure(figsize=(15, 6))
    plt.plot(sales.index[-365*2:], sales['Revenue'].tail(365*2), label='Historical (last 2y)')
    plt.plot(future_dates, output_df['Revenue_base'], label='Forecast Base')
    plt.plot(future_dates, output_df['Revenue_qspecialist'], label='Forecast Q-Specialist', alpha=0.5)
    plt.plot(future_dates, output_df['Revenue_calibrated'], label='Forecast Calibrated', alpha=0.5)
    plt.plot(future_dates, output_df['Revenue_lowrisk'], label='Forecast Low Risk', alpha=0.5)
    plt.title("Future Forecast Scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_012_future_forecast.png')
    plt.close()
    
    # Model Comparison Chart
    avg_metrics.set_index('model_name')['MAE'].sort_values().plot(kind='bar', figsize=(10, 6))
    plt.title("Model Comparison - MAE")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_012_model_comparison.png')
    plt.close()
    
    # Residuals by quarter
    plt.figure(figsize=(10, 6))
    all_preds_updated.boxplot(column='error', by='quarter')
    plt.title("Residuals by Quarter (Base Ensemble)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_012_residuals_by_quarter.png')
    plt.close()
    
    # Calendar effects
    shape_train = sales[(sales.index >= '2014-01-01') & (sales.index <= '2018-12-31')]
    shape_doy = shape_train.groupby(shape_train.index.dayofyear)['Revenue'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(shape_doy.index, shape_doy.values)
    plt.title("Day of Year Shape (2014-2018)")
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_012_calendar_effects.png')
    plt.close()
    
    print("Done.")

if __name__ == '__main__':
    main()
