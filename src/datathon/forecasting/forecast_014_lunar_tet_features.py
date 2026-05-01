import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.optimize as optimize
import warnings

warnings.filterwarnings('ignore')

def get_tet_dates():
    tet_dates = {
        2010: '2010-02-14',
        2011: '2011-02-03',
        2012: '2012-01-23',
        2013: '2013-02-10',
        2014: '2014-01-31',
        2015: '2015-02-19',
        2016: '2016-02-08',
        2017: '2017-01-28',
        2018: '2018-02-16',
        2019: '2019-02-05',
        2020: '2020-01-25',
        2021: '2021-02-12',
        2022: '2022-02-01',
        2023: '2023-01-22',
        2024: '2024-02-10',
        2025: '2025-01-29'
    }
    return {k: pd.to_datetime(v) for k, v in tet_dates.items()}

def mape(y_true, y_pred):
    mask = y_true > 0
    if not mask.any(): return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred, dates, feat_df=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mp = mape(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'month': dates.to_period('M'), 'quarter': dates.quarter})
    monthly_mae = df.groupby('month').apply(lambda x: mean_absolute_error(x['y_true'], x['y_pred'])).mean()
    
    q_mae = {}
    for q in [1, 2, 3, 4]:
        q_df = df[df['quarter'] == q]
        if len(q_df) > 0:
            q_mae[f'Q{q}_MAE'] = mean_absolute_error(q_df['y_true'], q_df['y_pred'])
            
    tet_window_mae = np.nan
    if feat_df is not None:
        mask = feat_df['abs_tet_days_diff'] <= 30
        if mask.any():
            tet_window_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            
    res = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mp, 'bias': bias, 'monthly_MAE': monthly_mae, 'tet_window_MAE': tet_window_mae}
    for q in [1, 2, 3, 4]:
        res[f'Q{q}_MAE'] = q_mae.get(f'Q{q}_MAE', np.nan)
    return res

def create_date_features(df, start_index=0):
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
    feat['time_index'] = np.arange(start_index, start_index + len(dates))
    feat['t_years'] = feat['time_index'] / 365.25
    
    feat['is_first1'] = (dates.day == 1).astype(int)
    feat['is_first2'] = (dates.day == 2).astype(int)
    feat['is_first3'] = (dates.day <= 3).astype(int)
    feat['days_to_eom'] = dates.days_in_month - dates.day
    feat['is_last1'] = (feat['days_to_eom'] == 0).astype(int)
    feat['is_last2'] = (feat['days_to_eom'] == 1).astype(int)
    feat['is_last3'] = (feat['days_to_eom'] <= 2).astype(int)
    
    for k in range(1, 7):
        feat[f'yearly_sin_{k}'] = np.sin(2 * np.pi * k * dates.dayofyear / 365.25)
        feat[f'yearly_cos_{k}'] = np.cos(2 * np.pi * k * dates.dayofyear / 365.25)
    for k in range(1, 4):
        feat[f'weekly_sin_{k}'] = np.sin(2 * np.pi * k * dates.dayofweek / 7.0)
        feat[f'weekly_cos_{k}'] = np.cos(2 * np.pi * k * dates.dayofweek / 7.0)
    for k in range(1, 4):
        feat[f'monthly_sin_{k}'] = np.sin(2 * np.pi * k * dates.day / dates.days_in_month)
        feat[f'monthly_cos_{k}'] = np.cos(2 * np.pi * k * dates.day / dates.days_in_month)
        
    feat['is_odd_year'] = (dates.year % 2 != 0).astype(int)
    feat['is_even_year'] = (dates.year % 2 == 0).astype(int)
    feat['is_august'] = (dates.month == 8).astype(int)
    feat['is_august_even_year'] = feat['is_august'] * feat['is_even_year']
    feat['is_august_odd_year'] = feat['is_august'] * feat['is_odd_year']
    
    for m in range(1, 13):
        feat[f'month_{m}_odd'] = ((dates.month == m) & feat['is_odd_year']).astype(int)
        feat[f'month_{m}_even'] = ((dates.month == m) & feat['is_even_year']).astype(int)
        
    for q in range(1, 5):
        feat[f'q{q}'] = (dates.quarter == q).astype(int)
    feat['q3_odd_year'] = feat['q3'] * feat['is_odd_year']
    feat['q3_even_year'] = feat['q3'] * feat['is_even_year']
    
    return feat

def add_tet_features(feat):
    tet_dates = get_tet_dates()
    
    diff_this_year = (feat.index - feat.index.year.map(tet_dates)).days
    diff_next_year = (feat.index - (feat.index.year + 1).map(tet_dates)).days
    diff_prev_year = (feat.index - (feat.index.year - 1).map(tet_dates)).days
    
    d1 = diff_this_year
    d2 = diff_next_year
    d3 = diff_prev_year
    
    d1_abs = np.abs(d1)
    d2_abs = np.abs(d2)
    d3_abs = np.abs(d3)
    
    diff = pd.Series(d1, index=feat.index)
    diff[d2_abs < np.abs(diff)] = d2[d2_abs < np.abs(diff)]
    diff[d3_abs < np.abs(diff)] = d3[d3_abs < np.abs(diff)]
    
    feat['tet_days_diff'] = diff
    feat['abs_tet_days_diff'] = diff.abs()
    
    feat['is_lunar_new_year_day'] = (feat['tet_days_diff'] == 0).astype(int)
    feat['tet_in_3'] = (feat['abs_tet_days_diff'] <= 3).astype(int)
    feat['tet_in_7'] = (feat['abs_tet_days_diff'] <= 7).astype(int)
    feat['tet_in_14'] = (feat['abs_tet_days_diff'] <= 14).astype(int)
    
    feat['pre_tet_14'] = ((feat['tet_days_diff'] >= -14) & (feat['tet_days_diff'] < 0)).astype(int)
    feat['post_tet_14'] = ((feat['tet_days_diff'] > 0) & (feat['tet_days_diff'] <= 14)).astype(int)
    feat['post_tet_30'] = ((feat['tet_days_diff'] > 0) & (feat['tet_days_diff'] <= 30)).astype(int)
    
    feat['tet_window_bucket'] = 'normal'
    feat.loc[(feat['tet_days_diff'] >= -30) & (feat['tet_days_diff'] <= -15), 'tet_window_bucket'] = 'pre_30_to_15'
    feat.loc[(feat['tet_days_diff'] >= -14) & (feat['tet_days_diff'] <= -8), 'tet_window_bucket'] = 'pre_14_to_8'
    feat.loc[(feat['tet_days_diff'] >= -7) & (feat['tet_days_diff'] <= -1), 'tet_window_bucket'] = 'pre_7_to_1'
    feat.loc[(feat['tet_days_diff'] >= 0) & (feat['tet_days_diff'] <= 3), 'tet_window_bucket'] = 'tet_0_to_3'
    feat.loc[(feat['tet_days_diff'] >= 4) & (feat['tet_days_diff'] <= 14), 'tet_window_bucket'] = 'post_4_to_14'
    feat.loc[(feat['tet_days_diff'] >= 15) & (feat['tet_days_diff'] <= 30), 'tet_window_bucket'] = 'post_15_to_30'
    
    dummies = pd.get_dummies(feat['tet_window_bucket'], prefix='bucket')
    for col in dummies.columns:
        feat[col] = dummies[col]
        
    feat = feat.drop(columns=['tet_window_bucket'])
    return feat

def add_regime_weights(dates):
    w = np.ones(len(dates))
    w[(dates >= '2012-07-04') & (dates <= '2013-12-31')] = 0.25
    w[(dates >= '2014-01-01') & (dates <= '2018-12-31')] = 0.75
    w[(dates >= '2019-01-01') & (dates <= '2019-12-31')] = 0.80
    w[(dates >= '2020-01-01') & (dates <= '2022-12-31')] = 1.50
    return w

def train_eval_models(df_train, df_val, feat_train, feat_val, opt_w=None):
    dates_train = df_train.index
    dates_val = df_val.index
    
    y_train = np.log1p(df_train['Revenue'].values)
    y_val = df_val['Revenue'].values
    w_train = add_regime_weights(dates_train)
    
    preds = pd.DataFrame(index=dates_val)
    preds['actual'] = y_val
    
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(feat_train, y_train, sample_weight=w_train)
    preds['C_Ridge_lunar'] = np.expm1(ridge.predict(feat_val))
    
    rf = ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=12, min_samples_leaf=5)
    rf.fit(feat_train, y_train, sample_weight=w_train)
    preds['B_ExtraTrees_lunar'] = np.expm1(rf.predict(feat_val))
    
    hgb = HistGradientBoostingRegressor(random_state=42, max_depth=10, min_samples_leaf=5)
    hgb.fit(feat_train, y_train, sample_weight=w_train)
    preds['A_HGB_lunar'] = np.expm1(hgb.predict(feat_val))
    
    shape_train = df_train[(df_train.index >= '2014-01-01') & (df_train.index <= '2018-12-31')]
    shape_doy = shape_train.groupby(shape_train.index.dayofyear)['Revenue'].mean()
    shape_doy = shape_doy / shape_doy.mean()
    
    level_train = df_train[(df_train.index >= '2020-01-01') & (df_train.index <= '2022-12-31')]
    level_mean = level_train['Revenue'].mean() if len(level_train) > 0 else df_train['Revenue'].mean()
    
    # Calculate explicit tet adjustment from residuals of base seasonal shape
    tet_shape = {}
    for i in range(-30, 31): tet_shape[i] = 1.0
    
    # Simple explicit TET adjustment based on training data
    df_t = df_train.copy()
    df_t['tet_days_diff'] = feat_train['tet_days_diff']
    df_t['base_shape'] = df_t.index.dayofyear.map(shape_doy).fillna(1.0)
    df_t['residual_ratio'] = df_t['Revenue'] / (df_t['base_shape'] * level_mean)
    tet_adj = df_t[(df_t['tet_days_diff'] >= -30) & (df_t['tet_days_diff'] <= 30)].groupby('tet_days_diff')['residual_ratio'].median()
    
    pred_m4 = []
    for i, d in enumerate(dates_val):
        doy = d.dayofyear
        val_shape = shape_doy.get(doy, 1.0)
        if pd.isna(val_shape): val_shape = 1.0
        
        tdiff = feat_val.iloc[i]['tet_days_diff']
        if -30 <= tdiff <= 30:
            val_shape *= tet_adj.get(tdiff, 1.0)
            
        pred_m4.append(val_shape * level_mean)
    preds['D_SeasonalProfile_lunar_adjusted'] = pred_m4
    
    model_cols = ['A_HGB_lunar', 'B_ExtraTrees_lunar', 'C_Ridge_lunar', 'D_SeasonalProfile_lunar_adjusted']
    if opt_w is not None:
        preds['E_Ensemble_lunar'] = np.dot(preds[model_cols].values, opt_w)
    
    cogs_ratio = df_train['COGS'] / df_train['Revenue']
    cogs_ratio = cogs_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    cogs_ratio_recent = cogs_ratio.tail(365).mean()
    
    return preds, cogs_ratio_recent

def main():
    print("Loading data...")
    sales = pd.read_csv('data/sales.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values('Date').set_index('Date')
    
    sample_sub = pd.read_csv('sample_submission.csv')
    sample_sub['Date'] = pd.to_datetime(sample_sub['Date'])
    future_dates = pd.DatetimeIndex(sample_sub['Date'])
    
    print("Creating lunar features...")
    sales_feat = create_date_features(sales, start_index=0)
    sales_feat = add_tet_features(sales_feat)
    
    future_feat = create_date_features(pd.DataFrame(index=future_dates), start_index=len(sales))
    future_feat = add_tet_features(future_feat)
    
    # Audit
    sales_tet = sales.copy()
    sales_tet['tet_days_diff'] = sales_feat['tet_days_diff']
    audit_df = pd.DataFrame({'year': get_tet_dates().keys(), 'tet_date': get_tet_dates().values()})
    audit_df.to_csv('artifacts/tables/forecast_014_lunar_feature_audit.csv', index=False)
    
    plt.figure(figsize=(10,6))
    sales_tet[(sales_tet['tet_days_diff'] >= -30) & (sales_tet['tet_days_diff'] <= 30)].groupby('tet_days_diff')['Revenue'].mean().plot()
    plt.title("Average Revenue around Tet (-30 to +30 days)")
    plt.xlabel("Days from Tet")
    plt.ylabel("Avg Revenue")
    plt.grid(True)
    plt.savefig('artifacts/figures/forecast_014_tet_effect_audit.png')
    plt.close()
    
    folds = [
        {'name': 'Fold 1', 'train_end': '2018-12-31', 'val_start': '2019-01-01', 'val_end': '2020-07-01'},
        {'name': 'Fold 2', 'train_end': '2019-12-31', 'val_start': '2020-01-01', 'val_end': '2021-07-01'},
        {'name': 'Fold 3', 'train_end': '2020-12-31', 'val_start': '2021-01-01', 'val_end': '2022-07-01'},
        {'name': 'Fold 4', 'train_end': '2021-07-01', 'val_start': '2021-07-02', 'val_end': '2022-12-31'}
    ]
    
    fold_preds = {}
    print("Running backtest folds...")
    for f in folds:
        df_train = sales[sales.index < f['val_start']]
        df_val = sales[(sales.index >= f['val_start']) & (sales.index <= f['val_end'])]
        feat_train = sales_feat.loc[df_train.index]
        feat_val = sales_feat.loc[df_val.index]
        preds, _ = train_eval_models(df_train, df_val, feat_train, feat_val)
        fold_preds[f['name']] = preds
        
    all_preds = pd.concat(fold_preds.values())
    model_cols = ['A_HGB_lunar', 'B_ExtraTrees_lunar', 'C_Ridge_lunar', 'D_SeasonalProfile_lunar_adjusted']
    
    y = all_preds['actual'].values
    X = all_preds[model_cols].values
    opt_w, _ = optimize.nnls(X, y)
    if opt_w.sum() > 0:
        opt_w = opt_w / opt_w.sum()
    else:
        opt_w = np.ones(len(model_cols)) / len(model_cols)
        
    print("Lunar Ensemble Weights:", opt_w)
    
    all_preds['E_Ensemble_lunar'] = np.dot(all_preds[model_cols].values, opt_w)
    all_feat_val = sales_feat.loc[all_preds.index]
    
    results = []
    for col in model_cols + ['E_Ensemble_lunar']:
        m = get_metrics(all_preds['actual'].values, all_preds[col].values, all_preds.index, all_feat_val)
        m['model_name'] = col
        results.append(m)
        
    pd.DataFrame(results).to_csv('artifacts/tables/forecast_014_model_comparison.csv', index=False)
    
    pd.DataFrame(results).set_index('model_name')['MAE'].plot(kind='bar', figsize=(10,6))
    plt.title('Lunar Models Comparison - CV MAE')
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_014_model_comparison.png')
    plt.close()
    
    print("Running calibration sweep...")
    cal_factors = [0.80, 0.82, 0.8336, 0.845, 0.86, 0.875]
    sweep_results = []
    for cal in cal_factors:
        cal_pred = all_preds['E_Ensemble_lunar'] * cal
        m = get_metrics(all_preds['actual'].values, cal_pred.values, all_preds.index, all_feat_val)
        m['cal_factor'] = cal
        sweep_results.append(m)
        
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv('artifacts/tables/forecast_014_calibration_metrics.csv', index=False)
    
    print("Generating Future Forecasts...")
    future_preds, cogs_ratio_recent = train_eval_models(
        sales, pd.DataFrame(index=future_dates, columns=['Revenue']).fillna(0), 
        sales_feat, future_feat, opt_w=opt_w
    )
    
    output_df = pd.DataFrame({'Date': future_dates.values})
    
    def save_sub(variant, revenue_pred, filename):
        sub = pd.DataFrame({
            'Date': future_dates.values,
            'Revenue': revenue_pred,
            'COGS': revenue_pred * cogs_ratio_recent
        })
        sub.to_csv(filename, index=False)
        return sub
        
    r_base = future_preds['E_Ensemble_lunar'].values
    save_sub('lunar_base', r_base, 'artifacts/submissions/submission_forecast_014_lunar_base.csv')
    output_df['Revenue_lunar_base'] = r_base
    output_df['COGS_lunar_base'] = r_base * cogs_ratio_recent
    
    r_cal083 = r_base * 0.8336
    save_sub('lunar_cal083', r_cal083, 'artifacts/submissions/submission_forecast_014_lunar_cal083.csv')
    output_df['Revenue_lunar_cal083'] = r_cal083
    output_df['COGS_lunar_cal083'] = r_cal083 * cogs_ratio_recent
    
    r_cal086 = r_base * 0.86
    save_sub('lunar_cal086', r_cal086, 'artifacts/submissions/submission_forecast_014_lunar_cal086.csv')
    output_df['Revenue_lunar_cal086'] = r_cal086
    output_df['COGS_lunar_cal086'] = r_cal086 * cogs_ratio_recent
    
    # Read the best 012 candidate (which is 012_calibrated)
    sub012 = pd.read_csv('artifacts/submissions/submission_forecast_012_calibrated.csv')
    r_blend012 = 0.5 * sub012['Revenue'].values + 0.5 * r_cal083
    save_sub('lunar_blend012', r_blend012, 'artifacts/submissions/submission_forecast_014_lunar_blend012.csv')
    output_df['Revenue_lunar_blend012'] = r_blend012
    output_df['COGS_lunar_blend012'] = r_blend012 * cogs_ratio_recent
    
    output_df.to_csv('artifacts/forecasts/forecast_014_future_candidates.csv', index=False)
    
    # Manifest
    # Find best CV
    best_cal = sweep_df.loc[sweep_df['MAE'].idxmin()]
    
    manifest = [
        {'candidate_file': 'submission_forecast_014_lunar_cal083.csv', 'candidate_name': 'lunar_cal083', 'method': 'Lunar Ensemble * 0.8336', 'hypothesis': 'Aligns with previous best multiplier but uses true lunar features', 'submit_or_hold': 'submit'},
        {'candidate_file': 'submission_forecast_014_lunar_blend012.csv', 'candidate_name': 'lunar_blend012', 'method': '50% Lunar 0.8336 + 50% 012 Calibrated', 'hypothesis': 'Low-risk update mitigating pure model shift', 'submit_or_hold': 'submit'},
        {'candidate_file': 'submission_forecast_014_lunar_cal086.csv', 'candidate_name': 'lunar_cal086', 'method': 'Lunar Ensemble * 0.86', 'hypothesis': 'Conservative safety net', 'submit_or_hold': 'hold'},
        {'candidate_file': 'submission_forecast_014_lunar_base.csv', 'candidate_name': 'lunar_base', 'method': 'Uncalibrated Lunar Ensemble', 'hypothesis': 'Raw projection', 'submit_or_hold': 'hold'}
    ]
    pd.DataFrame(manifest).to_csv('artifacts/tables/forecast_014_candidate_manifest.csv', index=False)
    
    # Extra error analysis based on lunar feature
    all_preds['error'] = all_preds['E_Ensemble_lunar'] - all_preds['actual']
    all_preds['abs_error'] = all_preds['error'].abs()
    all_preds['tet_window'] = all_feat_val['tet_days_diff'].apply(lambda x: 'tet_window' if -30 <= x <= 30 else 'normal')
    error_analysis = all_preds.groupby('tet_window')[['error', 'abs_error']].mean().reset_index()
    error_analysis.to_csv('artifacts/tables/forecast_014_error_analysis.csv', index=False)
    
    plt.figure(figsize=(15,6))
    plt.plot(future_dates, output_df['Revenue_lunar_cal083'], label='Lunar Calib 0.8336')
    plt.plot(future_dates, output_df['Revenue_lunar_cal086'], label='Lunar Calib 0.86')
    plt.plot(future_dates, output_df['Revenue_lunar_blend012'], label='Blend 012 + Lunar')
    plt.title('Future Candidates Monthly Profiles (Lunar)')
    plt.legend()
    plt.savefig('artifacts/figures/forecast_014_monthly_candidate_comparison.png')
    plt.close()
    
    plt.figure(figsize=(15,6))
    plt.plot(sales.index[-365*2:], sales['Revenue'].tail(365*2), label='Historical')
    plt.plot(future_dates, output_df['Revenue_lunar_cal083'], label='Forecast')
    plt.title('Future Lunar Profile')
    plt.legend()
    plt.savefig('artifacts/figures/forecast_014_future_profile.png')
    plt.close()
    
    print("Done.")

if __name__ == '__main__':
    main()
