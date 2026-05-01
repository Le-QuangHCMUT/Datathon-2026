import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.optimize as optimize
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
    bias = np.mean(y_pred - y_true)
    if dates is not None:
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'month': dates.to_period('M'), 'quarter': dates.quarter})
        monthly_mae = df.groupby('month').apply(lambda x: mean_absolute_error(x['y_true'], x['y_pred'])).mean()
        for q in [1, 2, 3, 4]:
            q_df = df[df['quarter'] == q]
            if len(q_df) > 0:
                q_mae[f'Q{q}'] = mean_absolute_error(q_df['y_true'], q_df['y_pred'])
        
    res = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mp, 'bias': bias, 'monthly_MAE': monthly_mae}
    for q in [1, 2, 3, 4]:
        res[f'Q{q}_MAE'] = q_mae.get(f'Q{q}', np.nan)
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
    
    feat['dom_bucket_first3'] = (dates.day <= 3).astype(int)
    feat['dom_bucket_last3'] = (feat['days_to_eom'] <= 2).astype(int)
    feat['dom_bucket_last7'] = (feat['days_to_eom'] <= 6).astype(int)
    feat['dom_bucket_mid'] = ((dates.day > 3) & (feat['days_to_eom'] > 6)).astype(int)
    
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

def train_eval_models(df_train, df_val, feat_train, feat_val, opt_w=None):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    dates_train = df_train.index
    dates_val = df_val.index
    
    y_train = np.log1p(df_train['Revenue'].values)
    y_val = df_val['Revenue'].values
    w_train = add_regime_weights(dates_train)
    
    preds = pd.DataFrame(index=dates_val)
    preds['actual'] = y_val
    
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(feat_train, y_train, sample_weight=w_train)
    preds['M1_Ridge'] = np.expm1(ridge.predict(feat_val))
    
    rf = ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=12, min_samples_leaf=5)
    rf.fit(feat_train, y_train, sample_weight=w_train)
    preds['M2_ExtraTrees'] = np.expm1(rf.predict(feat_val))
    
    hgb = HistGradientBoostingRegressor(random_state=42, max_depth=10, min_samples_leaf=5)
    hgb.fit(feat_train, y_train, sample_weight=w_train)
    preds['M3_HGB'] = np.expm1(hgb.predict(feat_val))
    
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
    
    if opt_w is not None:
        model_cols = ['M1_Ridge', 'M2_ExtraTrees', 'M3_HGB', 'M4_SeasonalProfile', 'M5_QSpecialist', 'M5_QSpecialist_Blend']
        preds['M6_Ensemble'] = np.dot(preds[model_cols].values, opt_w)
    
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
    
    print("Creating features with FIXED time_index...")
    tet_dates = detect_tet_proxy(sales)
    sales_feat = create_date_features(sales, start_index=0)
    sales_feat = add_tet_features(sales_feat, tet_dates)
    
    future_feat = create_date_features(pd.DataFrame(index=future_dates), start_index=len(sales))
    future_feat = add_tet_features(future_feat, tet_dates)
    
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
    model_cols = ['M1_Ridge', 'M2_ExtraTrees', 'M3_HGB', 'M4_SeasonalProfile', 'M5_QSpecialist', 'M5_QSpecialist_Blend']
    
    y = all_preds['actual'].values
    X = all_preds[model_cols].values
    opt_w, _ = optimize.nnls(X, y)
    if opt_w.sum() > 0:
        opt_w = opt_w / opt_w.sum()
    else:
        opt_w = np.ones(len(model_cols)) / len(model_cols)
        
    print("Ensemble Weights:", opt_w)
    
    all_preds['M6_Ensemble'] = np.dot(all_preds[model_cols].values, opt_w)
    
    print("Running calibration sweep...")
    cal_factors = [0.76, 0.78, 0.80, 0.81, 0.82, 0.8336, 0.845, 0.86, 0.875, 0.89, 0.92]
    sweep_results = []
    
    for cal in cal_factors:
        cal_pred = all_preds['M6_Ensemble'] * cal
        m = get_metrics(all_preds['actual'].values, cal_pred.values, all_preds.index)
        m['cal_factor'] = cal
        sweep_results.append(m)
        
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv('artifacts/tables/forecast_013_calibration_sweep_metrics.csv', index=False)
    
    # Q-specialist diagnostics
    print("Running Q-Specialist Diagnostics...")
    future_preds, cogs_ratio_recent = train_eval_models(
        sales, pd.DataFrame(index=future_dates, columns=['Revenue']).fillna(0), 
        sales_feat, future_feat, opt_w=opt_w
    )
    future_preds['quarter'] = future_preds.index.quarter
    qdiag = future_preds.groupby('quarter')['M5_QSpecialist'].describe()
    qdiag.to_csv('artifacts/tables/forecast_013_qspecialist_diagnostics.csv')
    
    # Q2 error correction
    print("Testing Q2 specific error corrections...")
    q2_factors = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
    q2_results = []
    base_calibrated_pred = all_preds['M6_Ensemble'] * 0.8336
    
    for q2f in q2_factors:
        pred_adj = base_calibrated_pred.copy()
        pred_adj[all_preds.index.quarter == 2] *= q2f
        m = get_metrics(all_preds['actual'].values, pred_adj.values, all_preds.index)
        m['q2_factor'] = q2f
        q2_results.append(m)
        
    q2_df = pd.DataFrame(q2_results)
    q2_df.to_csv('artifacts/tables/forecast_013_q2_error_correction_metrics.csv', index=False)
    
    print("Generating Future Forecasts...")
    
    # The actual output DF
    output_df = pd.DataFrame({'Date': future_dates.values})
    
    def save_sub(variant, revenue_pred, filename):
        sub = pd.DataFrame({
            'Date': future_dates.values,
            'Revenue': revenue_pred,
            'COGS': revenue_pred * cogs_ratio_recent
        })
        sub.to_csv(filename, index=False)
        return sub
        
    # Build the variants
    # 0.81
    r_81 = future_preds['M6_Ensemble'].values * 0.81
    save_sub('calib_081', r_81, 'artifacts/submissions/submission_forecast_013_calib_081.csv')
    output_df['Revenue_calib_081'] = r_81
    output_df['COGS_calib_081'] = r_81 * cogs_ratio_recent
    
    # 0.86
    r_86 = future_preds['M6_Ensemble'].values * 0.86
    save_sub('calib_086', r_86, 'artifacts/submissions/submission_forecast_013_calib_086.csv')
    output_df['Revenue_calib_086'] = r_86
    output_df['COGS_calib_086'] = r_86 * cogs_ratio_recent
    
    # QSpecialist Fixed
    r_qspec = future_preds['M5_QSpecialist_Blend'].values
    save_sub('qspecialist_fixed', r_qspec, 'artifacts/submissions/submission_forecast_013_qspecialist_fixed.csv')
    output_df['Revenue_qspecialist_fixed'] = r_qspec
    output_df['COGS_qspecialist_fixed'] = r_qspec * cogs_ratio_recent
    
    # Best Q2 adjustment from CV (look at minimum MAE in Q2)
    best_q2f = q2_df.loc[q2_df['MAE'].idxmin(), 'q2_factor']
    print(f"Best Q2 Adjustment Factor: {best_q2f}")
    
    r_q2_adj = future_preds['M6_Ensemble'].values * 0.8336
    r_q2_adj[future_preds.index.quarter == 2] *= best_q2f
    save_sub('q2_adjusted', r_q2_adj, 'artifacts/submissions/submission_forecast_013_q2_adjusted.csv')
    output_df['Revenue_q2_adjusted'] = r_q2_adj
    output_df['COGS_q2_adjusted'] = r_q2_adj * cogs_ratio_recent
    
    output_df.to_csv('artifacts/forecasts/forecast_013_future_candidates.csv', index=False)
    
    # Manifest
    manifest = [
        {'candidate_file': 'submission_forecast_013_calib_081.csv', 'candidate_name': 'calib_081', 'method': 'Ensemble * 0.81', 'hypothesis': 'Further level reduction', 'submit_or_hold': 'submit'},
        {'candidate_file': 'submission_forecast_013_calib_086.csv', 'candidate_name': 'calib_086', 'method': 'Ensemble * 0.86', 'hypothesis': 'Moderate level reduction', 'submit_or_hold': 'submit'},
        {'candidate_file': 'submission_forecast_013_q2_adjusted.csv', 'candidate_name': 'q2_adjusted', 'method': f'0.8336 with Q2 * {best_q2f}', 'hypothesis': 'Fix Q2 weakness', 'submit_or_hold': 'hold'},
        {'candidate_file': 'submission_forecast_013_qspecialist_fixed.csv', 'candidate_name': 'qspecialist_fixed', 'method': 'M5 Q-Specialist Fixed', 'hypothesis': 'Fixed time_index bug', 'submit_or_hold': 'hold'}
    ]
    pd.DataFrame(manifest).to_csv('artifacts/tables/forecast_013_candidate_manifest.csv', index=False)
    
    # Figures
    plt.figure(figsize=(10,6))
    plt.plot(sweep_df['cal_factor'], sweep_df['MAE'], marker='o')
    plt.title('Calibration Sweep (CV MAE)')
    plt.xlabel('Calibration Factor')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.savefig('artifacts/figures/forecast_013_calibration_sweep.png')
    
    plt.figure(figsize=(10,6))
    plt.plot(q2_df['q2_factor'], q2_df['MAE'], marker='o', label='Total MAE')
    plt.plot(q2_df['q2_factor'], q2_df['Q2_MAE'], marker='x', label='Q2 MAE')
    plt.title('Q2 Error Profile')
    plt.xlabel('Q2 Factor')
    plt.legend()
    plt.grid(True)
    plt.savefig('artifacts/figures/forecast_013_q2_error_profile.png')
    
    plt.figure(figsize=(15,6))
    plt.plot(future_dates, output_df['Revenue_calib_081'], label='Calib 0.81')
    plt.plot(future_dates, output_df['Revenue_calib_086'], label='Calib 0.86')
    plt.plot(future_dates, output_df['Revenue_q2_adjusted'], label='Q2 Adjusted')
    plt.plot(future_dates, output_df['Revenue_qspecialist_fixed'], label='QSpecialist Fixed', alpha=0.5)
    plt.title('Future Candidates Monthly Profiles')
    plt.legend()
    plt.savefig('artifacts/figures/forecast_013_candidate_monthly_profiles.png')
    
    print("Done.")

if __name__ == '__main__':
    main()
