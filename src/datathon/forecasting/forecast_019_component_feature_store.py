import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def get_tet_dates():
    tet_dates = {
        2010: '2010-02-14', 2011: '2011-02-03', 2012: '2012-01-23', 2013: '2013-02-10',
        2014: '2014-01-31', 2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
        2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12',
        2022: '2022-02-01', 2023: '2023-01-22', 2024: '2024-02-10', 2025: '2025-01-29'
    }
    return {k: pd.to_datetime(v) for k, v in tet_dates.items()}

def create_calendar_features(dates):
    feat = pd.DataFrame(index=dates)
    feat['year'] = dates.year
    feat['month'] = dates.month
    feat['quarter'] = dates.quarter
    feat['day'] = dates.day
    feat['dayofweek'] = dates.dayofweek
    feat['dayofyear'] = dates.dayofyear
    feat['weekofyear'] = dates.isocalendar().week.astype(int)
    
    feat['is_month_start'] = dates.is_month_start.astype(int)
    feat['is_month_end'] = dates.is_month_end.astype(int)
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
    
    # Tet features
    tet_dates = get_tet_dates()
    diff_this = (dates - dates.year.map(tet_dates)).days
    diff_next = (dates - (dates.year + 1).map(tet_dates)).days
    diff_prev = (dates - (dates.year - 1).map(tet_dates)).days
    
    d1_abs = np.abs(diff_this)
    d2_abs = np.abs(diff_next)
    d3_abs = np.abs(diff_prev)
    
    diff = pd.Series(diff_this, index=dates)
    diff[d2_abs < np.abs(diff)] = diff_next[d2_abs < np.abs(diff)]
    diff[d3_abs < np.abs(diff)] = diff_prev[d3_abs < np.abs(diff)]
    
    feat['tet_days_diff'] = diff
    feat['abs_tet_days_diff'] = np.abs(diff)
    feat['tet_in_14'] = (feat['abs_tet_days_diff'] <= 14).astype(int)
    
    return feat

def mape(y_true, y_pred):
    mask = y_true > 0
    if not mask.any(): return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_predict(target_series, df_train, df_val, feat_train, feat_val, use_tree=False):
    dates_train = df_train.index
    w_train = np.ones(len(dates_train))
    w_train[(dates_train >= '2020-01-01') & (dates_train <= '2022-12-31')] = 1.5
    
    y_train = np.log1p(target_series.loc[dates_train].values)
    
    if use_tree:
        model = HistGradientBoostingRegressor(random_state=42, max_depth=8, min_samples_leaf=10)
        model.fit(feat_train, y_train, sample_weight=w_train)
    else:
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        model.fit(feat_train, y_train, sample_weight=w_train)
        
    pred = np.expm1(model.predict(feat_val))
    return pred

def main():
    print("Loading raw data...")
    sales = pd.read_csv('data/sales.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values('Date').set_index('Date')
    
    sample_sub = pd.read_csv('sample_submission.csv')
    sample_sub['Date'] = pd.to_datetime(sample_sub['Date'])
    sample_sub = sample_sub.set_index('Date')
    future_dates = sample_sub.index
    
    sub012 = pd.read_csv('artifacts/submissions/submission_forecast_012_calibrated.csv')
    sub012['Date'] = pd.to_datetime(sub012['Date'])
    sub012 = sub012.set_index('Date')
    
    orders = pd.read_csv('data/orders.csv')
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['Date'] = orders['order_date'].dt.date
    
    order_items = pd.read_csv('data/order_items.csv')
    products = pd.read_csv('data/products.csv')
    
    print("Building component feature store...")
    items = order_items.merge(orders[['order_id', 'Date']], on='order_id')
    items = items.merge(products[['product_id', 'category', 'cogs']], on='product_id', how='left')
    
    items['revenue'] = items['quantity'] * items['unit_price']
    items['cogs_total'] = items['quantity'] * items['cogs']
    
    daily = items.groupby('Date').agg(
        Revenue=('revenue', 'sum'),
        COGS=('cogs_total', 'sum'),
        units=('quantity', 'sum'),
        order_items_rows=('order_id', 'count'),
        orders=('order_id', 'nunique')
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    
    # Reconcile
    sales_subset = sales.loc[daily.index]
    rev_diff = np.abs(daily['Revenue'] - sales_subset['Revenue']).max()
    cogs_diff = np.abs(daily['COGS'] - sales_subset['COGS']).max()
    print(f"Max Revenue Diff: {rev_diff:.4f}")
    print(f"Max COGS Diff: {cogs_diff:.4f}")
    assert rev_diff < 1.0, "Revenue reconstruction failed"
    assert cogs_diff < 1.0, "COGS reconstruction failed"
    
    # Components
    daily['gross_profit'] = daily['Revenue'] - daily['COGS']
    daily['gross_margin'] = daily['gross_profit'] / daily['Revenue']
    daily['revenue_per_order'] = daily['Revenue'] / daily['orders']
    daily['revenue_per_unit'] = daily['Revenue'] / daily['units']
    daily['units_per_order'] = daily['units'] / daily['orders']
    daily['cogs_ratio'] = daily['COGS'] / daily['Revenue']
    
    # Category shares
    cat_daily = items.groupby(['Date', 'category'])['revenue'].sum().unstack(fill_value=0)
    cat_daily.index = pd.to_datetime(cat_daily.index)
    for c in cat_daily.columns:
        daily[f'share_{c}'] = cat_daily[c] / daily['Revenue']
        
    daily.to_csv('artifacts/tables/forecast_019_daily_component_feature_store.csv')
    
    print("Component diagnostics...")
    corr = daily[['Revenue', 'orders', 'units', 'revenue_per_order', 'revenue_per_unit', 'units_per_order']].corr()
    corr.to_csv('artifacts/tables/forecast_019_component_correlation_diagnostics.csv')
    
    # Stability
    def get_regime(y):
        if y < 2019: return '2014-2018'
        if y == 2019: return '2019'
        return '2020-2022'
    
    daily['regime'] = daily.index.year.map(get_regime)
    stability = daily.groupby('regime')[['orders', 'revenue_per_order', 'units_per_order']].mean()
    stability.to_csv('artifacts/tables/forecast_019_component_stability_diagnostics.csv')
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    daily.resample('M')['orders'].sum().plot(title='Monthly Orders')
    plt.subplot(2, 1, 2)
    daily.resample('M')['revenue_per_order'].mean().plot(title='Monthly Avg Revenue per Order')
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_019_component_trends.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    daily.resample('Y')[['orders', 'Revenue']].sum().plot(secondary_y='Revenue')
    plt.title('Orders vs Revenue (Annual)')
    plt.savefig('artifacts/figures/forecast_019_revenue_decomposition.png')
    plt.close()
    
    print("Residual forensics vs 012...")
    # I don't have exact historical predictions of 012 in the input files (I only have future_predictions.csv).
    # I will construct a comparable Ridge CV on Revenue directly to get residuals.
    feat_all = create_calendar_features(daily.index)
    
    folds = [
        {'name': 'Fold 1', 'train_end': '2018-12-31', 'val_start': '2019-01-01', 'val_end': '2020-07-01'},
        {'name': 'Fold 2', 'train_end': '2019-12-31', 'val_start': '2020-01-01', 'val_end': '2021-07-01'},
        {'name': 'Fold 3', 'train_end': '2020-12-31', 'val_start': '2021-01-01', 'val_end': '2022-07-01'},
        {'name': 'Fold 4', 'train_end': '2021-07-01', 'val_start': '2021-07-02', 'val_end': '2022-12-31'}
    ]
    
    residuals = []
    backtest_res = []
    
    print("Running backtest component models...")
    for f in folds:
        df_train = daily[daily.index < f['val_start']]
        df_val = daily[(daily.index >= f['val_start']) & (daily.index <= f['val_end'])]
        feat_train = feat_all.loc[df_train.index]
        feat_val = feat_all.loc[df_val.index]
        
        # 012-like Baseline (Ridge directly on Revenue)
        pred_rev_baseline = train_and_predict(df_train['Revenue'], df_train, df_val, feat_train, feat_val)
        residuals.append(pd.DataFrame({
            'Date': df_val.index,
            'actual': df_val['Revenue'],
            'pred_baseline': pred_rev_baseline * 0.8336  # rough calibration proxy
        }))
        
        # Candidate 1: component_base (orders * rpo)
        pred_orders = train_and_predict(df_train['orders'], df_train, df_val, feat_train, feat_val, use_tree=True)
        pred_rpo = train_and_predict(df_train['revenue_per_order'], df_train, df_val, feat_train, feat_val, use_tree=False)
        pred_rev_c1 = pred_orders * pred_rpo
        
        # Candidate 2: units_value (units * rpu)
        pred_units = train_and_predict(df_train['units'], df_train, df_val, feat_train, feat_val, use_tree=True)
        pred_rpu = train_and_predict(df_train['revenue_per_unit'], df_train, df_val, feat_train, feat_val, use_tree=False)
        pred_rev_c2 = pred_units * pred_rpu
        
        # Candidate 3: monthly_total_daily_shape
        # train monthly model
        monthly_train = df_train.resample('M')['Revenue'].sum()
        monthly_dates_train = monthly_train.index
        feat_month_train = create_calendar_features(monthly_dates_train)
        w_m_train = np.ones(len(monthly_train))
        w_m_train[(monthly_dates_train.year >= 2020)] = 1.5
        ridge_m = RidgeCV(alphas=[0.1, 1.0, 10.0])
        ridge_m.fit(feat_month_train, np.log1p(monthly_train.values), sample_weight=w_m_train)
        
        # daily shape profile
        shape_train = df_train[(df_train.index >= '2020-01-01')]
        daily_shape = shape_train.groupby(shape_train.index.day)['Revenue'].mean()
        daily_shape = daily_shape / daily_shape.sum()
        
        pred_rev_c3 = []
        for d in df_val.index:
            m_end = pd.Timestamp(year=d.year, month=d.month, day=d.days_in_month)
            fm = create_calendar_features(pd.DatetimeIndex([m_end]))
            m_pred = np.expm1(ridge_m.predict(fm))[0]
            sh = daily_shape.get(d.day, 1.0/30)
            pred_rev_c3.append(m_pred * sh)
            
        pred_rev_c3 = np.array(pred_rev_c3)
        
        # COGS
        pred_cogs_ratio = train_and_predict(df_train['cogs_ratio'], df_train, df_val, feat_train, feat_val)
        
        for i, d in enumerate(df_val.index):
            backtest_res.append({
                'Date': d,
                'fold': f['name'],
                'actual_revenue': df_val['Revenue'].iloc[i],
                'actual_cogs': df_val['COGS'].iloc[i],
                'pred_baseline': pred_rev_baseline[i],
                'pred_rev_c1': pred_rev_c1[i],
                'pred_rev_c2': pred_rev_c2[i],
                'pred_rev_c3': pred_rev_c3[i],
                'pred_cogs_ratio': pred_cogs_ratio[i]
            })

    res_df = pd.concat(residuals)
    res_df['error'] = res_df['pred_baseline'] - res_df['actual']
    res_df['month'] = res_df['Date'].dt.month
    res_df['dayofweek'] = res_df['Date'].dt.dayofweek
    res_df.groupby('month')['error'].mean().to_csv('artifacts/tables/forecast_019_residual_forensics_vs_012.csv')
    
    plt.figure(figsize=(10,5))
    res_df.groupby('month')['error'].mean().plot(kind='bar')
    plt.title('Baseline Residual Error by Month')
    plt.savefig('artifacts/figures/forecast_019_residual_forensics_vs_012.png')
    plt.close()

    bt_df = pd.DataFrame(backtest_res).set_index('Date')
    bt_df.to_csv('artifacts/forecasts/forecast_019_backtest_predictions.csv')
    
    # Calculate metrics
    metrics = []
    for c in ['pred_baseline', 'pred_rev_c1', 'pred_rev_c2', 'pred_rev_c3']:
        mae = mean_absolute_error(bt_df['actual_revenue'], bt_df[c])
        rmse = np.sqrt(mean_squared_error(bt_df['actual_revenue'], bt_df[c]))
        r2 = r2_score(bt_df['actual_revenue'], bt_df[c])
        mp = mape(bt_df['actual_revenue'], bt_df[c])
        bias = np.mean(bt_df[c] - bt_df['actual_revenue'])
        metrics.append({
            'model': c, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mp, 'bias': bias
        })
    pd.DataFrame(metrics).to_csv('artifacts/tables/forecast_019_backtest_metrics.csv', index=False)
    pd.DataFrame(metrics).to_csv('artifacts/tables/forecast_019_model_comparison.csv', index=False) # requested
    pd.DataFrame(metrics).to_csv('artifacts/tables/forecast_019_component_model_metrics.csv', index=False) # requested
    
    plt.figure(figsize=(15,6))
    plt.plot(bt_df.index, bt_df['actual_revenue'], label='Actual', alpha=0.5)
    plt.plot(bt_df.index, bt_df['pred_rev_c1'], label='C1 (orders*rpo)', alpha=0.5)
    plt.plot(bt_df.index, bt_df['pred_rev_c3'], label='C3 (monthly_shape)', alpha=0.5)
    plt.legend()
    plt.savefig('artifacts/figures/forecast_019_backtest_actual_vs_pred.png')
    plt.close()
    
    print("Generating Future Forecasts...")
    feat_future = create_calendar_features(future_dates)
    
    pred_orders_f = train_and_predict(daily['orders'], daily, pd.DataFrame(index=future_dates), feat_all, feat_future, use_tree=True)
    pred_rpo_f = train_and_predict(daily['revenue_per_order'], daily, pd.DataFrame(index=future_dates), feat_all, feat_future, use_tree=False)
    rev_c1_f = pred_orders_f * pred_rpo_f
    
    monthly_train = daily.resample('M')['Revenue'].sum()
    monthly_dates_train = monthly_train.index
    feat_month_train = create_calendar_features(monthly_dates_train)
    w_m_train = np.ones(len(monthly_train))
    w_m_train[(monthly_dates_train.year >= 2020)] = 1.5
    ridge_m = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge_m.fit(feat_month_train, np.log1p(monthly_train.values), sample_weight=w_m_train)
    
    shape_train = daily[(daily.index >= '2020-01-01')]
    daily_shape = shape_train.groupby(shape_train.index.day)['Revenue'].mean()
    daily_shape = daily_shape / daily_shape.sum()
    
    rev_c3_f = []
    for d in future_dates:
        m_end = pd.Timestamp(year=d.year, month=d.month, day=d.days_in_month)
        fm = create_calendar_features(pd.DatetimeIndex([m_end]))
        m_pred = np.expm1(ridge_m.predict(fm))[0]
        sh = daily_shape.get(d.day, 1.0/30)
        rev_c3_f.append(m_pred * sh)
    rev_c3_f = np.array(rev_c3_f)
    
    # Let's apply a 0.8336 calibration factor to C1 and C3 since they are raw uncalibrated like base
    rev_c1_f *= 0.8336
    rev_c3_f *= 0.8336
    
    pred_cogs_ratio_f = train_and_predict(daily['cogs_ratio'], daily, pd.DataFrame(index=future_dates), feat_all, feat_future)
    
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Revenue_component_base'] = rev_c1_f
    future_df['COGS_component_base'] = rev_c1_f * pred_cogs_ratio_f
    future_df['Revenue_monthly_total_daily_shape'] = rev_c3_f
    future_df['COGS_monthly_total_daily_shape'] = rev_c3_f * pred_cogs_ratio_f
    
    future_df['Revenue_component_blend012_25'] = 0.25 * rev_c1_f + 0.75 * sub012['Revenue'].values
    future_df['COGS_component_blend012_25'] = future_df['Revenue_component_blend012_25'] * pred_cogs_ratio_f
    future_df['Revenue_component_blend012_50'] = 0.50 * rev_c1_f + 0.50 * sub012['Revenue'].values
    future_df['COGS_component_blend012_50'] = future_df['Revenue_component_blend012_50'] * pred_cogs_ratio_f
    
    future_df.to_csv('artifacts/forecasts/forecast_019_future_candidates.csv', index=False)
    
    def save_sub(name):
        sub = pd.DataFrame({
            'Date': future_df['Date'],
            'Revenue': future_df[f'Revenue_{name}'],
            'COGS': future_df[f'COGS_{name}']
        })
        sub.to_csv(f'artifacts/submissions/submission_forecast_019_{name}.csv', index=False)
        
    save_sub('component_base')
    save_sub('monthly_total_daily_shape')
    save_sub('component_blend012_25')
    save_sub('component_blend012_50')
    
    plt.figure(figsize=(15,6))
    plt.plot(future_df['Date'], future_df['Revenue_component_base'], label='Component Base')
    plt.plot(future_df['Date'], future_df['Revenue_monthly_total_daily_shape'], label='Monthly Shape')
    plt.plot(sub012.index, sub012['Revenue'], label='012 Calibrated', alpha=0.5)
    plt.legend()
    plt.savefig('artifacts/figures/forecast_019_future_candidate_profiles.png')
    plt.close()
    
    manifest = [
        {'candidate_file': 'submission_forecast_019_component_base.csv', 'candidate_name': 'component_base', 'method': 'orders * rpo', 'hypothesis': 'Decomposing volume and value is superior', 'submit_or_hold': 'hold'},
        {'candidate_file': 'submission_forecast_019_monthly_total_daily_shape.csv', 'candidate_name': 'monthly_total_daily_shape', 'method': 'monthly_total * daily_share', 'hypothesis': 'Protects monthly level, fixes shape', 'submit_or_hold': 'hold'},
        {'candidate_file': 'submission_forecast_019_component_blend012_25.csv', 'candidate_name': 'component_blend012_25', 'method': '25% C1 + 75% 012', 'hypothesis': 'Low risk blend', 'submit_or_hold': 'hold'},
        {'candidate_file': 'submission_forecast_019_component_blend012_50.csv', 'candidate_name': 'component_blend012_50', 'method': '50% C1 + 50% 012', 'hypothesis': 'Balanced blend', 'submit_or_hold': 'hold'}
    ]
    pd.DataFrame(manifest).to_csv('artifacts/tables/forecast_019_candidate_manifest.csv', index=False)
    print("Done.")

if __name__ == '__main__':
    main()
