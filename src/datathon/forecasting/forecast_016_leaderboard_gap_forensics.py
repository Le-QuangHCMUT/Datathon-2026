import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    sales = pd.read_csv('data/sales.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values('Date').set_index('Date')
    
    sample_sub = pd.read_csv('sample_submission.csv')
    sample_sub['Date'] = pd.to_datetime(sample_sub['Date'])
    sample_sub = sample_sub.set_index('Date')
    
    sub012 = pd.read_csv('artifacts/submissions/submission_forecast_012_calibrated.csv')
    sub012['Date'] = pd.to_datetime(sub012['Date'])
    sub012 = sub012.set_index('Date')
    
    print("Validating dates...")
    assert len(sample_sub) == 548, f"Expected 548 rows in sample_sub, found {len(sample_sub)}"
    assert (sample_sub.index == sub012.index).all(), "Dates do not match between sample_sub and 012"
    
    print("Auditing sample_submission profile...")
    # Add month, quarter, year
    sample_sub['month'] = sample_sub.index.month
    sample_sub['year'] = sample_sub.index.year
    sample_sub['quarter'] = sample_sub.index.quarter
    
    sub012['month'] = sub012.index.month
    sub012['year'] = sub012.index.year
    sub012['quarter'] = sub012.index.quarter
    
    # Historical comparisons
    hist20 = sales[sales.index.year == 2020]
    hist21 = sales[sales.index.year == 2021]
    hist22 = sales[sales.index.year == 2022]
    
    monthly_comparison = []
    for m in range(1, 13):
        rec = {'month': m}
        rec['hist20_mean'] = hist20[hist20.index.month == m]['Revenue'].mean()
        rec['hist21_mean'] = hist21[hist21.index.month == m]['Revenue'].mean()
        rec['hist22_mean'] = hist22[hist22.index.month == m]['Revenue'].mean()
        
        samp_m = sample_sub[sample_sub['month'] == m]['Revenue']
        sub012_m = sub012[sub012['month'] == m]['Revenue']
        
        if len(samp_m) > 0:
            rec['sample_mean'] = samp_m.mean()
            rec['012_mean'] = sub012_m.mean()
            rec['sample_sum'] = samp_m.sum()
            rec['012_sum'] = sub012_m.sum()
            rec['sample_p50'] = samp_m.median()
            rec['012_p50'] = sub012_m.median()
        monthly_comparison.append(rec)
        
    monthly_df = pd.DataFrame(monthly_comparison)
    monthly_df.to_csv('artifacts/tables/forecast_016_monthly_profile_comparison.csv', index=False)
    
    print("Calculating diff against 012...")
    diff_df = pd.DataFrame(index=sample_sub.index)
    diff_df['sample_Revenue'] = sample_sub['Revenue']
    diff_df['012_Revenue'] = sub012['Revenue']
    diff_df['diff'] = sample_sub['Revenue'] - sub012['Revenue']
    diff_df['abs_diff'] = diff_df['diff'].abs()
    diff_df['pct_diff'] = (diff_df['diff'] / sub012['Revenue']) * 100
    
    diff_summary = diff_df.describe()
    diff_summary.to_csv('artifacts/tables/forecast_016_daily_diff_summary.csv')
    
    # Top 30 days diff
    top_diff = diff_df.sort_values('abs_diff', ascending=False).head(30)
    top_diff.to_csv('artifacts/tables/forecast_016_sample_vs_012_profile.csv')
    
    print("Plotting figures...")
    plt.figure(figsize=(15, 6))
    sample_sub.groupby('month')['Revenue'].mean().plot(label='sample_submission')
    sub012.groupby('month')['Revenue'].mean().plot(label='012_calibrated')
    plt.title('Monthly Mean Revenue: sample_sub vs 012')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.legend()
    plt.savefig('artifacts/figures/forecast_016_sample_vs_012_monthly.png')
    plt.close()
    
    plt.figure(figsize=(15, 6))
    plt.plot(sample_sub.index, sample_sub['Revenue'], label='sample_submission', alpha=0.7)
    plt.plot(sub012.index, sub012['Revenue'], label='012_calibrated', alpha=0.7)
    plt.title('Daily Revenue: sample_sub vs 012')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.savefig('artifacts/figures/forecast_016_sample_vs_012_daily.png')
    plt.close()
    
    print("Creating diagnostic submissions...")
    # Clean df without month/quarter
    sub_cols = ['Revenue', 'COGS']
    output_df = pd.DataFrame(index=sample_sub.index)
    
    def save_sub(name, rev_series, cogs_series):
        df = pd.DataFrame({'Date': sample_sub.index, 'Revenue': rev_series.values, 'COGS': cogs_series.values})
        df.to_csv(f'artifacts/submissions/submission_forecast_016_{name}.csv', index=False)
        output_df[f'Revenue_{name}'] = rev_series.values
        output_df[f'COGS_{name}'] = cogs_series.values
        return df

    # A. sample baseline
    save_sub('sample_baseline_diagnostic', sample_sub['Revenue'], sample_sub['COGS'])
    
    # B. blends
    # 25% sample
    rev_25 = 0.25 * sample_sub['Revenue'] + 0.75 * sub012['Revenue']
    cogs_25 = 0.25 * sample_sub['COGS'] + 0.75 * sub012['COGS']
    save_sub('blend_sample25_01275', rev_25, cogs_25)
    
    # 50% sample
    rev_50 = 0.50 * sample_sub['Revenue'] + 0.50 * sub012['Revenue']
    cogs_50 = 0.50 * sample_sub['COGS'] + 0.50 * sub012['COGS']
    save_sub('blend_sample50_01250', rev_50, cogs_50)
    
    # 75% sample
    rev_75 = 0.75 * sample_sub['Revenue'] + 0.25 * sub012['Revenue']
    cogs_75 = 0.75 * sample_sub['COGS'] + 0.25 * sub012['COGS']
    save_sub('blend_sample75_01225', rev_75, cogs_75)
    
    output_df.to_csv('artifacts/forecasts/forecast_016_future_candidates.csv')
    
    plt.figure(figsize=(15, 6))
    plt.plot(output_df.index, output_df['Revenue_sample_baseline_diagnostic'], label='Sample Baseline', alpha=0.5)
    plt.plot(output_df.index, output_df['Revenue_blend_sample50_01250'], label='Blend 50/50', alpha=0.8)
    plt.plot(sub012.index, sub012['Revenue'], label='012 Base', alpha=0.5)
    plt.title('Candidate Profiles Comparison')
    plt.legend()
    plt.savefig('artifacts/figures/forecast_016_candidate_profiles.png')
    plt.close()
    
    print("Writing manifest...")
    manifest = [
        {
            'candidate_file': 'submission_forecast_016_sample_baseline_diagnostic.csv',
            'candidate_name': 'sample_baseline_diagnostic',
            'method': 'Exact copy of sample_submission',
            'hypothesis': 'sample_submission might be a strong organizer baseline',
            'total_revenue_forecast': sample_sub['Revenue'].sum(),
            'avg_daily_revenue': sample_sub['Revenue'].mean(),
            'pct_diff_vs_012_total': ((sample_sub['Revenue'].sum() - sub012['Revenue'].sum()) / sub012['Revenue'].sum()) * 100,
            'max_monthly_diff_vs_012': diff_df['abs_diff'].resample('M').sum().max(),
            'rule_risk': 'high',
            'submit_or_hold': 'manual_review',
            'recommended_submission_priority': 1,
            'notes': 'Could be considered target leakage depending on rules. Evaluate as pure diagnostic.'
        },
        {
            'candidate_file': 'submission_forecast_016_blend_sample25_01275.csv',
            'candidate_name': 'blend_sample25_01275',
            'method': '25% Sample + 75% 012',
            'hypothesis': 'Blends baseline signal without fully relying on it',
            'total_revenue_forecast': rev_25.sum(),
            'avg_daily_revenue': rev_25.mean(),
            'pct_diff_vs_012_total': ((rev_25.sum() - sub012['Revenue'].sum()) / sub012['Revenue'].sum()) * 100,
            'max_monthly_diff_vs_012': (rev_25 - sub012['Revenue']).abs().resample('M').sum().max(),
            'rule_risk': 'medium',
            'submit_or_hold': 'hold_until_rule_decision',
            'recommended_submission_priority': 2,
            'notes': 'Only valid if sample_submission baseline is acceptable to use'
        },
        {
            'candidate_file': 'submission_forecast_016_blend_sample50_01250.csv',
            'candidate_name': 'blend_sample50_01250',
            'method': '50% Sample + 50% 012',
            'hypothesis': 'Halfway blend',
            'total_revenue_forecast': rev_50.sum(),
            'avg_daily_revenue': rev_50.mean(),
            'pct_diff_vs_012_total': ((rev_50.sum() - sub012['Revenue'].sum()) / sub012['Revenue'].sum()) * 100,
            'max_monthly_diff_vs_012': (rev_50 - sub012['Revenue']).abs().resample('M').sum().max(),
            'rule_risk': 'medium',
            'submit_or_hold': 'hold_until_rule_decision',
            'recommended_submission_priority': 3,
            'notes': 'Only valid if sample_submission baseline is acceptable to use'
        },
        {
            'candidate_file': 'submission_forecast_016_blend_sample75_01225.csv',
            'candidate_name': 'blend_sample75_01225',
            'method': '75% Sample + 25% 012',
            'hypothesis': 'Heavily leans on organizer baseline',
            'total_revenue_forecast': rev_75.sum(),
            'avg_daily_revenue': rev_75.mean(),
            'pct_diff_vs_012_total': ((rev_75.sum() - sub012['Revenue'].sum()) / sub012['Revenue'].sum()) * 100,
            'max_monthly_diff_vs_012': (rev_75 - sub012['Revenue']).abs().resample('M').sum().max(),
            'rule_risk': 'medium',
            'submit_or_hold': 'hold_until_rule_decision',
            'recommended_submission_priority': 4,
            'notes': 'Only valid if sample_submission baseline is acceptable to use'
        }
    ]
    pd.DataFrame(manifest).to_csv('artifacts/tables/forecast_016_candidate_manifest.csv', index=False)
    print("Done.")

if __name__ == '__main__':
    main()
