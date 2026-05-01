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
    future_dates = pd.DatetimeIndex(sample_sub['Date'])
    
    sub012 = pd.read_csv('artifacts/submissions/submission_forecast_012_calibrated.csv')
    sub012['Date'] = pd.to_datetime(sub012['Date'])
    sub012 = sub012.set_index('Date')
    
    # Check if failed submissions exist
    failed_subs = {}
    f1 = 'artifacts/submissions/submission_forecast_013_q2_adjusted.csv'
    if os.path.exists(f1):
        df = pd.read_csv(f1)
        df['Date'] = pd.to_datetime(df['Date'])
        failed_subs['013_q2_adjusted'] = df.set_index('Date')
        
    f2 = 'artifacts/submissions/submission_forecast_014_lunar_blend012.csv'
    if os.path.exists(f2):
        df = pd.read_csv(f2)
        df['Date'] = pd.to_datetime(df['Date'])
        failed_subs['014_lunar_blend012'] = df.set_index('Date')

    print("Running failure diff diagnosis...")
    diagnosis_records = []
    if failed_subs:
        plt.figure(figsize=(15, 6))
        for name, sub in failed_subs.items():
            diff = sub['Revenue'] - sub012['Revenue']
            pct_diff = (diff / sub012['Revenue']) * 100
            
            # Monthly
            diff_monthly = diff.resample('M').mean()
            pct_diff_monthly = pct_diff.resample('M').mean()
            
            # Quarter
            sub['quarter'] = sub.index.quarter
            sub012_temp = sub012.copy()
            sub012_temp['quarter'] = sub012_temp.index.quarter
            q_diff = sub.groupby('quarter')['Revenue'].sum() - sub012_temp.groupby('quarter')['Revenue'].sum()
            q_pct_diff = (q_diff / sub012_temp.groupby('quarter')['Revenue'].sum()) * 100
            
            for m in diff_monthly.index:
                diagnosis_records.append({
                    'failed_variant': name,
                    'month': m.to_period('M'),
                    'mean_diff': diff_monthly[m],
                    'mean_pct_diff': pct_diff_monthly[m]
                })
                
            plt.plot(diff.index, diff, label=f'{name} diff vs 012', alpha=0.7)
            
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Failed Variants Daily Difference vs 012 Calibrated")
        plt.legend()
        plt.tight_layout()
        plt.savefig('artifacts/figures/forecast_015_failed_variant_diff.png')
        plt.close()
        
    diag_df = pd.DataFrame(diagnosis_records)
    if not diag_df.empty:
        diag_df.to_csv('artifacts/tables/forecast_015_failure_diff_diagnosis.csv', index=False)
        
    print("Auditing 012 profile...")
    hist = sales[(sales.index >= '2020-01-01') & (sales.index <= '2022-12-31')].copy()
    hist['month'] = hist.index.month
    
    sub012_audit = sub012.copy()
    sub012_audit['month'] = sub012_audit.index.month
    
    audit_records = []
    for m in range(1, 13):
        h_m = hist[hist['month'] == m]['Revenue']
        s_m = sub012_audit[sub012_audit['month'] == m]['Revenue']
        
        audit_records.append({
            'month': m,
            'hist_mean': h_m.mean(),
            '012_mean': s_m.mean(),
            'hist_p10': h_m.quantile(0.1),
            '012_p10': s_m.quantile(0.1),
            'hist_p50': h_m.median(),
            '012_p50': s_m.median(),
            'hist_p90': h_m.quantile(0.9),
            '012_p90': s_m.quantile(0.9)
        })
        
    audit_df = pd.DataFrame(audit_records)
    audit_df.to_csv('artifacts/tables/forecast_015_012_profile_audit.csv', index=False)
    
    print("Creating low-perturbation candidates...")
    candidates = {}
    
    # Calculate COGS ratio exactly from 012
    cogs_ratio = sub012['COGS'] / sub012['Revenue']
    
    # Candidate A: smooth012
    r_smooth = sub012['Revenue'].copy()
    r_smooth_ma = r_smooth.rolling(window=3, center=True).mean().fillna(r_smooth)
    
    # Preserve monthly totals
    r_smooth_adj = pd.Series(index=r_smooth.index, dtype=float)
    for m in r_smooth.index.to_period('M').unique():
        mask = r_smooth.index.to_period('M') == m
        orig_sum = r_smooth[mask].sum()
        new_sum = r_smooth_ma[mask].sum()
        r_smooth_adj[mask] = r_smooth_ma[mask] * (orig_sum / new_sum)
        
    candidates['smooth012'] = r_smooth_adj
    
    # Candidate B: clip_spikes012
    r_clip = sub012['Revenue'].copy()
    # Historical p97 by month
    p97_by_month = hist.groupby('month')['Revenue'].quantile(0.97)
    
    r_clip_adj = pd.Series(index=r_clip.index, dtype=float)
    for m in r_clip.index.to_period('M').unique():
        mask = r_clip.index.to_period('M') == m
        month_idx = m.month
        clip_val = p97_by_month.get(month_idx, r_clip[mask].max())
        
        # If the forecast has spikes way above historical p97, clip them.
        # But allow 20% growth over historical p97 just in case
        clip_val = clip_val * 1.2
        
        orig_vals = r_clip[mask]
        clipped_vals = orig_vals.clip(upper=clip_val)
        
        excess = orig_vals.sum() - clipped_vals.sum()
        
        # Redistribute excess proportionally to non-clipped days
        non_clipped_mask = clipped_vals < clip_val
        if excess > 0 and non_clipped_mask.any():
            dist = clipped_vals[non_clipped_mask]
            # add excess proportionally
            clipped_vals[non_clipped_mask] += excess * (dist / dist.sum())
        elif excess > 0:
            # if all clipped, just distribute evenly
            clipped_vals += excess / len(clipped_vals)
            
        r_clip_adj[mask] = clipped_vals
        
    candidates['clip_spikes012'] = r_clip_adj
    
    # Candidate C: monthcal012
    # Small calibration based on average of previous 012 folds if available.
    # Here we will do a neutral +1% on historically underforecasted months (e.g. Q1, Q4)
    # and -1% on historically overforecasted months, max +-2%
    r_monthcal = sub012['Revenue'].copy()
    monthcal_adj = pd.Series(index=r_monthcal.index, dtype=float)
    for m in r_monthcal.index.to_period('M').unique():
        mask = r_monthcal.index.to_period('M') == m
        q = m.quarter
        # If Q2 was historically overforecasted, Q2 adjusted failed because it was too strong (-10%).
        # We will do a tiny -1% on Q2, +1% on Q4.
        factor = 1.0
        if q == 2: factor = 0.99
        elif q == 4: factor = 1.01
        
        monthcal_adj[mask] = r_monthcal[mask] * factor
        
    candidates['monthcal012'] = monthcal_adj
    
    print("Generating Candidate Submissions...")
    output_df = pd.DataFrame({'Date': future_dates.values})
    manifest_records = []
    
    for name, rev_series in candidates.items():
        cogs_series = rev_series * cogs_ratio
        
        sub = pd.DataFrame({
            'Date': future_dates.values,
            'Revenue': rev_series.values,
            'COGS': cogs_series.values
        })
        sub.to_csv(f'artifacts/submissions/submission_forecast_015_{name}.csv', index=False)
        
        output_df[f'Revenue_{name}'] = rev_series.values
        output_df[f'COGS_{name}'] = cogs_series.values
        
        diff = rev_series - sub012['Revenue']
        pct_diff = diff / sub012['Revenue']
        
        max_abs_pct = pct_diff.abs().max() * 100
        mean_abs_pct = pct_diff.abs().mean() * 100
        total_pct = (rev_series.sum() - sub012['Revenue'].sum()) / sub012['Revenue'].sum() * 100
        
        high_risk = abs(total_pct) > 1.0 or max_abs_pct > 30.0 # smoothed can change a spike by >3% easily
        
        priority = 1 if name == 'smooth012' else (2 if name == 'clip_spikes012' else 3)
        action = 'submit' if name == 'smooth012' else 'hold'
        
        manifest_records.append({
            'candidate_file': f'submission_forecast_015_{name}.csv',
            'candidate_name': name,
            'method': name,
            'hypothesis': f'Low perturbation refinement - {name}',
            'max_abs_pct_change_vs_012': max_abs_pct,
            'mean_abs_pct_change_vs_012': mean_abs_pct,
            'total_revenue_change_pct': total_pct,
            'changed_windows': 'all' if name != 'monthcal012' else 'Q2, Q4',
            'expected_public_risk': 'high' if high_risk else 'low',
            'submit_or_hold': action,
            'recommended_priority': priority
        })
        
    output_df.to_csv('artifacts/forecasts/forecast_015_future_candidates.csv', index=False)
    
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv('artifacts/tables/forecast_015_candidate_manifest.csv', index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(sub012.index, sub012['Revenue'], label='012 Calibrated (Base)', linewidth=2)
    for name, rev_series in candidates.items():
        plt.plot(rev_series.index, rev_series, label=name, alpha=0.7)
    plt.title("Candidate Refinements vs 012 Calibrated Base")
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_015_candidate_comparison.png')
    plt.close()
    
    print("Done.")

if __name__ == '__main__':
    main()
