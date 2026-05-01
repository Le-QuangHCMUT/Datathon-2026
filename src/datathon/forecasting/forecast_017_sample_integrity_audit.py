import os
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime

def file_info(path):
    if not os.path.exists(path): return None
    stat = os.stat(path)
    with open(path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    return {
        'path': path,
        'size': stat.st_size,
        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'sha256': sha256
    }

def main():
    print("Running Sample Integrity Audit...")
    results = []
    
    # 1. Print current root sample_submission.csv metadata
    path_sample = 'sample_submission.csv'
    info_sample = file_info(path_sample)
    if info_sample:
        df_sample = pd.read_csv(path_sample)
        info_sample['shape'] = str(df_sample.shape)
        info_sample['columns'] = str(list(df_sample.columns))
        info_sample['date_min'] = df_sample['Date'].min()
        info_sample['date_max'] = df_sample['Date'].max()
        info_sample['rev_min'] = df_sample['Revenue'].min()
        info_sample['rev_max'] = df_sample['Revenue'].max()
        info_sample['cogs_min'] = df_sample['COGS'].min()
        info_sample['cogs_max'] = df_sample['COGS'].max()
        
        print("--- sample_submission.csv metadata ---")
        for k, v in info_sample.items():
            print(f"{k}: {v}")
            
        print("\nFirst 10 rows:")
        print(df_sample.head(10))
        print("\nLast 10 rows:")
        print(df_sample.tail(10))
        
        # 2. Compare against bootstrap-known values
        b_rev1 = df_sample.loc[df_sample['Date'] == '2023-01-01', 'Revenue'].values[0]
        b_cogs1 = df_sample.loc[df_sample['Date'] == '2023-01-01', 'COGS'].values[0]
        b_rev2 = df_sample.loc[df_sample['Date'] == '2023-01-02', 'Revenue'].values[0]
        b_cogs2 = df_sample.loc[df_sample['Date'] == '2023-01-02', 'COGS'].values[0]
        
        match_b1 = np.isclose(b_rev1, 2665507.20) and np.isclose(b_cogs1, 2518885.15)
        match_b2 = np.isclose(b_rev2, 1280007.89) and np.isclose(b_cogs2, 1136463.00)
        
        matches_bootstrap = match_b1 and match_b2
        
        results.append({
            'check': 'matches_bootstrap_values',
            'status': matches_bootstrap,
            'details': f"2023-01-01 Rev={b_rev1}, 2023-01-02 Rev={b_rev2}"
        })
        print(f"\nMatches bootstrap values: {matches_bootstrap}")
    else:
        print("sample_submission.csv is missing!")
        return
        
    # 3. Compare against artifacts/diagnostics/schema_summary.csv
    schema_path = 'artifacts/diagnostics/schema_summary.csv'
    if os.path.exists(schema_path):
        schema_df = pd.read_csv(schema_path)
        schema_sample = schema_df[schema_df['file_path'] == 'sample_submission.csv']
        if not schema_sample.empty:
            print("\n--- schema_summary.csv values for sample_submission.csv ---")
            print(schema_sample[['column_name', 'sample_values']])
            results.append({'check': 'schema_summary_found', 'status': True, 'details': 'Found in schema_summary.csv'})
    
    # 4. Compare current sample_submission.csv to submission_forecast_016_sample_baseline_diagnostic.csv
    diag_path = 'artifacts/submissions/submission_forecast_016_sample_baseline_diagnostic.csv'
    if os.path.exists(diag_path):
        df_diag = pd.read_csv(diag_path)
        
        # Merge by Date to check exact equality
        merged = pd.merge(df_sample, df_diag, on='Date', suffixes=('_root', '_diag'))
        merged['rev_diff'] = merged['Revenue_root'] - merged['Revenue_diag']
        merged['cogs_diff'] = merged['COGS_root'] - merged['COGS_diag']
        
        exact_equal = (merged['rev_diff'].abs() < 1e-6).all() and (merged['cogs_diff'].abs() < 1e-6).all()
        results.append({
            'check': 'equals_forecast_016_diagnostic',
            'status': exact_equal,
            'details': f"Total rev diff: {merged['rev_diff'].abs().sum()}, Total cogs diff: {merged['cogs_diff'].abs().sum()}"
        })
        
        print("\n--- Compare sample_submission to diagnostic ---")
        print(f"Exactly equal: {exact_equal}")
        print(f"Total Absolute Rev Diff: {merged['rev_diff'].abs().sum()}")
        print(f"Total Absolute COGS Diff: {merged['cogs_diff'].abs().sum()}")
        
        merged.to_csv('artifacts/tables/forecast_017_sample_vs_diagnostic_diff.csv', index=False)
        print("Saved forecast_017_sample_vs_diagnostic_diff.csv")
    
    # Check 016 timestamps
    info_016_script = file_info('src/datathon/forecasting/forecast_016_leaderboard_gap_forensics.py')
    info_016_diag = file_info(diag_path)
    
    results.append({'check': 'sample_modified_time', 'status': True, 'details': info_sample['modified_time']})
    if info_016_script:
        results.append({'check': 'script_016_modified_time', 'status': True, 'details': info_016_script['modified_time']})
    
    pd.DataFrame(results).to_csv('artifacts/tables/forecast_017_sample_integrity_checks.csv', index=False)
    
if __name__ == '__main__':
    main()
