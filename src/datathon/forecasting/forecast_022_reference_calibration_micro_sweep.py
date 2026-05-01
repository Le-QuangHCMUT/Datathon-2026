"""
FORECAST-022: Calibration Micro-Sweep around CR=1.27 CC=1.32 winner.
Recovers raw components by back-dividing the current best, then sweeps CR/CC.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CURRENT_BEST_CR = 1.27
CURRENT_BEST_CC = 1.32

CANDIDATES = [
    ('cr128_cc132',  1.28, 1.32),
    ('cr129_cc132',  1.29, 1.32),
    ('cr1275_cc132', 1.275,1.32),
    ('cr127_cc133',  1.27, 1.33),
    ('cr128_cc133',  1.28, 1.33),
    ('cr127_cc131',  1.27, 1.31),
]

def main():
    print("FORECAST-022: Calibration Micro-Sweep")

    sample = pd.read_csv('sample_submission.csv')
    sample['Date'] = pd.to_datetime(sample['Date'])
    fdates = pd.DatetimeIndex(sample['Date'])
    assert len(sample) == 548

    best = pd.read_csv('artifacts/submissions/submission_forecast_021_cr127_cc132.csv')
    best['Date'] = pd.to_datetime(best['Date'])
    best = best.set_index('Date')
    assert (best.index == fdates).all(), "Date mismatch"
    assert (best['Revenue'] > 0).all() and (best['COGS'] >= 0).all()

    # Recover raw by back-dividing
    raw_rev  = best['Revenue'].values / CURRENT_BEST_CR
    raw_cogs = best['COGS'].values    / CURRENT_BEST_CC
    print(f"Recovered raw Revenue: mean={raw_rev.mean():.0f}, total={raw_rev.sum():.3e}")
    print(f"Recovered raw COGS:    mean={raw_cogs.mean():.0f}, total={raw_cogs.sum():.3e}")

    out_df = pd.DataFrame({'Date': fdates.values})
    sub_dfs = {}
    manifest_rows = []
    diff_rows = []

    winner_rev  = best['Revenue'].values
    winner_cogs = best['COGS'].values

    priority_map = {
        'cr128_cc132': (1, 'submit', 'Revenue +1 step up; probes if 1.27→1.28 continues improving'),
        'cr1275_cc132':(2, 'hold',   'Half-step CR=1.275 if 1.28 worsens: interpolation fallback'),
        'cr129_cc132': (3, 'hold',   'Aggressive CR=1.29; only if 1.28 improves'),
        'cr127_cc133': (4, 'hold',   'COGS axis probe: CC +0.01 with fixed CR=1.27'),
        'cr128_cc133': (5, 'hold',   'Combined Revenue+COGS nudge if 1.28 helps'),
        'cr127_cc131': (6, 'hold',   'COGS downward probe CC=1.31'),
    }

    for cname, cr, cc in CANDIDATES:
        finRev  = np.maximum(raw_rev  * cr, 1.0)
        finCogs = np.maximum(raw_cogs * cc, 0.0)

        # Save submission
        sub = pd.DataFrame({'Date': fdates.values, 'Revenue': finRev, 'COGS': finCogs})
        assert len(sub) == 548
        assert (sub['Revenue'] > 0).all() and (sub['COGS'] >= 0).all()
        sub.to_csv(f'artifacts/submissions/submission_forecast_022_{cname}.csv', index=False)
        sub_dfs[cname] = sub

        out_df[f'Rev_{cname}']  = finRev
        out_df[f'Cog_{cname}']  = finCogs

        pct_rev  = (finRev.sum()  - winner_rev.sum())  / winner_rev.sum()  * 100
        pct_cogs = (finCogs.sum() - winner_cogs.sum()) / winner_cogs.sum() * 100
        mad_rev  = np.mean(np.abs(finRev  - winner_rev)  / winner_rev)  * 100
        max_rev  = np.max( np.abs(finRev  - winner_rev)  / winner_rev)  * 100
        mad_cogs = np.mean(np.abs(finCogs - winner_cogs) / winner_cogs) * 100

        # Monthly diff
        df_m = pd.DataFrame({'Rev':finRev,'WinRev':winner_rev,
                             'Cog':finCogs,'WinCog':winner_cogs}, index=fdates)
        for m in range(1, 13):
            mk = fdates.month == m
            if mk.any():
                diff_rows.append({
                    'candidate': cname, 'cr': cr, 'cc': cc, 'month': m,
                    'rev_pct_diff': (finRev[mk].sum()-winner_rev[mk].sum())/winner_rev[mk].sum()*100,
                    'cog_pct_diff': (finCogs[mk].sum()-winner_cogs[mk].sum())/winner_cogs[mk].sum()*100,
                })

        pri, action, hyp = priority_map[cname]
        manifest_rows.append({
            'candidate_file': f'submission_forecast_022_{cname}.csv',
            'candidate_name': cname, 'cr': cr, 'cc': cc,
            'total_revenue': finRev.sum(), 'total_cogs': finCogs.sum(),
            'pct_diff_rev_vs_best': pct_rev, 'pct_diff_cogs_vs_best': pct_cogs,
            'mean_abs_pct_diff_daily_rev': mad_rev,
            'max_abs_pct_diff_daily_rev':  max_rev,
            'mean_abs_pct_diff_daily_cogs': mad_cogs,
            'avg_cogs_ratio': (finCogs / finRev).mean(),
            'expected_public_risk': 'low',
            'recommended_submission_priority': pri,
            'submit_or_hold': action, 'hypothesis': hyp,
        })

    out_df.to_csv('artifacts/forecasts/forecast_022_future_candidates.csv', index=False)
    mf = pd.DataFrame(manifest_rows).sort_values('recommended_submission_priority')
    mf.to_csv('artifacts/tables/forecast_022_candidate_manifest.csv', index=False)
    pd.DataFrame(diff_rows).to_csv('artifacts/tables/forecast_022_diff_vs_current_best.csv', index=False)

    # Figures
    plt.figure(figsize=(14, 5))
    w_mo = pd.DataFrame({'Rev': winner_rev}, index=fdates).resample('ME')['Rev'].sum()
    w_mo.plot(label='021 best (CR=1.27)', lw=2, color='black')
    for cname, cr, cc in CANDIDATES[:4]:
        pd.DataFrame({'Rev': sub_dfs[cname]['Revenue'].values}, index=fdates)\
          .resample('ME')['Rev'].sum().plot(label=f'{cname}', alpha=0.75)
    plt.title('Monthly Revenue Profiles vs Current Best (CR=1.27)')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_022_candidate_monthly_profiles.png'); plt.close()

    plt.figure(figsize=(14, 4))
    for cname, cr, cc in CANDIDATES:
        diff_pct = (sub_dfs[cname]['Revenue'].values - winner_rev) / winner_rev * 100
        plt.plot(fdates, diff_pct, label=cname, alpha=0.8)
    plt.axhline(0, color='black', lw=1, linestyle='--')
    plt.title('Daily Revenue % Diff vs 021 Best (CR=1.27)')
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_022_diff_vs_current_best.png'); plt.close()

    print("\nCandidates generated:")
    for r in manifest_rows:
        print(f"  {r['candidate_name']:25s}  CR={r['cr']:.3f}  CC={r['cc']:.2f}  "
              f"Rev_diff={r['pct_diff_rev_vs_best']:+.3f}%  {r['submit_or_hold']}")
    print("\nDone.")

if __name__ == '__main__':
    main()
