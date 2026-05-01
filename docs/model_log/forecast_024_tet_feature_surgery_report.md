# FORECAST-024: Tết Feature Surgery Report

## Current Best / Recent Failed

- Current best public score: **submission_forecast_022_cr128_cc132.csv — 706,621.35616**
- Recently failed: submission_forecast_023_ref_adaptive_qalpha_cr128_cc132.csv — 708,392.43742
- Interpretation: Adaptive alpha failed on public LB; focus shifts to Tết feature surgery.

## Why Tết Surgery

FORECAST-023 feature ablation found **dropping Tết features improved FoldA MAE materially**. This suggests misalignment/double-counting vs month/Fourier/promo windows or overfitting around Q1.

## Tết Historical Audit (Sales)

Audit recomputed `tet_days_diff` exactly as in FORECAST-020 and summarized Revenue/COGS within ±45 days. See `artifacts/tables/forecast_024_tet_feature_audit.csv` and `artifacts/figures/forecast_024_tet_window_historical_effect.png`.

## Ablation Metrics (3 folds, reference architecture, CR=1.28 CC=1.32)

| variant | Rev_MAE | COGS_MAE | Combined_MAE | Q1_Combined_MAE | TetWin_Combined_MAE | Q1_NonTet_Combined_MAE |
| --- | --- | --- | --- | --- | --- | --- |
| no_tet_all | 1,252,144.5 | 1,006,852.5 | 2,258,997.0 | 1,483,850.9 | 1,526,142.1 | 1,441,559.7 |
| soft_tet_only | 1,236,581.6 | 1,023,938.9 | 2,260,520.5 | 1,526,127.0 | 1,585,475.8 | 1,466,778.1 |
| post_tet_only | 1,252,422.6 | 1,051,369.0 | 2,303,791.6 | 1,541,308.9 | 1,595,721.4 | 1,486,896.3 |
| full_reference | 1,282,708.1 | 1,126,972.8 | 2,409,680.9 | 1,477,210.1 | 1,467,221.0 | 1,487,199.2 |


FoldA spotlight: full_reference Rev MAE=793,847 vs no_tet_all Rev MAE=833,004 (**-4.93%**).

## Candidates (future horizon)

| candidate_name | OOF_Combined_MAE | OOF_Q1_Combined_MAE | OOF_TetWin_Combined_MAE | pct_diff_revenue_vs_current_best | mean_abs_pct_diff_daily_revenue_vs_best | expected_public_risk | submit_or_hold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_tet_cr128_cc132 | 2,258,997.021 | 1,483,850.905 | 1,526,142.131 | -2.488 | 4.577 | high | hold |
| no_tet_cr127_cc132 | 2,258,997.021 | 1,483,850.905 | 1,526,142.131 | -3.250 | 4.811 | high | hold |
| no_tet_cr129_cc132 | 2,258,997.021 | 1,483,850.905 | 1,526,142.131 | -1.727 | 4.468 | high | hold |
| soft_tet_cr128_cc132 | 2,260,520.467 | 1,526,126.967 | 1,585,475.825 | -2.228 | 4.562 | high | hold |
| post_tet_only_cr128_cc132 | 2,303,791.610 | 1,541,308.854 | 1,595,721.447 | -1.043 | 4.486 | high | hold |
| no_tet_blend_current25 | 2,372,009.967 | 1,478,870.291 | 1,481,951.292 | -0.622 | 1.144 | medium | submit |


## Recommendation

Recommend submitting **submission_forecast_024_no_tet_blend_current25.csv** first.

## Risks

- If `no_tet_*` shifts totals > ±2% or daily diffs are large, prefer the blend candidate as a safer perturbation.
- `post_tet_only` may underfit if the true effect is partly anticipatory/pre-Tết.

## Leakage Checklist

- [x] No sample_submission Revenue/COGS used (Date-only).
- [x] No external data used.
- [x] All future features are Date/schedule-derived (no lags).
- [x] All folds train-before-validation.
- [x] Current best file unchanged; only read.
- [x] All submissions: 548 rows, Date order == sample_submission, Revenue>0, COGS>=0.
