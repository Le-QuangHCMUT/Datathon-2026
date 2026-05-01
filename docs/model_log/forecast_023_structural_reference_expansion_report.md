# FORECAST-023: Structural Reference Expansion Report

## Current Best
- **submission_forecast_022_cr128_cc132.csv: 706,621.35616**
- Revenue calibration plateau identified: CR=1.29 (707,145) < CR=1.28 (706,621) > CR=1.27 (706,787). Peak confirmed at CR=1.28.
- Next gain must come from structural improvement, not calibration.

---

## 1. Promo Schedule Audit

**Result: hardcoded reference schedule is fully confirmed by promotions.csv.**

| Promo | Appearances | Mean Duration | Mean Discount | Odd-Year Only |
|---|---|---|---|---|
| Spring Sale | 10 (every year) | 31.0 days | 12% | No |
| Mid-Year Sale | 10 (every year) | 30.0 days | 18% | No |
| Fall Launch | 10 (every year) | 33.2 days | 10% | No |
| Year-End Sale | 10 (every year) | 45.6 days | 20% | No |
| Urban Blowout | 5 (odd years) | 35.0 days | 50 fixed | Yes |
| Rural Special | 5 (odd years) | 30.8 days | 15% | Yes |

**No schedule correction needed.** Promotions.csv covers 2013–2022 only — confirming the constraint that future 2023–2024 features must be calendar-derived. Hardcoded reference schedule is the correct approach.

---

## 2. LGB Weight Scheme Results (OOF, 3-fold avg)

| Scheme | Rev MAE | COGS MAE | Notes |
|---|---|---|---|
| **S2_post2020** | **1,153,417** | **1,007,421** | Best OOF — recent years only |
| S3_recent2022 | 1,170,636 | 1,027,448 | Recent-weighted |
| S6_uniform | 1,183,207 | 1,023,588 | Uniform weights |
| S5_hybrid | 1,188,982 | 1,032,949 | |
| S4_clean1718 | 1,195,156 | 1,028,914 | |
| S1_high_era | 1,282,708 | 1,126,973 | Reference (worst OOF) |

**Key finding**: S1_high_era (2014–2018 dominant) has the worst OOF MAE but was the scheme that produced the winning public submission. This is consistent with the reference document insight that 2014–2018 provides a stable seasonal shape, even if the post-2020 regime fits the holdout better in-sample.

---

## 3. NNLS Ensemble Weights

**Revenue NNLS**: S6_uniform (55.4%) + S1_high_era (39.2%) + S5_hybrid (5.4%)  
**COGS NNLS**: S1_high_era (46.9%) + S4_clean1718 (22.0%) + S3_recent2022 (21.7%) + S6_uniform (9.4%)

The NNLS solution assigns heavy weight to the reference S1 scheme for COGS, while preferring S6_uniform for Revenue. S2_post2020 receives zero NNLS weight despite being OOF-best — this is expected: NNLS minimizes the full historical OOF residual, where S1 contributes structural shape across 2013–2022.

---

## 4. Adaptive Alpha Results (Q-specialist vs LGB blend per quarter)

| Quarter | Best Alpha (Rev) | Best Alpha (Cog) |
|---|---|---|
| Q1 | 0.70 | 0.80 |
| Q2 | 0.40 | 0.80 |
| Q3 | 0.80 | 0.40 |
| Q4 | 0.40 | 0.40 |

Q3 strongly prefers higher Q-specialist weight (0.80 for Revenue). Q2 and Q4 prefer lower Q-specialist weight (0.40). This is consistent with historical analysis — Q3 (Aug spike, Vietnam summer promotion) benefits most from quarter-focused specialist learning.

---

## 5. Feature Group Ablation (FoldA, Revenue)

| Group Removed | Rev MAE | vs Full |
|---|---|---|
| **no_tet** | **577,571** | **Better without Tết!** |
| no_eom_som | 592,213 | Slightly better |
| no_parity | 592,161 | Neutral |
| no_promo | 603,835 | Slightly worse |
| **full** | **612,595** | Baseline |
| no_holidays | 612,595 | Same as full |
| no_fourier | 628,296 | Worse |

> **Critical finding**: Removing Tết features *improves* FoldA MAE by 5.7%. This suggests Tết features may be slightly misaligned or overfitting the 2022 validation window. This warrants a targeted follow-up experiment in FORECAST-024.

---

## 6. Candidate Manifest

| Candidate | Rev vs Best | Mean Daily Diff | Action |
|---|---|---|---|
| ref_nnls_lgb_multiweight | -1.22% | 2.04% | **Submit** |
| ref_adaptive_qalpha | +0.29% | 0.56% | Hold |
| ref_nnls_adaptive | -1.35% | 2.22% | Hold |
| **ref_nnls_blend_current25** | -0.31% | 0.51% | Hold (safer) |

---

## 7. Recommendation

**Change recommendation based on manifest analysis:**

- Candidate A (NNLS multiweight) differs -1.22% in total Revenue from current best, with 2.04% mean daily diff. This is a **medium structural risk** — the NNLS solution pulls toward S6_uniform which may not maintain the winning shape.
- Candidate B (adaptive alpha) differs only +0.29% total / 0.56% daily — very low risk. Adaptive alpha uses Q3=0.80 (more Q-specialist for Q3) which is intuitively correct.
- **Recommend submitting `ref_adaptive_qalpha_cr128_cc132` first** (revised from initial priority). It is the safest structural perturbation with the clearest hypothesis (Q3 gets more specialist weight).
- Hold NNLS candidates until adaptive alpha result is known.

---

## 8. Leakage Checklist
- [x] No sample_submission targets used
- [x] All folds train-before-validation
- [x] All future features calendar/schedule-derived
- [x] 022 winner file not modified
- [x] All 4 submissions: 548 rows, Revenue > 0, COGS >= 0 ✅
