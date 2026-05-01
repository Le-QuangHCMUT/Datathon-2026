# Post-LB Forecast Patch Report
## Sprint: FORECAST-013

### 1. Public Score Update and Iteration Goal
The `submission_forecast_012_calibrated.csv` candidate scored **1,067,034.48** on the public leaderboard. This is an improvement of roughly 20.6% over the previous baseline (`1.34M`), validating the training-slide alignments.
Our objective in this iteration was to:
- Sweep calibration factors around the empirical CV value (0.8336) to extract more margin.
- Fix a critical bug causing the `QSpecialist` predictions to output implausibly low values.
- Apply a controlled error correction for Q2.

### 2. Issues Fixed from FORECAST-012
- **Report Wording Error**: The previous report incorrectly framed the baseline model as "under-forecasting". Given the calibration factor was `0.8336` (meaning predicted was multiplied down), the base ensemble was in fact **over-forecasting** relative to reality on CV.
- **selected_flag Error**: The flags were updated in the pipeline scripts so that `M6_Ensemble_calibrated` could be correctly selected.
- **Q-Specialist Scale Bug**: In FORECAST-012, `future_feat` generation initialized the `time_index` field at 0 (effectively resetting the year to 2012). This caused the Ridge model underlying `QSpecialist` to infer early startup volume levels, predicting ~19k Revenue for early 2023. We fixed this by correctly offsetting `time_index` by the size of the historical `sales` dataset during future feature generation. 

### 3. Calibration Sweep Result
A sweep over calibration factors (0.76 to 0.92) across all 4 historical backtest CV folds was executed.
- The minimum total CV MAE was located around `0.81` - `0.83`.
- `0.81` pushed the predictions slightly lower than our `0.8336` baseline, reflecting a more aggressive dampening of over-forecasting behavior.
- `0.86` represents a highly conservative buffer (closer to 1.0) in case the 2023-2024 reality runs hotter than 2022's trajectory.
- We constructed submissions for both `0.81` and `0.86` to bracket the public LB score.

### 4. Q-Specialist Bug Diagnosis
With the `time_index` bug fixed, `QSpecialist` output returned to an expected nominal scale (e.g., millions). The quarterly targeted Ridge models now correctly project 2023/2024 seasonal amplitudes while preserving the recent historical base levels.

### 5. Q2 Correction Result
From FORECAST-012, Q2 was identified as having the largest absolute validation error.
- We tested applying an independent multiplicative adjustment exclusively for Q2 dates.
- A factor of `0.90` (reducing Q2 specific predictions by a further 10%) optimally minimized Q2 MAE without destroying global MAE.
- We generated `submission_forecast_013_q2_adjusted.csv` using the 0.8336 global base combined with the 0.90 Q2 correction.

### 6. Candidate Ranking and Next Action
| Candidate | Method | Hypothesis | Action |
|---|---|---|---|
| **submission_forecast_013_calib_081.csv** | Ensemble * 0.81 | Strongest reduction based on pure CV. | **Primary Submit** |
| **submission_forecast_013_calib_086.csv** | Ensemble * 0.86 | Moderate reduction. Acts as a safety net. | **Secondary Submit** |
| **submission_forecast_013_q2_adjusted.csv** | Calib 0.8336 + Q2 * 0.9 | Fixes structural Q2 error weakness. | Hold (unless Q2 explicitly confirmed as problem) |
| **submission_forecast_013_qspecialist_fixed.csv** | Q-Specialist Blend | Uses resolved quarterly weights. | Hold |

**Recommended Kaggle Submission Order:**
1. Submit `submission_forecast_013_calib_081.csv`. This aligns with the lowest points of the CV sweep and handles over-forecasting aggressively.
2. Submit `submission_forecast_013_calib_086.csv`. If `0.81` worsens the LB, it means 2023 levels are rebounding, and the `0.86` candidate will correct for that bounce.

### 7. Leakage Checklist and Risks
- [x] No `sample_submission.csv` target values used.
- [x] No external holiday/lunar datasets used.
- [x] Time indexes perfectly continuous from training to forecast.
- **Risk**: Global multipliers assume homogeneous drift. If actual reality has a different shape (e.g. Q4 spikes harder than normal), a static multiplier penalizes Q4 inappropriately. The Q-specialist or Q2-adjusted approaches exist to mitigate this if public LB tests expose heterogeneous regime shapes.
