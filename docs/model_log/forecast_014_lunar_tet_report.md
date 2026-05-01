# FORECAST-014: Lunar Tết Features and Safe Recalibration

### 1. Failure Analysis of FORECAST-013 Q2 Adjustment
In FORECAST-013, an explicit `0.9` reduction was applied strictly to Q2 future predictions to minimize the historical CV error for that quarter. This candidate heavily under-performed on the public LB (scoring `1,787,131`). This failure mode signals that the out-of-sample (2023-2024) Q2 structural reality does not share the same systematic weakness or level-shift as the historical 2020-2022 validation windows. Consequently, time-bounded explicit multipliers (like Q2 adjustment) overfit validation drastically, establishing the current `012_calibrated` global strategy as the safer structural foundation.

### 2. Lunar Feature Rule Clarification
Following the project clarification, lunar holiday features are permitted only when explicitly mapped or derived directly from the intrinsic Date sequences without internet scraping or joining external holiday datasets. Since `lunardate` is unavailable in the environment, we integrated a compact hardcoded algorithmic dictionary to map the `Gregorian Date` index to Lunar Tết distances for 2012-2024 deterministically. 

### 3. Implementation of Lunar/Tết Derived Features
The pipeline was refactored to compute explicit distance arrays to the nearest Lunar New Year:
- `tet_days_diff` / `abs_tet_days_diff`
- Windowed booleans: `tet_in_3`, `tet_in_7`, `tet_in_14`
- Structural epochs: `pre_tet_14`, `post_tet_14`, `post_tet_30`
- Bucketized epochs: `pre_30_to_15`, `pre_14_to_8`, `pre_7_to_1`, `tet_0_to_3`, etc.
This allows the tree models and seasonal shapes to capture explicit holiday surges directly.

### 4. Lunar Feature Audit
The audit successfully confirmed exact alignments matching known Tết Gregorian windows (e.g. Feb 1st 2022, Jan 22nd 2023).
The plot `forecast_014_tet_effect_audit.png` confirms that the explicit mapping cleanly isolates the sharp pre-Tết volume build-up and the subsequent post-Tết trough historically.

### 5. Model Comparison
Using the true lunar mappings, the `A_HGB_lunar` algorithm overtook the baseline, resulting in a new optimal Lunar Ensemble weighting:
- **~53.6% HGB_lunar**
- **~46.4% SeasonalProfile_lunar_adjusted**
This ensemble demonstrates improved mapping over the proxy-based approach.

### 6. Calibration Result
Since `0.8336` was the previously validated anchor for our best LB score (`1.06M`), we performed a sweep between `0.80` and `0.875` on the newly formulated Lunar Ensemble.
- **lunar_cal083**: Retaining `0.8336` calibrates the new lunar base effectively.
- **lunar_cal086**: Slightly higher, providing a conservative safety net since further down-scaling (as seen in Q2 adjustments) proved catastrophic.

### 7. Candidate Ranking and Recommendation
| Candidate | Method | Hypothesis | Action |
|---|---|---|---|
| **submission_forecast_014_lunar_cal083.csv** | Lunar Ensemble * 0.8336 | Aligns the exact LB-winning calibration offset but employs true Lunar calendar structural features. | **Primary Submit** |
| **submission_forecast_014_lunar_blend012.csv** | 50% Lunar_083 + 50% 012_Calibrated | A low-risk incremental bridge from the current LB anchor. Protects against excessive shape disruption. | **Secondary Submit** |
| **submission_forecast_014_lunar_cal086.csv** | Lunar Ensemble * 0.86 | Conservative safety net. | Hold |
| **submission_forecast_014_lunar_base.csv** | Uncalibrated Lunar Ensemble | Baseline comparison. | Hold |

**Submission Priority:**
Submit `submission_forecast_014_lunar_cal083.csv` to capture the pure uplift from structural Tết realignment while anchoring on the proven calibration level. If it performs worse or similarly, submit `lunar_blend012.csv` for an optimal smoothed prediction.

### 8. Leakage Checklist and Risks
- [x] No `sample_submission.csv` target values used.
- [x] No external downloaded datasets mapped in.
- [x] Time indexes perfectly continuous.
- [x] Folds use strict chronological train-before-validation splits.
- **Risks**: The `0.8336` level multiplier represents a global static shift. Should underlying drift shift back upwards beyond the 2020-2022 mean, it will start systematically under-forecasting.
