# FORECAST-019: Component Feature Store Reset and Forensics

### 1. Objective and Rationale
After exhausting pure target-modeling and heuristic-adjustment strategies (e.g. Q2 adjustments, Lunar shifts), we rebuilt the forecasting pipeline from the fundamental components of the revenue equation. Instead of predicting macroscopic `Revenue` directly, we modeled underlying behavioral drivers: `orders`, `units`, `revenue_per_order`, and `revenue_per_unit`.

### 2. Daily Component Feature Store Validation
We successfully reconstructed the core metrics by rolling up the transactional grain from `order_items`, `orders`, and `products`. 
- **Validation**: The daily summation of `quantity * unit_price` identically matched `sales.csv` Revenue (Max Diff = `0.0000`). 
- **Output Artifact**: `forecast_019_daily_component_feature_store.csv` now provides model-ready, leakage-safe daily metrics without explosion or nulls.

### 3. Component Diagnostics
By tracking `revenue_per_order` and volume metrics, we observed:
- A strong shift in the 2020-2022 operating regime: Daily volume (`orders`) dictates macro volatility much more intensely than value (`revenue_per_order`), which remains surprisingly stable inside its regime.
- This tells us that modeling raw transactional *volume* separately from transactional *value* allows tree models (HGB) to fit nonlinear shapes better than when tracking the multiplied product (Revenue) directly.

### 4. Residual Forensics vs 012
Comparing the direct baseline against our new component models:
- The baseline model (similar to the 012 logic) struggles with edge-of-month and intra-week allocation fidelity.
- The `monthly_total_daily_shape` structure forces the daily spikes to normalize down to the historically accurate shape curve, removing pathological daily overshoots without breaking the monthly level.

### 5. Future Candidates
We generated four new submission candidates representing the component modeling strategies:
1. **`component_base`**: Predicts `orders` (HGB) * `revenue_per_order` (Ridge). De-risks extreme edge cases.
2. **`monthly_total_daily_shape`**: Predicts Monthly Gross * Daily Intra-month share. Strongly stabilizes variance.
3. **`component_blend012_25`**: 25% Component Base + 75% 012 Calibrated.
4. **`component_blend012_50`**: 50% Component Base + 50% 012 Calibrated.

### 6. Submission Recommendation and Rules
If permitted to submit purely model-driven enhancements, the **`component_blend012_50`** model is strongly recommended. It incorporates the improved stability of the behavioral component breakdown (volume x value) while bridging to the proven public-leaderboard shape of `012`. 

**Leakage Checklist & Risk Assessment**:
- [x] No `sample_submission.csv` data fed into training loops.
- [x] COGS fully constrained mathematically.
- [x] Strict time-based chronologic validations (Folds 1-4) were used, no random splits.
