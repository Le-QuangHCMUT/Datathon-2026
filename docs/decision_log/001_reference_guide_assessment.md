# Reference Guide Assessment

## 1. Blueprint Utility
The `Datathon_v2.pdf` provides an excellent structural blueprint for the EDA, Visualization, and Business Insight phases. We will adopt the proposed Tableau dashboard structure:
- **D1**: Revenue & Profitability (Macro trends, margins)
- **D2**: Customer Segmentation & Lifecycle (RFM, cohorts)
- **D3**: Product Analysis (Quality, returns, category performance)
- **D4**: Marketing & Channel Effectiveness (Web traffic, promos)
- **D5**: Operations & Supply Chain (Inventory, delivery times)
- **CEO Story**: Synthesis of D1-D5 into actionable insights.

## 2. Unverified Claims (Must Verify)
The guide contains specific numeric examples and claims that we must treat as *hypotheses* rather than facts until validated against the actual datasets:
- Claims of "negative CAGR", "margin decline", or "revenue dropping".
- Specific churn rates, lifetime value (LTV) figures, or Customer Acquisition Cost (CAC) metrics.
- Promo share percentages or ROI values.
- Stockout rates, delivery delay frequencies, or specific product return rates (e.g., claims that "DragonWear" is failing).

## 3. Risks of Blind Copying
- **Grain Mismatches**: Copying a pre-defined Tableau schema without understanding the underlying table grains (e.g., directly joining `payments` to `order_items`) will inflate revenue by the number of line items per order.
- **Data Discrepancies**: If the actual data ranges differ from the reference guide's assumptions (e.g., if promotional periods don't match the guide), the insights will be invalid.
- **Causality Assumptions**: The guide might imply causal links (e.g., "high marketing spend caused revenue to drop"). We must strictly avoid making causal claims unless definitively proven, sticking to correlation and observed trends.

## 4. Required Corrections for Pipeline
- **Aggregation before Joins**: Ensure that tables like `payments` or `shipments` (1:1 with `orders`) are joined to an aggregated version of `order_items` or vice versa, to avoid row explosion.
- **Returns & Reviews**: These must be aggregated to the `product_id` or `order_id` level before being joined to the main fact tables.
- **Data Validation**: We must validate that the `sales.csv` daily aggregates match the sum of line-item revenues before assuming they are equivalent.
