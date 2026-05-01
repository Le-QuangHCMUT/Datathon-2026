# Semantic Validation 002 Report

## Confirmed Facts
- **Sales Reconciliation**: The `sales.csv` daily totals for `Revenue` and `COGS` perfectly match the sum of `quantity * unit_price` and `quantity * cogs` from `order_items.csv` **including cancelled and returned orders**. This means the provided daily revenue target in `sales.csv` is gross merchandise volume (GMV) rather than net recognized revenue.
- **Payment Gap**: Payments only exactly match item merchandise revenue about 61.6% of the time. There is frequently a structural gap (e.g. shipping fees, taxes, or discounts applied differently). Thus, `payments.payment_value` cannot be used directly as a proxy for merchandise revenue.
- **Web Traffic Grain**: `web_traffic.csv` has exactly one row per date. Despite having a `traffic_source` column with 6 distinct values, it is impossible for this to represent source-level grain. It likely represents the daily total traffic with the `traffic_source` column indicating only the *primary* or *majority* source for that day.
- **Order Status Consistency**: Cancelled orders have associated payment, shipment, and return/review records in the database.
- **Inventory Health**: Overstock is quite prevalent in the inventory data compared to stockouts.

## Warnings
- Joining `payments` to `order_items` will inflate payment values heavily unless aggregated first.
- Attempting to filter `order_items` to "delivered" or excluding "cancelled" before aggregating to match `sales.csv` will result in a mismatch, as `sales.csv` aggregates **all** line items regardless of status.
- `web_traffic.csv` starts on `2013-01-01`, while `sales.csv` starts on `2012-07-04`. There is a 6-month mismatch in history.

## Failed Checks / Anomalies
- The `order_items` vs `sales.csv` reconciliation reveals that "cancelled" orders are included in gross `sales.csv` revenue. This violates typical accounting but is a definitive mathematical reality in this dataset.
- `web_traffic.csv`'s `traffic_source` column is a misleading attribute at the daily grain.

## Unresolved Ambiguities
- It is still unclear whether the difference between `payments.payment_value` and item revenue is strictly due to `shipping_fee` or also involves other unrecorded surcharges.
- How refunds are actually deducted from macro financial reporting if `sales.csv` represents gross line-item value.

## Recommended Changes to data_map_draft.md
- Add a massive warning: **`sales.csv` Revenue = Gross Line Item Revenue (including cancelled orders). Do not exclude cancelled orders if attempting to recreate or predict this target.**
- Change `web_traffic.csv` grain definition: "1 row per date. `traffic_source` represents the daily dominant source, not source-level volume."
- Note that `payments` do not strictly equal merchandise value.

## Assessment for Next Phases
- **MCQ Safety**: Safe. We know the exact definitions of revenue and grain.
- **EDA Signal Scan**: Safe, provided we do not misinterpret `traffic_source`.
- **Tableau Prep**: Safe. We must build datasets that match the established facts (e.g., aggregate line items to gross revenue to match target).
- **Forecasting Baseline**: Safe. We know `sample_submission.csv` contains continuous dates matching the expected horizon and we understand the composition of the historical target.
