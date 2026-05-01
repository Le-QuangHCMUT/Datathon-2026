# Schema Summary

Please refer to the machine-readable diagnostic file located at: `artifacts/diagnostics/schema_summary.csv` for the full schema summary.

## Highlights
- **Dates**: Present in `sales.csv`, `orders.csv`, `shipments.csv`, `returns.csv`, `reviews.csv`, `promotions.csv`, `inventory.csv`, `web_traffic.csv`, `customers.csv`. 
- **Keys**: 
  - `customer_id` is unique in `customers.csv`.
  - `product_id` is unique in `products.csv`.
  - `zip` is unique in `geography.csv`.
  - `order_id` is unique in `orders.csv`, `payments.csv`, `shipments.csv`.
  - `order_id` is **not** unique in `order_items.csv` (line-item grain), `returns.csv`, and `reviews.csv`.
  - `promo_id` is unique in `promotions.csv`.
  - `Date` is unique in `sales.csv` and `web_traffic.csv`.
  - `review_id` is unique in `reviews.csv`.
  - `return_id` is unique in `returns.csv`.
