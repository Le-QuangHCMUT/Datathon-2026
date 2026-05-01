# Tableau Prep Candidate Outputs

| Output Table | Intended Grain | Required Source Tables | Key Joins | Metrics | Dashboard Use | Risks & Validation |
|---|---|---|---|---|---|---|
| **dim_products** | 1 row / product_id | `products`, `inventory` (agg) | None (base table) | cogs, price, avg_margin | Product Analysis (D3) | Low risk. Base dimension. |
| **dim_geography** | 1 row / zip | `geography` | None (base table) | N/A | Spatial Maps | Low risk. Base dimension. |
| **dim_promotions** | 1 row / promo_id | `promotions` | None (base table) | discount_value | Marketing (D4) | Low risk. Base dimension. |
| **fact_orders_enriched** | 1 row / order_id | `orders`, `payments`, `shipments`, `customers`, `geography` | `orders` + `payments` (1:1), + `shipments` (1:1), + `customers` (N:1), + `geography` (N:1) | payment_value, shipping_fee, lead_time | Revenue, Ops (D1, D5) | High risk if `payments` or `shipments` are not strictly 1:1. Verify uniqueness first. |
| **fact_order_items_enriched** | 1 row / order_id + product_id | `order_items`, `products`, `orders` | `order_items` + `products` (N:1), + `orders` (N:1) | quantity, unit_price, discount_amount | Revenue, Product (D1, D3) | Do not join `payments` here. Will inflate `payment_value`. |
| **fact_returns_enriched** | 1 row / return_id | `returns`, `products` | `returns` + `products` (N:1) | return_quantity, refund_amount | Product, Quality (D3) | Do not join to `order_items` directly. Aggregate returns to `order_id` + `product_id` grain before joining. |
| **dim_customers_rfm** | 1 row / customer_id | `customers`, `orders`, `payments` | `customers` + AGG(`orders` + `payments`) | Recency, Frequency, Monetary | Segmentation (D2) | Must aggregate `orders` and `payments` to `customer_id` grain first. |
| **fact_shipments_enriched** | 1 row / order_id | `shipments`, `orders`, `geography` | `shipments` + `orders` (1:1) + `geography` (N:1) | delivery_time, shipping_fee | Ops (D5) | Low risk if `order_id` is 1:1. |
| **fact_sales_daily** | 1 row / date | `sales` | None | Revenue, COGS | Forecasting, Macro Trends (D1) | Do not sum line-items to match this unless verified. They may not match perfectly. |
| **fact_inventory** | 1 row / product_id + month | `inventory`, `products` | `inventory` + `products` (N:1) | stock_on_hand, fill_rate, stockout_days | Supply Chain (D5) | Grain is monthly snapshots. Do not join to daily sales without care. |
| **fact_web_traffic** | 1 row / date | `web_traffic` | None | sessions, page_views, bounce_rate | Marketing (D4) | Ensure dates align with `sales` dates. |
| **agg_cohort_retention** | 1 row / cohort_month + active_month | `orders`, `customers` | N/A (Aggregated) | retention_rate, active_customers | Lifecycle (D2) | Custom SQL/Python aggregation required. |
| **agg_reviews_summary** | 1 row / product_id | `reviews` | N/A (Aggregated) | avg_rating, review_count | Product Quality (D3) | Aggregate before joining to `products`. |
