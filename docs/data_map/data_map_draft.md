# Data Map Draft

## 1. Grain & Keys
- **Customers**: 1 row per `customer_id`. Unique keys: `customer_id`.
- **Products**: 1 row per `product_id`. Unique keys: `product_id`.
- **Geography**: 1 row per `zip`. Unique keys: `zip`.
- **Orders**: 1 row per `order_id`. Unique keys: `order_id`.
- **Order Items**: 1 row per `order_id` & `product_id`. `order_id` repeats for multi-item orders.
- **Payments**: 1 row per `order_id`. `order_id` is unique. 1:1 with Orders.
- **Shipments**: 1 row per `order_id`. `order_id` is unique. 1:1 with Orders.
- **Returns**: 1 row per `return_id`. `order_id` and `product_id` can repeat.
- **Reviews**: 1 row per `review_id`. `order_id` and `product_id` can repeat.
- **Promotions**: 1 row per `promo_id`. Unique keys: `promo_id`.
- **Inventory**: 1 row per `product_id` per `snapshot_date` (month-end snapshot).
- **Web Traffic**: 1 row per `date` per `traffic_source` (Wait, web_traffic `date` is unique, so it is daily aggregated across sources or only has one record per day. Let's assume daily grain overall).
- **Sales**: 1 row per `Date`. Unique keys: `Date`. (Daily continuous grain).

## 2. Table Relationships
- `orders` -> `customers` (on `customer_id`)
- `orders` -> `geography` (on `zip`)
- `order_items` -> `orders` (on `order_id`)
- `order_items` -> `products` (on `product_id`)
- `order_items` -> `promotions` (on `promo_id`, `promo_id_2`)
- `payments` -> `orders` (on `order_id`) - **WARNING**: Joining payments to `order_items` creates payment_value inflation!
- `shipments` -> `orders` (on `order_id`)
- `returns` -> `orders` (on `order_id`) & `products` (on `product_id`) - **WARNING**: Direct join to `order_items` can create one-to-many inflation unless aggregated first!
- `reviews` -> `orders` (on `order_id`) & `products` (on `product_id`) - **WARNING**: Direct join to `order_items` can create one-to-many inflation unless aggregated first!
- `inventory` -> `products` (on `product_id`)

## 3. Date Ranges
- **Sales**: 2012-07-04 to 2022-12-31 (continuous daily).
- **Forecasting Target**: `sample_submission.csv` contains dates 2023-01-01 to 2024-07-01 (continuous daily).
- **Other Tables**: Generally span from mid-2012 to end of 2022.
