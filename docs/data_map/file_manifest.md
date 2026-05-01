# File Manifest

| File Path | Status | Size (Bytes) | Notes |
|---|---|---|---|
| sample_submission.csv | Found | 18,530 | Present in root directory. Expected format. |
| data/customers.csv | Found | 7,079,679 | Present in data directory. |
| data/geography.csv | Found | 1,402,281 | Present in data directory. |
| data/inventory.csv | Found | 5,668,678 | Present in data directory. |
| data/orders.csv | Found | 45,963,232 | Present in data directory. |
| data/order_items.csv | Found | 23,942,735 | Present in data directory. |
| data/payments.csv | Found | 18,383,288 | Present in data directory. |
| data/products.csv | Found | 195,223 | Present in data directory. |
| data/promotions.csv | Found | 4,444 | Present in data directory. |
| data/returns.csv | Found | 2,281,456 | Present in data directory. |
| data/reviews.csv | Found | 6,791,414 | Present in data directory. |
| data/sales.csv | Found | 129,634 | Present in data directory. Date range 2012-07-04 to 2022-12-31. |
| data/shipments.csv | Found | 19,756,082 | Present in data directory. |
| data/web_traffic.csv | Found | 208,722 | Present in data directory. |
| baseline.ipynb | Found | 8,608 | Baseline notebook present in root directory. |

## Discrepancies
- The official Datathon document stated 15 CSV files, including `sales_train.csv` and `sales_test.csv`.
- The actual workspace contains 14 CSV files.
- `sales_train.csv` and `sales_test.csv` do **not** exist. Instead, we have a single `sales.csv` file in the `data/` directory and `sample_submission.csv` at the root.
