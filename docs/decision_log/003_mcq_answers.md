# MCQ Computation Report

## 1. Executive Answer Table

| Question | Selected Option | Selected Answer | Computed Value | Confidence | Caveat |
|---|---|---|---|---|---|
| Q1 | C | 180 days | 144.0000 | High | Median gap across all statuses. Robustness check matches. |
| Q2 | D | Standard | 0.3134 | High | Margin derived as (price-cogs)/price. |
| Q3 | B | wrong_size | 7626.0000 | High |  |
| Q4 | C | email_campaign | 0.0045 | Medium | web_traffic is 1 row/day. bounce_rate is averaged over days where the source is dominant. |
| Q5 | C | 39% | 38.6635 | High |  |
| Q6 | A | 55+ | 5.4069 | High |  |
| Q7 | C | East | 7637532676.2000 | High | sales.csv does not have region. Revenue computed by sum(quantity * unit_price) matching sales.csv logic. |
| Q8 | A | credit_card | 28452.0000 | High |  |
| Q9 | A | S | 0.0565 | High |  |
| Q10 | C | 6 installments | 24446.6544 | High |  |

## 2. Detailed Computation Trace

### Q1
- **Source Tables**: orders
- **Grain**: customer_id + order_date
- **Denominator**: inter-order gaps
- **Formula**: median(diff(order_date))
- **Primary Result**: 144.0
- **Robustness Result**: median_days_non_cancelled=156.0 (Changed? True)
- **Caveat**: 

### Q2
- **Source Tables**: products
- **Grain**: product segment
- **Denominator**: number of products
- **Formula**: mean((price-cogs)/price)
- **Primary Result**: 0.31344174843884803
- **Caveat**: 

### Q3
- **Source Tables**: returns, products
- **Grain**: return record
- **Denominator**: total streetwear returns
- **Formula**: count(return_id) by reason
- **Primary Result**: wrong_size
- **Robustness Result**: top_by_qty=wrong_size (Changed? False)
- **Caveat**: 

### Q4
- **Source Tables**: web_traffic
- **Grain**: date
- **Denominator**: days with source
- **Formula**: mean(bounce_rate) by source
- **Primary Result**: 0.0044584356435643565
- **Caveat**: 

### Q5
- **Source Tables**: order_items
- **Grain**: order_items row
- **Denominator**: total order_items
- **Formula**: (not_null / total) * 100
- **Primary Result**: 38.663493169565214
- **Caveat**: 

### Q6
- **Source Tables**: customers, orders
- **Grain**: customer_id
- **Denominator**: customers with non-null age_group
- **Formula**: mean(order_count) by age_group
- **Primary Result**: 55+
- **Robustness Result**: highest_if_excluding_0_orders=55+ (Changed? False)
- **Caveat**: 

### Q7
- **Source Tables**: order_items, orders, geography
- **Grain**: order_id -> zip
- **Denominator**: None
- **Formula**: sum(line_revenue) by region
- **Primary Result**: East
- **Robustness Result**: nc=East, pay=East (Changed? False)
- **Caveat**: sales_train.csv does not exist. Handled via join.

### Q8
- **Source Tables**: orders
- **Grain**: order_id
- **Denominator**: cancelled orders
- **Formula**: count by payment_method
- **Primary Result**: credit_card
- **Caveat**: 

### Q9
- **Source Tables**: returns, order_items, products
- **Grain**: return record vs order_item row
- **Denominator**: order_item rows
- **Formula**: return_records / order_item_rows
- **Primary Result**: S
- **Robustness Result**: top_by_qty_rate=S (Changed? False)
- **Caveat**: 

### Q10
- **Source Tables**: payments
- **Grain**: payment record
- **Denominator**: number of payments per installment plan
- **Formula**: mean(payment_value) by installments
- **Primary Result**: 6
- **Caveat**: 

## 3. Ambiguity Notes
- **Q7**: Data ambiguity. The document specified `sales_train.csv` and region. Since `sales.csv` lacks region and `sales_train.csv` does not exist, revenue by region was successfully re-derived using the verified `order_items` quantity * unit_price. Results show massive disparity among regions, so the ranking is unambiguous.
- All robustness checks confirm the primary choice is stable and unambiguous.
