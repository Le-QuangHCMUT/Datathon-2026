# Tableau Build Guide v1

## General Warnings
- **Tuyệt đối KHÔNG JOIN** `payments` với `order_items` để tính Revenue.
- **Tuyệt đối KHÔNG DÙNG** `web_traffic.traffic_source` làm cơ sở phân bổ attribution.
- **Tuyệt đối KHÔNG DÙNG** `sample_submission` làm truth table cho Revenue/COGS.
- **LUÔN SỬ DỤNG** các file CSV trong thư mục `artifacts/dashboard_data/` để xây dựng dashboard nhằm tránh sai sót về grain.

## A. Executive Revenue Quality
- **Source**: `dashboard_revenue_quality.csv`
- **Fields Needed**: Year, Revenue, Orders, Units.
- **Calculated Fields**:
  - `Volume Effect` = Dựng sẵn trong EDA, có thể import trực tiếp.
  - `Price/Mix Effect` = Dựng sẵn.
- **Chart Type**: Waterfall.
- **Mistake to Avoid**: Tránh nhầm lẫn giữa Gross Revenue và Net Revenue. File đã xử lý Gross Line Item.

## B. Promo Margin Leakage
- **Source**: `dashboard_promo_margin.csv`
- **Fields Needed**: Category, Year, margin_gap, diagnostic_gp_gap.
- **Chart Type**: Highlight Table / Heatmap.
- **Mistake to Avoid**: Không cộng dồn (SUM) margin, luôn dùng AVG hoặc tính lại bằng SUM(GP)/SUM(Revenue).

## C. Category Portfolio
- **Source**: `dashboard_category_portfolio.csv` & `dashboard_product_actions.csv`
- **Fields Needed**: Margin, Revenue Share, CAGR, Return Rate, Classification.
- **Chart Type**: Scatter Plots (Bubble).
- **Mistake to Avoid**: Phân loại sản phẩm (Hero/Problem) phải được lọc tĩnh, không để các bộ lọc động phá vỡ logic tính median ban đầu.

## D. Inventory Mismatch
- **Source**: `dashboard_inventory_mismatch.csv`
- **Fields Needed**: Overlap, product_months, stock_retail_value.
- **Chart Type**: Bar charts.
- **Mistake to Avoid**: Không tính sum các cờ (flags) trực tiếp mà dùng cột `overlap` đã dựng sẵn.

## E. Customer Lifecycle
- **Source**: `dashboard_customer_lifecycle.csv`
- **Fields Needed**: rfm_segment, recency, frequency, monetary.
- **Chart Type**: Scatter (Recency vs Frequency), Pie/Donut (Inactive Share).
- **Mistake to Avoid**: Quên bao gồm các nhóm 'Lost', 'Registered No Orders', 'Only Cancelled'.
