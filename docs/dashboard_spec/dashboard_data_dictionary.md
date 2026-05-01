# Dashboard Data Dictionary

## 1. dashboard_revenue_quality.csv
- **Dashboard Page**: Executive Revenue Quality.
- **Grain**: Year.
- **Row Meaning**: Tổng hợp KPIs tài chính và vận hành cho từng năm.
- **Key Columns**: year, revenue, cogs, gp, orders, units, margin.
- **Caveats**: Không có.

## 2. dashboard_promo_margin.csv
- **Dashboard Page**: Promo Margin Leakage.
- **Grain**: Category x Year.
- **Row Meaning**: Sự so sánh hiệu suất giữa các đơn hàng có Promo và Không Promo của một danh mục trong một năm.
- **Key Columns**: category, year, margin_promo, margin_nopromo, margin_gap, diagnostic_gp_gap.
- **Caveats**: diagnostic_gp_gap là mức ước tính tối đa, không nhân quả.

## 3. dashboard_category_portfolio.csv
- **Dashboard Page**: Category & Product Portfolio.
- **Grain**: Category x Segment.
- **Row Meaning**: Phân loại vai trò danh mục trong tổng thể.
- **Key Columns**: portfolio_role, revenue_share, margin.
- **Caveats**: CAGR tính từ 2013-2022.

## 4. dashboard_inventory_mismatch.csv
- **Dashboard Page**: Inventory Demand-Fit.
- **Grain**: Product x Month.
- **Row Meaning**: Trạng thái lỗi tồn kho trong tháng.
- **Key Columns**: overlap, stock_retail_value.
- **Caveats**: Dựa trên snapshot cuối tháng.

## 5. dashboard_customer_lifecycle.csv
- **Dashboard Page**: Customer Lifecycle.
- **Grain**: Customer.
- **Row Meaning**: Phân khúc vòng đời và RFM của một khách hàng.
- **Key Columns**: customer_id, rfm_segment, recency, frequency, monetary.
- **Caveats**: Recency tính bằng ngày, RFM chỉ dùng line_revenue từ các đơn thành công.

## 6. dashboard_product_actions.csv
- **Dashboard Page**: Category & Product Portfolio.
- **Grain**: Product.
- **Row Meaning**: Chỉ định hành động dựa trên hiệu suất của SKU.
- **Key Columns**: classification, action_candidate, return_rate.
- **Caveats**: Bỏ qua tác động của biến thể (size/color).
