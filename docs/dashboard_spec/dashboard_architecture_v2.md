# Dashboard Architecture v2

## A. Executive Revenue Quality
- **Objective**: Cung cấp bức tranh toàn cảnh về sức khoẻ doanh thu và động lực sụt giảm.
- **Audience Question**: Tại sao doanh thu của chúng ta giảm?
- **Core Claim**: Sụt giảm doanh thu hoàn toàn do mất đi khối lượng đơn hàng, không phải do giá.
- **KPI Cards**: Total Revenue, YoY Decline, Volume Effect, Price/Mix Effect.
- **Charts**: Revenue Decline Driver Bridge (Waterfall).
- **Filters**: Year.
- **Data Source**: dashboard_revenue_quality.csv.
- **Grain**: Year.
- **Calculated Fields**: Volume Effect logic, Price/Mix logic.
- **Action Map**: Đầu tư vào kênh thu hút (top funnel).
- **Caveat**: Phân tích index-based, không nhân quả.

## B. Promo Margin Leakage
- **Objective**: Làm rõ sự xói mòn lợi nhuận do các chương trình khuyến mãi.
- **Audience Question**: Khuyến mãi đang ảnh hưởng đến lợi nhuận ra sao?
- **Core Claim**: Khuyến mãi phá huỷ biên lợi nhuận đồng loạt trên tất cả các năm và danh mục.
- **KPI Cards**: Global Promo Margin, Global No-Promo Margin, Diagnostic GP Gap.
- **Charts**: Margin Gap Heatmap (Category x Year), Diagnostic GP Gap by Category (Bar).
- **Filters**: Category, Year.
- **Data Source**: dashboard_promo_margin.csv.
- **Grain**: Category x Year.
- **Calculated Fields**: Margin Gap, Diagnostic GP Gap.
- **Action Map**: Xây dựng hàng rào kiểm soát khuyến mãi.
- **Caveat**: Upper-bound diagnostic.

## C. Category & Product Portfolio
- **Objective**: Quản trị danh mục sản phẩm theo hiệu quả sinh lời.
- **Audience Question**: Danh mục nào sinh lời, sản phẩm nào đang gặp rủi ro?
- **Core Claim**: Tập trung quá nhiều vào Streetwear lợi nhuận thấp; Problem SKUs kéo tụt chất lượng.
- **KPI Cards**: Core Engine Share, Profit Pool Margin, Problem SKU Count.
- **Charts**: Category Portfolio Matrix (Bubble), Product Action Quadrant (Scatter).
- **Filters**: Category, Segment.
- **Data Source**: dashboard_category_portfolio.csv, dashboard_product_actions.csv.
- **Grain**: Category-Segment / Product.
- **Calculated Fields**: Portfolio Role, Product Classification.
- **Action Map**: Review danh mục, xử lý Problem SKUs.
- **Caveat**: Đánh giá trên rating hiện tại, có thể nhiễu bởi size/color.

## D. Inventory Demand-Fit
- **Objective**: Phát hiện và tối ưu hoá sự sai lệch trong lập kế hoạch tồn kho.
- **Audience Question**: Tồn kho có đang đáp ứng đúng nhu cầu không?
- **Core Claim**: Lập kế hoạch sai lầm gây ra tình trạng vừa thừa vừa thiếu cùng lúc.
- **KPI Cards**: Overstock+Stockout Count, Retail Value at Risk.
- **Charts**: Inventory Flag Overlap (Bar), Mismatch Revenue at Risk (Stacked Bar).
- **Filters**: Category, Overlap State.
- **Data Source**: dashboard_inventory_mismatch.csv.
- **Grain**: Product x Month.
- **Calculated Fields**: Overlap State logic.
- **Action Map**: Nâng cấp hệ thống Replenishment/Planning.
- **Caveat**: Cờ tồn kho chốt cuối tháng.

## E. Customer Lifecycle
- **Objective**: Đánh giá chất lượng tệp khách hàng.
- **Audience Question**: Tập khách hàng nào đang sinh lời, ai đang rời bỏ?
- **Core Claim**: Doanh thu phụ thuộc mạnh vào Champions, nhưng tệp Lost quá lớn.
- **KPI Cards**: Champions Revenue Share, Lost Customer Count.
- **Charts**: Customer Segment Value Map (Scatter), Inactive Pool (Pie).
- **Filters**: RFM Segment.
- **Data Source**: dashboard_customer_lifecycle.csv.
- **Grain**: Customer.
- **Calculated Fields**: RFM Segment logic.
- **Action Map**: Chạy chiến dịch targeted reactivation cho nhóm Lost.
- **Caveat**: Tính toán dựa trên non-cancelled line-item revenue.
