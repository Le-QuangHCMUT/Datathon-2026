import os
import pandas as pd

def run_eda_008():
    print("Starting EDA-008 Dashboard Package Generation...")

    out_t = "artifacts/tables"
    out_d = "docs/dashboard_spec"
    out_r = "docs/report_outline"
    out_i = "docs/insight_log"

    os.makedirs(out_t, exist_ok=True)
    os.makedirs(out_d, exist_ok=True)
    os.makedirs(out_r, exist_ok=True)
    os.makedirs(out_i, exist_ok=True)

    # 1. Final story spine v2
    with open(f"{out_r}/final_story_spine_v2.md", "w", encoding="utf-8") as f:
        f.write("""# Final Story Spine v2

## Khuyến nghị Luận điểm (Recommended Thesis)
Doanh nghiệp đang đánh mất sự tăng trưởng chất lượng. Sự sụt giảm doanh thu chủ yếu là do giảm sản lượng (volume-driven), trong khi phần nhu cầu còn lại bị bào mòn bởi các chương trình khuyến mãi cắt giảm biên lợi nhuận, sự tập trung vào danh mục Streetwear có biên lợi nhuận thấp, sự mất cân đối giữa lượng tồn kho và nhu cầu, cùng với khả năng giữ chân khách hàng kém. Cần chuyển hướng chiến lược từ việc chạy theo GMV (dựa trên giảm giá) sang tăng trưởng lợi nhuận chất lượng: quản trị khuyến mãi, tập trung vào nhóm sản phẩm mang lại lợi nhuận, điều phối tồn kho khớp với nhu cầu, và chiến lược vòng đời khách hàng mục tiêu.

## Tóm tắt Luận điểm (One-sentence Executive Thesis)
Chuyển dịch từ tăng trưởng doanh thu dựa trên giảm giá sang tăng trưởng lợi nhuận chất lượng thông qua việc cắt giảm rò rỉ biên lợi nhuận khuyến mãi, tối ưu hoá tồn kho, và kích hoạt lại tệp khách hàng.

## Tại sao luận điểm này tốt hơn các phương án khác
Luận điểm này dựa trên các dữ liệu đã được kiểm chứng (sự sụt giảm biên lợi nhuận -0.19 do promo, 1.226B sụt giảm do sản lượng, và sự mất cân đối tồn kho) thay vì chỉ nhìn vào một chỉ số doanh thu sụt giảm. Nó chỉ ra các đòn bẩy kinh doanh có thể can thiệp ngay lập tức mà không cần phụ thuộc vào các giả định nhân quả chưa rõ ràng.

## Cấu trúc Câu chuyện (Narrative Arc)
1. **Chuyện gì đã xảy ra**: Doanh thu giảm mạnh sau đỉnh năm 2016 (-935M từ 2016 đến 2022).
2. **Tại sao điều này quan trọng**: Động lực sụt giảm là do sản lượng (mất 1.226B do volume) chứ không chỉ do giá. Động cơ tăng trưởng cốt lõi đang yếu đi.
3. **Giá trị đang rò rỉ ở đâu**: Biên lợi nhuận bị phá huỷ nặng nề bởi các chương trình khuyến mãi (chênh lệch -0.19 margin), áp dụng đồng loạt qua 44 phân khúc danh mục-năm.
4. **Mô hình vận hành gãy ở đâu**: Sự mất cân đối trong lập kế hoạch tồn kho dẫn đến 30,495 trường hợp vừa dư thừa (overstock) vừa thiếu hụt (stockout) trong cùng tháng, khóa chặt ~34.88B giá trị bán lẻ.
5. **Cần bảo vệ và tái kích hoạt ai**: Tệp khách hàng Champions đóng góp 63.8% doanh thu, trong khi có tới 25,009 khách hàng đã chuyển sang trạng thái Lost.
6. **Cần làm gì**: Khai thác danh mục hành động (quản trị promo, điều chỉnh danh mục sản phẩm, cải thiện planning, kích hoạt lại khách hàng).
7. **Đo lường như thế nào**: Hệ thống KPI theo dõi Margin, Stockout Rate, Order Volume, và Repeat Rate.

## Hero Insights (Với bằng chứng chính xác)
- **Promo margin leakage**: Chênh lệch margin -0.19. Gap chẩn đoán lợi nhuận gộp lên tới 86.45M. Hành động: Cắt giảm khuyến mãi diện rộng.
- **Volume-driven revenue decline**: Tác động từ sản lượng là -1.226B. Đơn hàng giảm 56%. Hành động: Tái đầu tư phễu thu hút.
- **Inventory mismatch**: 30,495 tháng-sản phẩm bị Overstock+Stockout đồng thời. Hành động: Nâng cấp hệ thống dự báo.
- **Customer lifecycle**: Tệp Lost đạt 25,009 người. Hành động: Chiến dịch kích hoạt mục tiêu.

## Supporting Insights
- Danh mục Streetwear chiếm 31.2% doanh thu nhưng biên lợi nhuận thấp.
- GenZ là nhóm lợi nhuận tiềm năng nhưng tỷ trọng doanh thu quá nhỏ.
- Phân khúc sản phẩm chia rõ Hero, Problem, Sleeper.

## Rejected Signals
- Khu vực địa lý (Region): Sự chênh lệch quá nhỏ để làm hero insight.
- Nguồn truy cập Web: Dữ liệu ở mức tổng hợp ngày, không thể dùng làm attribution.

## Caveats (Lưu ý)
- Phân tích rò rỉ lợi nhuận là chẩn đoán mức trần, không phải lợi nhuận phục hồi nhân quả.
- Tồn kho là báo cáo cuối tháng, stockout diễn ra trong tháng.

## Thứ tự báo cáo đề xuất
1. Executive Summary & Revenue Quality
2. Profitability (Promo Leakage & Category Mix)
3. Operations (Inventory Mismatch)
4. Customer Lifecycle
""")

    # 2. Report outline v1
    with open(f"{out_r}/report_outline_v1.md", "w", encoding="utf-8") as f:
        f.write("""# Report Outline v1 (NeurIPS Style - 4 pages)

## Kế hoạch trang (Page Budget Plan)
- **Page 1**: Executive Summary, Problem Framing, Data Reliability.
- **Page 2**: Revenue Quality & Promo Margin Leakage (Hero 1 & 2).
- **Page 3**: Inventory Operations & Customer Lifecycle (Hero 3 & 4).
- **Page 4**: Business Recommendations, Forecasting Placeholder, Limitations.

## Section 1: Problem framing and data reliability
- Tuyên bố bài toán: Chuyển dịch từ tăng trưởng GMV sang lợi nhuận chất lượng.
- Xác thực dữ liệu: Khớp doanh thu line-item với sales target, làm sạch tệp RFM. (Appendix: Technical Sales Reconciliation).

## Section 2: EDA story / hero insights
- **Claim 1**: Sụt giảm doanh thu chủ yếu do khối lượng (-1.226B volume effect).
- **Claim 2**: Rò rỉ biên lợi nhuận do lạm dụng khuyến mãi (gap -0.19).
- **Claim 3**: Mất cân đối tồn kho hệ thống (30,495 trường hợp overstock+stockout).
- **Claim 4**: Chảy máu khách hàng (25k Lost, 33k Registered No Orders).

## Section 3: Forecasting approach placeholder
- Khung mô hình dự báo sẽ tích hợp tính thời vụ, xu hướng danh mục, và kế hoạch tồn kho. (Dự kiến thực hiện ở vòng sau).

## Section 4: Business recommendations and limitations
- Khuyến nghị: Thiết lập ngưỡng margin tối thiểu cho promo, làm sạch danh mục problem SKU, cải tiến planning tồn kho, tái kích hoạt tập khách hàng mục tiêu.
- Lưu ý (Caveats): Các đánh giá gap lợi nhuận mang tính chẩn đoán; dữ liệu web không có tính nhân quả cho kênh thu hút; tồn kho phản ánh hiện tượng cuối tháng.

## Figure/Table Placement Plan
- Page 1: Revenue Decline Bridge (Figure 1).
- Page 2: Promo Gap Heatmap & Category Portfolio Bubble (Figure 2, 3).
- Page 3: Inventory Flag Overlap & Customer Inactive Pool (Figure 4, 5).
- Appendix: Tóm tắt KPI và Reconciliation Tables.
""")

    # 3. Dashboard architecture v2
    with open(f"{out_d}/dashboard_architecture_v2.md", "w", encoding="utf-8") as f:
        f.write("""# Dashboard Architecture v2

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
""")

    # 4. Tableau build guide v1
    with open(f"{out_d}/tableau_build_guide_v1.md", "w", encoding="utf-8") as f:
        f.write("""# Tableau Build Guide v1

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
""")

    # 5. Dashboard data dictionary
    with open(f"{out_d}/dashboard_data_dictionary.md", "w", encoding="utf-8") as f:
        f.write("""# Dashboard Data Dictionary

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
""")

    # 6. Hero / Supporting Insights
    hi_df = pd.DataFrame([
        {"insight_id": "HI-01", "priority": 1, "status": "hero", "theme": "Promotions", "final_claim": "Promotions deeply erode margin across all categories.", "evidence_metric_1": "-0.19 Margin Gap", "evidence_metric_2": "86.45M GP Gap", "evidence_metric_3": "44 cells", "evidence_metric_4": "", "source_tables": "oi, prod", "grain": "category-year", "denominator": "sales", "recommended_action": "Reduce broad promos", "measurement_kpi": "Gross Margin", "tradeoff": "Volume loss", "caveat": "Diagnostic bound", "dashboard_page": "Promo", "report_section": "Profit", "figure_candidate": "eda_006_promo_profit_gap_by_category.png"},
        {"insight_id": "HI-02", "priority": 2, "status": "hero", "theme": "Revenue Decline", "final_claim": "Volume loss is the primary driver of revenue decline.", "evidence_metric_1": "-1.226B Volume Effect", "evidence_metric_2": "-56% orders", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "oi", "grain": "year", "denominator": "NA", "recommended_action": "Invest in top funnel", "measurement_kpi": "Orders", "tradeoff": "CAC", "caveat": "None", "dashboard_page": "Exec", "report_section": "Growth", "figure_candidate": "eda_006_revenue_decline_driver_bridge.png"},
        {"insight_id": "HI-03", "priority": 3, "status": "hero", "theme": "Inventory", "final_claim": "Systemic inventory mismatch: simultaneous overstock and stockout.", "evidence_metric_1": "30,495 overlap cases", "evidence_metric_2": "34.88B retail value at risk", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "inv", "grain": "product-month", "denominator": "stock", "recommended_action": "Fix planning system", "measurement_kpi": "Stockout Rate", "tradeoff": "None", "caveat": "Month-end snapshot", "dashboard_page": "Inv", "report_section": "Supply", "figure_candidate": "eda_006_inventory_flag_overlap.png"},
        {"insight_id": "HI-04", "priority": 4, "status": "hero", "theme": "Lifecycle", "final_claim": "Large base of purchased customers become Lost.", "evidence_metric_1": "25,009 Lost", "evidence_metric_2": "63.8% rev from Champions", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "cust", "grain": "customer", "denominator": "customers", "recommended_action": "Targeted reactivation", "measurement_kpi": "Repeat Rate", "tradeoff": "Promo cost", "caveat": "None", "dashboard_page": "Lifecycle", "report_section": "Retention", "figure_candidate": "eda_006_customer_inactive_pool.png"},
        {"insight_id": "SI-01", "priority": 5, "status": "supporting", "theme": "Category", "final_claim": "Streetwear/category concentration with lower margin.", "evidence_metric_1": "31.2% rev share", "evidence_metric_2": "12.8% margin", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "oi", "grain": "category", "denominator": "revenue", "recommended_action": "Diversify category", "measurement_kpi": "Margin", "tradeoff": "Topline drop", "caveat": "None", "dashboard_page": "Category", "report_section": "Profit", "figure_candidate": "eda_006_category_portfolio_bubble.png"},
        {"insight_id": "SI-02", "priority": 6, "status": "supporting", "theme": "Category", "final_claim": "GenZ/Trendy as profit pool but small revenue share.", "evidence_metric_1": "19.1% margin", "evidence_metric_2": "2.1% rev share", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "oi", "grain": "category", "denominator": "revenue", "recommended_action": "Test growth", "measurement_kpi": "Rev Share", "tradeoff": "Inventory risk", "caveat": "None", "dashboard_page": "Category", "report_section": "Growth", "figure_candidate": "eda_006_category_portfolio_bubble.png"},
        {"insight_id": "SI-03", "priority": 7, "status": "supporting", "theme": "Products", "final_claim": "Product action matrix separates Hero, Problem SKUs.", "evidence_metric_1": "Problem SKUs identified", "evidence_metric_2": "", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "prod", "grain": "product", "denominator": "SKUs", "recommended_action": "QC/Fit review", "measurement_kpi": "Return rate", "tradeoff": "Short term rev", "caveat": "Rating bias", "dashboard_page": "Category", "report_section": "Quality", "figure_candidate": "eda_006_product_action_quadrant.png"},
        {"insight_id": "SI-04", "priority": 8, "status": "supporting", "theme": "Geography", "final_claim": "Region useful as filter but not hero driver.", "evidence_metric_1": "Marginal friction diff", "evidence_metric_2": "", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "geo", "grain": "region", "denominator": "NA", "recommended_action": "Monitor", "measurement_kpi": "Del days", "tradeoff": "None", "caveat": "None", "dashboard_page": "Any", "report_section": "Appx", "figure_candidate": ""},
        {"insight_id": "AI-01", "priority": 9, "status": "appendix", "theme": "Web", "final_claim": "Web sessions/revenue correlation is weak/moderate.", "evidence_metric_1": "r ~ 0.32", "evidence_metric_2": "", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "web", "grain": "daily", "denominator": "NA", "recommended_action": "Improve tracking", "measurement_kpi": "Corr", "tradeoff": "None", "caveat": "Daily grain", "dashboard_page": "None", "report_section": "Appx", "figure_candidate": ""},
        {"insight_id": "AI-02", "priority": 10, "status": "appendix", "theme": "Data", "final_claim": "Technical validation: sales equals line-item GMV.", "evidence_metric_1": "100% Match", "evidence_metric_2": "", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "sales", "grain": "order", "denominator": "NA", "recommended_action": "Use GMV logic", "measurement_kpi": "Reconciliation", "tradeoff": "None", "caveat": "None", "dashboard_page": "None", "report_section": "Appx", "figure_candidate": ""},
        {"insight_id": "RI-01", "priority": 11, "status": "reject", "theme": "Payments", "final_claim": "Payments as revenue proxy.", "evidence_metric_1": "Mismatched grain", "evidence_metric_2": "", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "pay", "grain": "NA", "denominator": "NA", "recommended_action": "Reject", "measurement_kpi": "NA", "tradeoff": "NA", "caveat": "NA", "dashboard_page": "None", "report_section": "None", "figure_candidate": ""},
        {"insight_id": "RI-02", "priority": 12, "status": "reject", "theme": "Web Attribution", "final_claim": "Source-level web traffic attribution.", "evidence_metric_1": "Missing dimensions", "evidence_metric_2": "", "evidence_metric_3": "", "evidence_metric_4": "", "source_tables": "web", "grain": "NA", "denominator": "NA", "recommended_action": "Reject", "measurement_kpi": "NA", "tradeoff": "NA", "caveat": "NA", "dashboard_page": "None", "report_section": "None", "figure_candidate": ""}
    ])
    hi_df.to_csv(f"{out_t}/eda_008_final_hero_supporting_insight_portfolio.csv", index=False)

    # 7. Expanded Map
    cm_df = pd.DataFrame([
        {"claim": "Promotions deeply erode margin across all categories.", "evidence": "-0.19 Margin Gap", "visual": "eda_006_promo_profit_gap_by_category.png", "audience_question": "Why is margin dropping?", "decision_enabled": "Promo governance", "recommended_action": "Category-year monitoring, A/B test promo depth", "KPI_to_monitor": "Margin", "caveat": "Diagnostic bound", "dashboard_placement": "Promo", "report_placement": "Profit", "owner_function": "Exec", "expected_tradeoff": "Volume loss"},
        {"claim": "Volume loss is the primary driver of revenue decline.", "evidence": "-1.226B Volume Effect", "visual": "eda_006_revenue_decline_driver_bridge.png", "audience_question": "Why did revenue drop?", "decision_enabled": "Growth engine", "recommended_action": "Reacquire volume without blanket discount", "KPI_to_monitor": "Orders", "caveat": "Descriptive", "dashboard_placement": "Exec", "report_placement": "Growth", "owner_function": "Exec", "expected_tradeoff": "CAC increase"},
        {"claim": "Systemic inventory mismatch: simultaneous overstock and stockout.", "evidence": "30,495 cases", "visual": "eda_006_inventory_flag_overlap.png", "audience_question": "Is supply chain working?", "decision_enabled": "Inventory", "recommended_action": "Demand-fit replenishment", "KPI_to_monitor": "Stockout Rate", "caveat": "Snapshot based", "dashboard_placement": "Inv", "report_placement": "Ops", "owner_function": "Ops", "expected_tradeoff": "System cost"},
        {"claim": "Large base of purchased customers become Lost.", "evidence": "25,009 Lost", "visual": "eda_006_customer_inactive_pool.png", "audience_question": "Are we keeping customers?", "decision_enabled": "Lifecycle", "recommended_action": "Reactivate At Risk/Lost", "KPI_to_monitor": "Repeat Rate", "caveat": "Non-cancelled base", "dashboard_placement": "Lifecycle", "report_placement": "Retention", "owner_function": "Marketing", "expected_tradeoff": "Promo cost"},
        {"claim": "Streetwear/category concentration with lower margin.", "evidence": "31.2% rev share, 12.8% margin", "visual": "eda_006_category_portfolio_bubble.png", "audience_question": "What is our mix?", "decision_enabled": "Category", "recommended_action": "Reduce blanket discount on risk concentration", "KPI_to_monitor": "Mix Share", "caveat": "None", "dashboard_placement": "Category", "report_placement": "Profit", "owner_function": "Merch", "expected_tradeoff": "Sales dip"},
        {"claim": "GenZ/Trendy as profit pool but small revenue share.", "evidence": "19.1% margin", "visual": "eda_006_category_portfolio_bubble.png", "audience_question": "Where is the growth?", "decision_enabled": "Category", "recommended_action": "Test growth for profit-pool", "KPI_to_monitor": "Revenue", "caveat": "None", "dashboard_placement": "Category", "report_placement": "Growth", "owner_function": "Merch", "expected_tradeoff": "Stock risk"},
        {"claim": "Product action matrix separates Hero, Problem SKUs.", "evidence": "Problem SKUs flag", "visual": "eda_006_product_action_quadrant.png", "audience_question": "What SKUs to fix?", "decision_enabled": "Product", "recommended_action": "Investigate fit/QC", "KPI_to_monitor": "Return Rate", "caveat": "Rating bias", "dashboard_placement": "Category", "report_placement": "Quality", "owner_function": "Merch", "expected_tradeoff": "Delist cost"},
        {"claim": "Champions drive the business.", "evidence": "63.8% rev share", "visual": "eda_006_customer_segment_value_map.png", "audience_question": "Who buys the most?", "decision_enabled": "Lifecycle", "recommended_action": "Protect Champions", "KPI_to_monitor": "CLV", "caveat": "None", "dashboard_placement": "Lifecycle", "report_placement": "Retention", "owner_function": "Marketing", "expected_tradeoff": "None"},
        {"claim": "Registered No Orders is a huge untapped pool.", "evidence": "33k users", "visual": "eda_006_customer_inactive_pool.png", "audience_question": "Who are we missing?", "decision_enabled": "Lifecycle", "recommended_action": "Activate Registered No Orders", "KPI_to_monitor": "Conversion", "caveat": "None", "dashboard_placement": "Lifecycle", "report_placement": "Growth", "owner_function": "Marketing", "expected_tradeoff": "CAC"},
        {"claim": "Inventory Overstock only is a secondary issue.", "evidence": "15,447 cases", "visual": "eda_006_inventory_flag_overlap.png", "audience_question": "Do we just have too much stock?", "decision_enabled": "Inventory", "recommended_action": "Liquidation for low-demand overstock", "KPI_to_monitor": "Stock Value", "caveat": "None", "dashboard_placement": "Inv", "report_placement": "Ops", "owner_function": "Ops", "expected_tradeoff": "Margin hit"},
        {"claim": "Region friction is minimal.", "evidence": "Similar delivery days", "visual": "None", "audience_question": "Are regions different?", "decision_enabled": "Logistics", "recommended_action": "Standardize SLA", "KPI_to_monitor": "Delivery days", "caveat": "None", "dashboard_placement": "Any", "report_placement": "Appx", "owner_function": "Ops", "expected_tradeoff": "None"},
        {"claim": "Target line-item GMV reconciliation.", "evidence": "100% Match", "visual": "None", "audience_question": "Is data right?", "decision_enabled": "Analytics", "recommended_action": "Use GMV", "KPI_to_monitor": "Match %", "caveat": "None", "dashboard_placement": "None", "report_placement": "Appx", "owner_function": "Data", "expected_tradeoff": "None"}
    ])
    cm_df.to_csv(f"{out_t}/eda_008_claim_evidence_action_map_expanded.csv", index=False)

    # 8. Report Figure Shortlist
    fs_df = pd.DataFrame([
        {"artifact_path": "artifacts/figures/eda_006_revenue_decline_driver_bridge.png", "figure_title": "Revenue Decline Bridge", "claim_supported": "Volume loss drives decline", "report_priority": "High", "dashboard_page": "Exec", "needs_redraw_for_publication": "Yes", "caveat": "None", "notes_for_redraw": "Improve colors"},
        {"artifact_path": "artifacts/figures/eda_006_promo_profit_gap_by_category.png", "figure_title": "Promo Gap", "claim_supported": "Margin leakage", "report_priority": "High", "dashboard_page": "Promo", "needs_redraw_for_publication": "Yes", "caveat": "Upper bound diagnostic", "notes_for_redraw": "Add data labels"},
        {"artifact_path": "artifacts/figures/eda_006_promo_margin_gap_heatmap_like.png", "figure_title": "Margin Gap Heatmap", "claim_supported": "Promo leakage ubiquitous", "report_priority": "High", "dashboard_page": "Promo", "needs_redraw_for_publication": "Yes", "caveat": "None", "notes_for_redraw": "Refine grid"},
        {"artifact_path": "artifacts/figures/eda_006_inventory_flag_overlap.png", "figure_title": "Inventory Overlap", "claim_supported": "Planning mismatch", "report_priority": "High", "dashboard_page": "Inv", "needs_redraw_for_publication": "Yes", "caveat": "Month end", "notes_for_redraw": "Clean axes"},
        {"artifact_path": "artifacts/figures/eda_006_inventory_mismatch_revenue_at_risk.png", "figure_title": "Inventory Value Risk", "claim_supported": "High value locked", "report_priority": "Medium", "dashboard_page": "Inv", "needs_redraw_for_publication": "No", "caveat": "Retail value", "notes_for_redraw": "None"},
        {"artifact_path": "artifacts/figures/eda_006_customer_inactive_pool.png", "figure_title": "Inactive Pool", "claim_supported": "Large Lost/No Order base", "report_priority": "High", "dashboard_page": "Lifecycle", "needs_redraw_for_publication": "Yes", "caveat": "Non-cancelled base", "notes_for_redraw": "Explode slice"},
        {"artifact_path": "artifacts/figures/eda_006_customer_segment_value_map.png", "figure_title": "Customer Segment Map", "claim_supported": "Champions drive revenue", "report_priority": "Medium", "dashboard_page": "Lifecycle", "needs_redraw_for_publication": "Yes", "caveat": "None", "notes_for_redraw": "Log scale"},
        {"artifact_path": "artifacts/figures/eda_006_category_portfolio_bubble.png", "figure_title": "Category Bubble", "claim_supported": "Streetwear risk, GenZ growth", "report_priority": "Medium", "dashboard_page": "Category", "needs_redraw_for_publication": "Yes", "caveat": "CAGR timeframe", "notes_for_redraw": "Label clarity"},
        {"artifact_path": "artifacts/figures/eda_006_product_action_quadrant.png", "figure_title": "Product Quadrant", "claim_supported": "Problem SKUs", "report_priority": "Appendix", "dashboard_page": "Category", "needs_redraw_for_publication": "No", "caveat": "Rating bias", "notes_for_redraw": "None"},
        {"artifact_path": "artifacts/figures/eda_006_cohort_retention_decay.png", "figure_title": "Cohort Retention", "claim_supported": "Churn", "report_priority": "Appendix", "dashboard_page": "Lifecycle", "needs_redraw_for_publication": "No", "caveat": "None", "notes_for_redraw": "None"}
    ])
    fs_df.to_csv(f"{out_t}/eda_008_report_figure_shortlist_expanded.csv", index=False)

    # 9. Dashboard Spec CSVs
    pd.DataFrame([
        {"page_id": "A", "page_name": "Executive Revenue Quality", "objective": "Track topline", "audience_question": "Why drop?", "primary_claim": "Volume loss", "data_source": "dashboard_revenue_quality", "grain": "Year", "filters": "Year", "caveat": "None", "rubric_value": "High"},
        {"page_id": "B", "page_name": "Promo Margin Leakage", "objective": "Track margin", "audience_question": "Why margin dip?", "primary_claim": "Promo leak", "data_source": "dashboard_promo_margin", "grain": "Category-Year", "filters": "Category", "caveat": "Bound", "rubric_value": "High"},
        {"page_id": "C", "page_name": "Category & Product", "objective": "Track mix", "audience_question": "What sells?", "primary_claim": "Streetwear risk", "data_source": "dashboard_category_portfolio", "grain": "Segment", "filters": "Segment", "caveat": "None", "rubric_value": "Medium"},
        {"page_id": "D", "page_name": "Inventory", "objective": "Track stock", "audience_question": "Stockout?", "primary_claim": "Planning error", "data_source": "dashboard_inventory_mismatch", "grain": "Product-Month", "filters": "Category", "caveat": "Month end", "rubric_value": "High"},
        {"page_id": "E", "page_name": "Lifecycle", "objective": "Track users", "audience_question": "Who buys?", "primary_claim": "Champions", "data_source": "dashboard_customer_lifecycle", "grain": "Customer", "filters": "RFM", "caveat": "None", "rubric_value": "High"}
    ]).to_csv(f"{out_t}/eda_008_dashboard_page_spec.csv", index=False)
    
    pd.DataFrame([
        {"page_id": "A", "kpi_name": "Revenue", "formula": "SUM(revenue)", "data_source": "dashboard_revenue_quality", "grain": "Year", "interpretation": "Topline", "caveat": "None"},
        {"page_id": "A", "kpi_name": "Volume Effect", "formula": "SUM(Volume Effect)", "data_source": "dashboard_revenue_quality", "grain": "Year", "interpretation": "Loss from orders", "caveat": "Descriptive"},
        {"page_id": "B", "kpi_name": "Diagnostic GP Gap", "formula": "SUM(diagnostic_gp_gap)", "data_source": "dashboard_promo_margin", "grain": "Cat-Year", "interpretation": "Max leak", "caveat": "Bound"},
        {"page_id": "C", "kpi_name": "Margin", "formula": "AVG(margin)", "data_source": "dashboard_category_portfolio", "grain": "Segment", "interpretation": "Profitability", "caveat": "None"},
        {"page_id": "D", "kpi_name": "Retail Value", "formula": "SUM(stock_retail_value)", "data_source": "dashboard_inventory_mismatch", "grain": "Month", "interpretation": "Value at risk", "caveat": "None"},
        {"page_id": "E", "kpi_name": "Customer Count", "formula": "COUNT(customer_id)", "data_source": "dashboard_customer_lifecycle", "grain": "Customer", "interpretation": "Audience size", "caveat": "None"}
    ]).to_csv(f"{out_t}/eda_008_dashboard_kpi_spec.csv", index=False)

    pd.DataFrame([
        {"page_id": "A", "chart_id": "A1", "chart_title": "Waterfall", "chart_type": "Waterfall", "data_source": "dashboard_revenue_quality", "grain": "Year", "x_field": "Category", "y_field": "Value", "color_field": "Sign", "size_field": "None", "filter_fields": "None", "tooltip_fields": "Value", "claim_supported": "Volume loss", "caveat": "None"},
        {"page_id": "B", "chart_id": "B1", "chart_title": "Heatmap", "chart_type": "Heatmap", "data_source": "dashboard_promo_margin", "grain": "Cat-Year", "x_field": "Year", "y_field": "Category", "color_field": "Margin Gap", "size_field": "None", "filter_fields": "Category", "tooltip_fields": "Gap", "claim_supported": "Promo leak", "caveat": "None"},
        {"page_id": "D", "chart_id": "D1", "chart_title": "Overlap", "chart_type": "Bar", "data_source": "dashboard_inventory_mismatch", "grain": "Month", "x_field": "Overlap", "y_field": "Count", "color_field": "Overlap", "size_field": "None", "filter_fields": "Category", "tooltip_fields": "Count", "claim_supported": "Mismatch", "caveat": "None"}
    ]).to_csv(f"{out_t}/eda_008_dashboard_chart_spec.csv", index=False)

    pd.DataFrame([
        {"page_id": "A", "field_name": "Volume Effect", "tableau_formula_or_logic": "Lookup prev year", "data_source": "dashboard_revenue_quality", "purpose": "Bridge", "caveat": "None"},
        {"page_id": "B", "field_name": "Margin Gap", "tableau_formula_or_logic": "margin_promo - margin_nopromo", "data_source": "dashboard_promo_margin", "purpose": "Diff", "caveat": "None"}
    ]).to_csv(f"{out_t}/eda_008_calculated_fields_spec.csv", index=False)

    # 10. Checklist
    pd.DataFrame([
        {"artifact": "dashboard_revenue_quality.csv", "exists": "Yes", "ready_for_dashboard": "Yes", "ready_for_report": "Yes", "issue": "None", "next_action": "None"},
        {"artifact": "dashboard_promo_margin.csv", "exists": "Yes", "ready_for_dashboard": "Yes", "ready_for_report": "Yes", "issue": "None", "next_action": "None"},
        {"artifact": "dashboard_category_portfolio.csv", "exists": "Yes", "ready_for_dashboard": "Yes", "ready_for_report": "Yes", "issue": "None", "next_action": "None"},
        {"artifact": "dashboard_inventory_mismatch.csv", "exists": "Yes", "ready_for_dashboard": "Yes", "ready_for_report": "Yes", "issue": "None", "next_action": "None"},
        {"artifact": "dashboard_customer_lifecycle.csv", "exists": "Yes", "ready_for_dashboard": "Yes", "ready_for_report": "Yes", "issue": "None", "next_action": "None"},
        {"artifact": "dashboard_product_actions.csv", "exists": "Yes", "ready_for_dashboard": "Yes", "ready_for_report": "Yes", "issue": "None", "next_action": "None"}
    ]).to_csv(f"{out_t}/eda_008_artifact_readiness_checklist.csv", index=False)

    # 11. Package Report
    with open(f"{out_i}/eda_008_story_dashboard_package_report.md", "w", encoding="utf-8") as f:
        f.write("""# EDA-008 Story & Dashboard Package Report

## Những gì đã được cải thiện từ EDA-007
- Mở rộng chi tiết Map Claim-Evidence-Action.
- Cụ thể hoá tài liệu thiết kế Dashboard (Data dictionary, tableau guide).
- Bổ sung các Insights Supporting & Rejected.

## Final Story Decision
Câu chuyện chuyển dịch từ GMV sang Quality Growth (Profit, Inventory, Lifecycle) là xương sống chính.

## Remaining Risks
Dữ liệu nhân quả chưa có, mọi thứ đang dừng ở mức chẩn đoán.

## Next Sprint
Sẵn sàng cho việc xây dựng Tableau trực tiếp.
""")

    print("EDA-008 Completed.")

if __name__ == "__main__":
    run_eda_008()
