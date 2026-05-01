import os
import pandas as pd
import numpy as np

def run_semantic_audit():
    print("Starting Semantic Audit...")
    
    # 1. Load CSVs safely
    data_dir = "data"
    
    def load_csv(name, **kwargs):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path) and name == "sample_submission.csv":
            path = name
        if not os.path.exists(path):
            print(f"Warning: {name} not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path, **kwargs)

    customers = load_csv("customers.csv")
    geography = load_csv("geography.csv")
    inventory = load_csv("inventory.csv")
    orders = load_csv("orders.csv", parse_dates=["order_date"])
    order_items = load_csv("order_items.csv", dtype={"promo_id": str, "promo_id_2": str})
    payments = load_csv("payments.csv")
    products = load_csv("products.csv")
    promotions = load_csv("promotions.csv")
    returns = load_csv("returns.csv")
    reviews = load_csv("reviews.csv")
    sales = load_csv("sales.csv", parse_dates=["Date"])
    shipments = load_csv("shipments.csv")
    web_traffic = load_csv("web_traffic.csv")
    sample_submission = load_csv("sample_submission.csv")

    # Outputs
    out_dir = "artifacts/diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    # 2. Composite key and grain checks
    print("Running 2. Composite key and grain checks...")
    comp_keys = []
    
    def check_dups(df, cols, name):
        if not df.empty and all(c in df.columns for c in cols):
            dups = df.duplicated(subset=cols).sum()
            comp_keys.append({"check": name, "columns": str(cols), "duplicate_count": dups})
            
    check_dups(order_items, ["order_id", "product_id"], "order_items by (order_id, product_id)")
    if not order_items.empty:
        check_dups(order_items, [c for c in order_items.columns if c != "quantity"], "order_items all except quantity")
    check_dups(inventory, ["snapshot_date", "product_id"], "inventory by (snapshot_date, product_id)")
    check_dups(web_traffic, ["date"], "web_traffic by date")
    check_dups(returns, ["order_id", "product_id", "return_date"], "returns by (order_id, product_id, return_date)")
    check_dups(reviews, ["order_id", "product_id", "customer_id", "review_date"], "reviews by (order_id, product_id, customer_id, review_date)")
    check_dups(shipments, ["order_id"], "shipments by order_id")
    check_dups(payments, ["order_id"], "payments by order_id")
    
    pd.DataFrame(comp_keys).to_csv(f"{out_dir}/validation_002_composite_key_checks.csv", index=False)

    # 3. Referential integrity checks
    print("Running 3. Referential integrity checks...")
    ref_int = []
    def check_ref(df_from, col_from, df_to, col_to, name):
        if not df_from.empty and not df_to.empty and col_from in df_from.columns and col_to in df_to.columns:
            orphans = df_from[~df_from[col_from].isin(df_to[col_to].dropna())][col_from].dropna()
            count = len(orphans)
            pct = count / len(df_from) if len(df_from) > 0 else 0
            ref_int.append({"check": name, "orphan_count": count, "orphan_pct": pct})
            
    check_ref(orders, "customer_id", customers, "customer_id", "orders.customer_id not in customers")
    check_ref(orders, "zip", geography, "zip", "orders.zip not in geography")
    check_ref(order_items, "order_id", orders, "order_id", "order_items.order_id not in orders")
    check_ref(order_items, "product_id", products, "product_id", "order_items.product_id not in products")
    check_ref(payments, "order_id", orders, "order_id", "payments.order_id not in orders")
    check_ref(shipments, "order_id", orders, "order_id", "shipments.order_id not in orders")
    check_ref(returns, "order_id", orders, "order_id", "returns.order_id not in orders")
    check_ref(returns, "product_id", products, "product_id", "returns.product_id not in products")
    check_ref(reviews, "order_id", orders, "order_id", "reviews.order_id not in orders")
    check_ref(reviews, "product_id", products, "product_id", "reviews.product_id not in products")
    check_ref(reviews, "customer_id", customers, "customer_id", "reviews.customer_id not in customers")
    check_ref(inventory, "product_id", products, "product_id", "inventory.product_id not in products")
    check_ref(order_items.dropna(subset=["promo_id"]), "promo_id", promotions, "promo_id", "order_items.promo_id not in promotions")
    check_ref(order_items.dropna(subset=["promo_id_2"]), "promo_id_2", promotions, "promo_id", "order_items.promo_id_2 not in promotions")

    pd.DataFrame(ref_int).to_csv(f"{out_dir}/validation_002_referential_integrity.csv", index=False)

    # 4. Status consistency checks
    print("Running 4. Status consistency checks...")
    status_checks = []
    
    if not orders.empty and "order_status" in orders.columns:
        counts = orders["order_status"].value_counts().to_dict()
        for k, v in counts.items():
            status_checks.append({"check": f"orders.order_status == '{k}'", "count": v})
            
        if not shipments.empty:
            merged = shipments.merge(orders[["order_id", "order_status"]], on="order_id", how="left")
            for k, v in merged["order_status"].value_counts().to_dict().items():
                status_checks.append({"check": f"shipments order_status == '{k}'", "count": v})
                
        if not returns.empty:
            merged = returns.merge(orders[["order_id", "order_status"]], on="order_id", how="left")
            for k, v in merged["order_status"].value_counts().to_dict().items():
                status_checks.append({"check": f"returns order_status == '{k}'", "count": v})
                
        if not reviews.empty:
            merged = reviews.merge(orders[["order_id", "order_status"]], on="order_id", how="left")
            for k, v in merged["order_status"].value_counts().to_dict().items():
                status_checks.append({"check": f"reviews order_status == '{k}'", "count": v})

        cancelled = orders[orders["order_status"] == "cancelled"]
        if not payments.empty:
            c = payments[payments["order_id"].isin(cancelled["order_id"])].shape[0]
            status_checks.append({"check": "cancelled orders with payment records", "count": c})
        if not shipments.empty:
            c = shipments[shipments["order_id"].isin(cancelled["order_id"])].shape[0]
            status_checks.append({"check": "cancelled orders with shipment records", "count": c})
        if not returns.empty:
            c = returns[returns["order_id"].isin(cancelled["order_id"])].shape[0]
            status_checks.append({"check": "cancelled orders with return records", "count": c})
        if not reviews.empty:
            c = reviews[reviews["order_id"].isin(cancelled["order_id"])].shape[0]
            status_checks.append({"check": "cancelled orders with review records", "count": c})
            
    pd.DataFrame(status_checks).to_csv(f"{out_dir}/validation_002_status_consistency.csv", index=False)

    # 5. Business constraint checks
    print("Running 5. Business constraint checks...")
    biz_constraints = []
    
    if not products.empty:
        c = products[products["cogs"] >= products["price"]].shape[0]
        biz_constraints.append({"check": "products cogs >= price", "violation_count": c})
    if not order_items.empty:
        c1 = order_items[order_items["quantity"] <= 0].shape[0]
        c2 = order_items[order_items["unit_price"] < 0].shape[0]
        c3 = order_items[order_items["discount_amount"] < 0].shape[0]
        biz_constraints.append({"check": "order_items quantity <= 0", "violation_count": c1})
        biz_constraints.append({"check": "order_items unit_price < 0", "violation_count": c2})
        biz_constraints.append({"check": "order_items discount_amount < 0", "violation_count": c3})
    if not payments.empty:
        c = payments[payments["payment_value"] < 0].shape[0]
        biz_constraints.append({"check": "payments payment_value < 0", "violation_count": c})
    if not sales.empty:
        c1 = sales[(sales["Revenue"] <= 0) | (sales["COGS"] < 0)].shape[0]
        c2 = sales[sales["COGS"] > sales["Revenue"]].shape[0]
        biz_constraints.append({"check": "sales Revenue <= 0 or COGS < 0", "violation_count": c1})
        biz_constraints.append({"check": "sales COGS > Revenue", "violation_count": c2})
    if not inventory.empty:
        c1 = inventory[inventory["stock_on_hand"] < 0].shape[0]
        c2 = inventory[inventory["units_received"] < 0].shape[0]
        c3 = inventory[inventory["units_sold"] < 0].shape[0]
        c4 = inventory[(inventory["fill_rate"] < 0) | (inventory["fill_rate"] > 1)].shape[0]
        c5 = inventory[(inventory["sell_through_rate"] < 0) | (inventory["sell_through_rate"] > 1)].shape[0]
        biz_constraints.append({"check": "inventory stock_on_hand < 0", "violation_count": c1})
        biz_constraints.append({"check": "inventory units_received < 0", "violation_count": c2})
        biz_constraints.append({"check": "inventory units_sold < 0", "violation_count": c3})
        biz_constraints.append({"check": "inventory fill_rate outside [0,1]", "violation_count": c4})
        biz_constraints.append({"check": "inventory sell_through_rate outside [0,1]", "violation_count": c5})
    if not reviews.empty:
        c = reviews[(reviews["rating"] < 1) | (reviews["rating"] > 5)].shape[0]
        biz_constraints.append({"check": "reviews rating outside 1-5", "violation_count": c})
    if not shipments.empty:
        c = shipments[shipments["delivery_date"] < shipments["ship_date"]].shape[0]
        biz_constraints.append({"check": "shipments delivery_date < ship_date", "violation_count": c})
        if not orders.empty:
            merged = shipments.merge(orders[["order_id", "order_date"]], on="order_id")
            c = merged[merged["ship_date"] < merged["order_date"].astype(str)].shape[0]
            biz_constraints.append({"check": "shipments ship_date < order_date", "violation_count": c})
    if not returns.empty and not orders.empty:
        merged = returns.merge(orders[["order_id", "order_date"]], on="order_id")
        c = merged[merged["return_date"] < merged["order_date"].astype(str)].shape[0]
        biz_constraints.append({"check": "returns return_date < order_date", "violation_count": c})

    pd.DataFrame(biz_constraints).to_csv(f"{out_dir}/validation_002_business_constraints.csv", index=False)

    # 6. Null hotspot profile
    print("Running 6. Null hotspot profile...")
    nulls = []
    datasets = {
        "customers.csv": customers, "geography.csv": geography, "inventory.csv": inventory,
        "orders.csv": orders, "order_items.csv": order_items, "payments.csv": payments,
        "products.csv": products, "promotions.csv": promotions, "returns.csv": returns,
        "reviews.csv": reviews, "sales.csv": sales, "shipments.csv": shipments, "web_traffic.csv": web_traffic
    }
    for name, df in datasets.items():
        if not df.empty:
            for col in df.columns:
                n = df[col].isnull().sum()
                if n > 0:
                    pct = n / len(df)
                    samp = str(df[col].dropna().head(3).tolist())
                    nulls.append({
                        "file": name, "column": col, "null_count": n, "null_pct": pct,
                        "dtype": str(df[col].dtype), "sample_non_null_values": samp
                    })
    nulls_df = pd.DataFrame(nulls).sort_values("null_pct", ascending=False)
    nulls_df.to_csv(f"{out_dir}/validation_002_null_hotspots.csv", index=False)

    # 7. Sales reconciliation audit
    print("Running 7. Sales reconciliation audit...")
    recon_summary = []
    recon_daily = pd.DataFrame()
    
    if not sales.empty and not orders.empty and not order_items.empty and not products.empty:
        sales_agg = sales.groupby("Date").agg({"Revenue": "sum", "COGS": "sum"}).reset_index()
        sales_agg["Date"] = pd.to_datetime(sales_agg["Date"]).dt.date
        
        oi_ext = order_items.merge(orders[["order_id", "order_date", "order_status"]], on="order_id")
        oi_ext = oi_ext.merge(products[["product_id", "cogs"]], on="product_id", how="left")
        oi_ext["order_date"] = pd.to_datetime(oi_ext["order_date"]).dt.date
        oi_ext["line_revenue"] = oi_ext["quantity"] * oi_ext["unit_price"]
        oi_ext["line_cogs"] = oi_ext["quantity"] * oi_ext["cogs"]
        
        # A. all_order_items
        agg_A = oi_ext.groupby("order_date").agg(A_Rev=("line_revenue", "sum"), A_COGS=("line_cogs", "sum")).reset_index()
        
        # B. non_cancelled_order_items
        agg_B = oi_ext[oi_ext["order_status"] != "cancelled"].groupby("order_date").agg(B_Rev=("line_revenue", "sum"), B_COGS=("line_cogs", "sum")).reset_index()
        
        # C. delivered_or_returned_order_items
        agg_C = oi_ext[oi_ext["order_status"].isin(["delivered", "returned"])].groupby("order_date").agg(C_Rev=("line_revenue", "sum"), C_COGS=("line_cogs", "sum")).reset_index()
        
        # D. payments_all_orders
        if not payments.empty:
            pay_ext = payments.merge(orders[["order_id", "order_date", "order_status"]], on="order_id")
            pay_ext["order_date"] = pd.to_datetime(pay_ext["order_date"]).dt.date
            agg_D = pay_ext.groupby("order_date").agg(D_Rev=("payment_value", "sum")).reset_index()
            agg_E = pay_ext[pay_ext["order_status"] != "cancelled"].groupby("order_date").agg(E_Rev=("payment_value", "sum")).reset_index()
        else:
            agg_D, agg_E = pd.DataFrame(columns=["order_date", "D_Rev"]), pd.DataFrame(columns=["order_date", "E_Rev"])

        merged = sales_agg.merge(agg_A, left_on="Date", right_on="order_date", how="left").drop(columns=["order_date"])
        merged = merged.merge(agg_B, left_on="Date", right_on="order_date", how="left").drop(columns=["order_date"])
        merged = merged.merge(agg_C, left_on="Date", right_on="order_date", how="left").drop(columns=["order_date"])
        if not agg_D.empty:
            merged = merged.merge(agg_D, left_on="Date", right_on="order_date", how="left").drop(columns=["order_date"])
        if not agg_E.empty:
            merged = merged.merge(agg_E, left_on="Date", right_on="order_date", how="left").drop(columns=["order_date"])
        
        def compare(df, name, col_rev):
            mask = df[col_rev].notnull() & df["Revenue"].notnull()
            df_m = df[mask]
            matched = len(df_m)
            if matched == 0:
                return {}
            diff = df_m[col_rev] - df_m["Revenue"]
            abs_diff = diff.abs()
            mae = abs_diff.mean()
            rmse = np.sqrt((diff**2).mean())
            mape = (abs_diff / df_m["Revenue"]).mean()
            corr = df_m[col_rev].corr(df_m["Revenue"])
            total_diff = df_m[col_rev].sum() - df_m["Revenue"].sum()
            med_diff = abs_diff.median()
            max_diff = abs_diff.max()
            pct_diff = abs_diff / df_m["Revenue"]
            within_01 = (pct_diff <= 0.001).sum()
            within_1 = (pct_diff <= 0.01).sum()
            within_5 = (pct_diff <= 0.05).sum()
            
            return {
                "definition": name, "matched_dates": matched, "MAE": mae, "RMSE": rmse, "MAPE": mape,
                "correlation": corr, "total_diff": total_diff, "median_abs_diff": med_diff, "max_abs_diff": max_diff,
                "dates_within_0.1%": within_01, "dates_within_1%": within_1, "dates_within_5%": within_5
            }
            
        recon_summary.append(compare(merged, "A_all_order_items", "A_Rev"))
        recon_summary.append(compare(merged, "B_non_cancelled", "B_Rev"))
        recon_summary.append(compare(merged, "C_delivered_or_returned", "C_Rev"))
        recon_summary.append(compare(merged, "D_payments_all", "D_Rev"))
        recon_summary.append(compare(merged, "E_payments_non_cancelled", "E_Rev"))
        
        pd.DataFrame(recon_summary).to_csv(f"{out_dir}/validation_002_sales_reconciliation_summary.csv", index=False)
        
        # Save daily mismatches for B as a representative
        merged["B_diff_abs"] = (merged["B_Rev"] - merged["Revenue"]).abs()
        merged.sort_values("B_diff_abs", ascending=False).to_csv(f"{out_dir}/validation_002_sales_reconciliation_daily.csv", index=False)

    # 8. Payment versus line-item order total audit
    print("Running 8. Payment vs Line-Item audit...")
    if not order_items.empty and not payments.empty and not products.empty and not orders.empty:
        oi_merged = order_items.merge(products[["product_id", "cogs"]], on="product_id", how="left")
        oi_merged["item_revenue"] = oi_merged["quantity"] * oi_merged["unit_price"]
        oi_merged["item_cogs"] = oi_merged["quantity"] * oi_merged["cogs"]
        
        agg_oi = oi_merged.groupby("order_id").agg(
            item_revenue_sum=("item_revenue", "sum"),
            item_discount_sum=("discount_amount", "sum"),
            item_cogs_sum=("item_cogs", "sum"),
            line_count=("order_id", "size")
        ).reset_index()
        
        pay_merged = payments.groupby("order_id")["payment_value"].sum().reset_index()
        comp = agg_oi.merge(pay_merged, on="order_id", how="inner").merge(orders[["order_id", "order_status"]], on="order_id")
        comp["gap"] = comp["payment_value"] - comp["item_revenue_sum"]
        comp["gap_ratio"] = comp["payment_value"] / comp["item_revenue_sum"].replace(0, np.nan)
        comp["abs_gap"] = comp["gap"].abs()
        
        pct_exact = (comp["abs_gap"] < 0.01).mean() * 100
        pct_01 = ((comp["abs_gap"] / comp["item_revenue_sum"]) <= 0.001).mean() * 100
        pct_1 = ((comp["abs_gap"] / comp["item_revenue_sum"]) <= 0.01).mean() * 100
        pct_5 = ((comp["abs_gap"] / comp["item_revenue_sum"]) <= 0.05).mean() * 100
        
        summary_pay = [{
            "metric": "pct_exact", "value": pct_exact
        }, {
            "metric": "pct_within_0.1%", "value": pct_01
        }, {
            "metric": "pct_within_1%", "value": pct_1
        }, {
            "metric": "pct_within_5%", "value": pct_5
        }]
        
        for status in comp["order_status"].unique():
            s_comp = comp[comp["order_status"] == status]
            summary_pay.append({"metric": f"mean_gap_{status}", "value": s_comp["gap"].mean()})
            summary_pay.append({"metric": f"median_gap_{status}", "value": s_comp["gap"].median()})
            
        pd.DataFrame(summary_pay).to_csv(f"{out_dir}/validation_002_payment_item_reconciliation_summary.csv", index=False)
        comp.sort_values("abs_gap", ascending=False).head(20).to_csv(f"{out_dir}/validation_002_payment_item_reconciliation_top_gaps.csv", index=False)

    # 9. Web traffic grain audit
    print("Running 9. Web traffic audit...")
    wt_summary = []
    if not web_traffic.empty:
        wt_summary.append({"metric": "row_count", "value": len(web_traffic)})
        wt_summary.append({"metric": "date_min", "value": web_traffic["date"].min()})
        wt_summary.append({"metric": "date_max", "value": web_traffic["date"].max()})
        wt_summary.append({"metric": "distinct_sources", "value": web_traffic["traffic_source"].nunique()})
        wt_summary.append({"metric": "date_unique", "value": web_traffic["date"].is_unique})
        pd.DataFrame(wt_summary).to_csv(f"{out_dir}/validation_002_web_traffic_grain_profile.csv", index=False)

    # 10. Inventory grain and health audit
    print("Running 10. Inventory grain audit...")
    inv_health = []
    if not inventory.empty:
        dups = inventory.duplicated(subset=["snapshot_date", "product_id"]).sum()
        u_prod = inventory["product_id"].nunique()
        min_date = inventory["snapshot_date"].min()
        max_date = inventory["snapshot_date"].max()
        
        inventory["health"] = "Healthy"
        inventory.loc[inventory["stockout_flag"] == 1, "health"] = "Stockout"
        inventory.loc[inventory["overstock_flag"] == 1, "health"] = "Overstock"
        inventory.loc[(inventory["reorder_flag"] == 1) & (inventory["health"] == "Healthy"), "health"] = "Reorder Needed"
        
        health_dist = inventory["health"].value_counts().to_dict()
        
        inv_health.extend([
            {"metric": "duplicates_by_date_product", "value": dups},
            {"metric": "unique_products", "value": u_prod},
            {"metric": "snapshot_date_min", "value": min_date},
            {"metric": "snapshot_date_max", "value": max_date},
        ])
        for k, v in health_dist.items():
            inv_health.append({"metric": f"health_status_{k}", "value": v})
            
        pd.DataFrame(inv_health).to_csv(f"{out_dir}/validation_002_inventory_health_summary.csv", index=False)

    # 11. Promotion semantic audit
    print("Running 11. Promotion semantic audit...")
    promo_aud = []
    if not order_items.empty and not promotions.empty:
        p1_nulls = order_items["promo_id"].isnull().sum()
        p1_non = order_items["promo_id"].notnull().sum()
        p2_nulls = order_items["promo_id_2"].isnull().sum()
        p2_non = order_items["promo_id_2"].notnull().sum()
        both = order_items[order_items["promo_id"].notnull() & order_items["promo_id_2"].notnull()].shape[0]
        
        promo_aud.extend([
            {"metric": "promo_id_nulls", "value": p1_nulls},
            {"metric": "promo_id_non_nulls", "value": p1_non},
            {"metric": "promo_id_2_nulls", "value": p2_nulls},
            {"metric": "promo_id_2_non_nulls", "value": p2_non},
            {"metric": "both_promos_non_null", "value": both},
        ])
        
        no_promo_disc = order_items[order_items["promo_id"].isnull()]["discount_amount"].mean()
        promo_disc = order_items[order_items["promo_id"].notnull()]["discount_amount"].mean()
        
        promo_aud.append({"metric": "mean_discount_no_promo", "value": no_promo_disc})
        promo_aud.append({"metric": "mean_discount_with_promo", "value": promo_disc})
        
        pd.DataFrame(promo_aud).to_csv(f"{out_dir}/validation_002_promotion_semantic_audit.csv", index=False)

    print("Semantic Audit Completed Successfully.")

if __name__ == "__main__":
    run_semantic_audit()
