import os
import pandas as pd
import numpy as np

def audit_data():
    base_dir = "."
    dirs_to_check = [base_dir, os.path.join(base_dir, "data")]
    csv_files = []

    for d in dirs_to_check:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".csv"):
                    csv_files.append(os.path.normpath(os.path.join(d, f)))

    manifest_records = []
    schema_records = []
    null_records = []
    key_uniqueness_records = []
    date_range_records = []

    for path in csv_files:
        try:
            size_bytes = os.path.getsize(path)
            manifest_records.append({
                "file_path": path,
                "size_bytes": size_bytes,
                "status": "Found",
                "error": ""
            })

            # Read safely
            df = pd.read_csv(path)
            row_count, col_count = df.shape
            mem_usage = df.memory_usage(deep=True).sum()
            duplicates = df.duplicated().sum()

            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                null_pct = null_count / row_count if row_count > 0 else 0
                sample_vals = str(df[col].dropna().head(3).tolist())
                
                schema_records.append({
                    "file_path": path,
                    "column_name": col,
                    "dtype": dtype,
                    "sample_values": sample_vals
                })

                null_records.append({
                    "file_path": path,
                    "column_name": col,
                    "null_count": null_count,
                    "null_pct": null_pct
                })

                # Key check
                if "id" in col.lower() or "zip" in col.lower() or "key" in col.lower() or col.lower() == "date":
                    is_unique = df[col].is_unique
                    n_unique = df[col].nunique()
                    key_uniqueness_records.append({
                        "file_path": path,
                        "column_name": col,
                        "is_unique": is_unique,
                        "n_unique": n_unique,
                        "total_rows": row_count
                    })

                # Date check
                if "date" in col.lower() or "timestamp" in col.lower() or df[col].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").any():
                    try:
                        parsed_dates = pd.to_datetime(df[col], errors='coerce')
                        if parsed_dates.notnull().sum() > 0:
                            min_date = parsed_dates.min()
                            max_date = parsed_dates.max()
                            date_range_records.append({
                                "file_path": path,
                                "column_name": col,
                                "min_date": min_date,
                                "max_date": max_date,
                                "is_continuous": "TBD", # We can refine this if needed
                                "is_sorted": df[col].is_monotonic_increasing if not df[col].isnull().all() else False
                            })
                    except Exception:
                        pass
        except Exception as e:
            manifest_records[-1]["status"] = "Error"
            manifest_records[-1]["error"] = str(e)

    pd.DataFrame(manifest_records).to_csv("artifacts/diagnostics/file_manifest.csv", index=False)
    pd.DataFrame(schema_records).to_csv("artifacts/diagnostics/schema_summary.csv", index=False)
    pd.DataFrame(null_records).to_csv("artifacts/diagnostics/null_summary.csv", index=False)
    pd.DataFrame(key_uniqueness_records).to_csv("artifacts/diagnostics/key_uniqueness_summary.csv", index=False)
    pd.DataFrame(date_range_records).to_csv("artifacts/diagnostics/date_range_summary.csv", index=False)
    
    print("Diagnostics saved successfully.")

if __name__ == "__main__":
    audit_data()
