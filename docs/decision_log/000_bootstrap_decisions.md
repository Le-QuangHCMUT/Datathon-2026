# Bootstrap Decisions

## 1. Project Initialization
- Created standard project structure (`notebooks/`, `src/`, `configs/`, `artifacts/`, `docs/`, `tests/`).
- Added `README.md` and `.gitignore` stubs.
- Retained all raw data files in `data/` and root without modification.

## 2. File Manifest & Discrepancies
- **Discrepancy 1**: Official documentation mentions 15 files including `sales_train.csv` and `sales_test.csv`.
- **Finding**: There are only 14 files total. `sales_train.csv` and `sales_test.csv` are absent. We found a single `sales.csv` file in `data/` encompassing dates from `2012-07-04` to `2022-12-31`.
- **Decision**: We will use `sales.csv` as our primary training dataset for macroeconomic trends and target values, and ignore the mention of the two split files. `sample_submission.csv` is correctly present in the root folder with the expected horizon `2023-01-01` to `2024-07-01`.

## 3. Data Audit
- Executed `bootstrap_audit.py` to extract shape, schema, null counts, date ranges, and key uniqueness.
- Saved diagnostics securely to `artifacts/diagnostics/`.
- No files were corrupted or overwritten.

## 4. Next Steps
- Validate line-item revenue summation against the daily `sales.csv` aggregates.
- Investigate missing values (if any) identified in the null diagnostics.
- Validate `sample_submission.csv` date sequence and constraints for the forecasting model.
