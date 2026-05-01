"""FORECAST-024: Tết Feature Surgery

Objective
- Exploit FORECAST-023 signal: removing Tết features improved FoldA MAE.
- Build controlled variants of the FORECAST-020 reference architecture:
  - no_tet_all
  - soft_tet_only
  - post_tet_only (broad post-Tết recovery)
- Keep calibration neighborhood centered on CR=1.28, CC=1.32.

Outputs
- Tables/figures/candidates/submissions under artifacts/
- Report under docs/model_log/
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join("src", "datathon", "forecasting"))
from forecast_020_reference_pipeline import build_features, TET_TS, PROMOS, LGB_PARAMS, get_weights  # noqa: E402


SEED = 42
np.random.seed(SEED)

CR_BASE, CC_BASE = 1.28, 1.32
ALPHA = 0.60

FOLDS = [
    ("FoldA", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("FoldB", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("FoldC", "2021-06-30", "2021-07-01", "2022-06-30"),
]

OUT_TABLES = "artifacts/tables"
OUT_FIGS = "artifacts/figures"
OUT_FORECASTS = "artifacts/forecasts"
OUT_SUBS = "artifacts/submissions"
OUT_REPORT = "docs/model_log/forecast_024_tet_feature_surgery_report.md"


def _ensure_dirs() -> None:
    for p in [OUT_TABLES, OUT_FIGS, OUT_FORECASTS, OUT_SUBS, os.path.dirname(OUT_REPORT)]:
        os.makedirs(p, exist_ok=True)


def _df_to_markdown(df: pd.DataFrame, index: bool = True, floatfmt: str = ",.1f") -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table.

    Avoids pandas.DataFrame.to_markdown() which depends on optional 'tabulate'.
    """

    if df is None or len(df) == 0:
        return "(empty)"

    tdf = df.copy()
    if index:
        tdf = tdf.reset_index()

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return ""
            return format(float(v), floatfmt)
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,d}"
        return str(v)

    cols = list(tdf.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in tdf.iterrows():
        rows.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def _assert_daily_continuous(idx: pd.DatetimeIndex, name: str) -> None:
    idx = pd.DatetimeIndex(idx).sort_values()
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    missing = full.difference(idx)
    if len(missing) > 0:
        raise AssertionError(
            f"{name} is not daily-continuous: missing {len(missing)} days; "
            f"examples={missing[:5].strftime('%Y-%m-%d').tolist()}"
        )


def _validate_inputs(sales: pd.DataFrame, sample: pd.DataFrame, best: pd.DataFrame) -> pd.DatetimeIndex:
    if "Date" not in sales.columns:
        raise AssertionError("sales.csv must contain Date")
    if "Revenue" not in sales.columns or "COGS" not in sales.columns:
        raise AssertionError("sales.csv must contain Revenue and COGS")
    if "Date" not in sample.columns:
        raise AssertionError("sample_submission.csv must contain Date")

    sales = sales.copy()
    sales["Date"] = pd.to_datetime(sales["Date"])
    sales = sales.sort_values("Date").set_index("Date")

    if not (sales["Revenue"] > 0).all():
        raise AssertionError("Revenue must be > 0 for log-model training")
    if not (sales["COGS"] > 0).all():
        raise AssertionError("COGS must be > 0 for log-model training")

    _assert_daily_continuous(sales.index, "sales")

    sample = sample[["Date"]].copy()  # hard guard: never touch sample targets
    sample["Date"] = pd.to_datetime(sample["Date"])
    if len(sample) != 548:
        raise AssertionError(f"Sample must have 548 rows; got {len(sample)}")
    if not sample["Date"].is_monotonic_increasing:
        raise AssertionError("sample_submission Date must be sorted")
    if not sample["Date"].is_unique:
        raise AssertionError("sample_submission Date must be unique")
    fdates = pd.DatetimeIndex(sample["Date"])
    if fdates.min() != pd.Timestamp("2023-01-01") or fdates.max() != pd.Timestamp("2024-07-01"):
        raise AssertionError(
            f"Sample horizon mismatch: expected 2023-01-01..2024-07-01; got {fdates.min()}..{fdates.max()}"
        )
    _assert_daily_continuous(fdates, "sample_submission")

    best = best.copy()
    best["Date"] = pd.to_datetime(best["Date"])
    best = best.set_index("Date")
    if not (best.index == fdates).all():
        raise AssertionError("Current best submission Date order does not match sample_submission")
    if not (best["Revenue"] > 0).all() or not (best["COGS"] >= 0).all():
        raise AssertionError("Current best submission must satisfy Revenue>0 and COGS>=0")

    return fdates


def _tet_segments(diff: pd.Series) -> pd.Series:
    """Non-overlapping audit segments matching the task spec."""
    seg = pd.Series("other", index=diff.index, dtype=object)
    seg[(diff >= -45) & (diff <= -15)] = "pre_tet_-45_-15"
    seg[(diff >= -14) & (diff <= -1)] = "pre_tet_-14_-1"
    seg[diff == 0] = "tet_day_0"
    seg[(diff >= 1) & (diff <= 14)] = "post_tet_1_14"
    seg[(diff >= 15) & (diff <= 30)] = "post_tet_15_30"
    seg[(diff >= 31) & (diff <= 45)] = "post_tet_31_45"
    return seg


def _run_tet_audit(sales: pd.DataFrame, feat_all: pd.DataFrame) -> pd.DataFrame:
    diff = pd.Series(feat_all["tet_days_diff"].values, index=sales.index)
    seg = _tet_segments(diff)

    df = sales[["Revenue", "COGS"]].copy()
    df["year"] = df.index.year
    df["tet_days_diff"] = diff.values
    df["segment"] = seg.values
    df = df[(df["tet_days_diff"].abs() <= 45)].copy()

    year_mean = sales.groupby(sales.index.year)[["Revenue", "COGS"]].mean().rename(
        columns={"Revenue": "Revenue_year_mean", "COGS": "COGS_year_mean"}
    )
    df = df.join(year_mean, on="year")
    df["Revenue_pct_vs_year_mean"] = (df["Revenue"] / df["Revenue_year_mean"] - 1.0) * 100
    df["COGS_pct_vs_year_mean"] = (df["COGS"] / df["COGS_year_mean"] - 1.0) * 100

    by_year = (
        df.groupby(["year", "segment"], as_index=False)
        .agg(
            n_days=("Revenue", "size"),
            Revenue_mean=("Revenue", "mean"),
            COGS_mean=("COGS", "mean"),
            Revenue_pct_vs_year_mean=("Revenue_pct_vs_year_mean", "mean"),
            COGS_pct_vs_year_mean=("COGS_pct_vs_year_mean", "mean"),
        )
        .sort_values(["year", "segment"])
    )
    by_year.insert(0, "scope", "year")

    overall = (
        df.groupby(["segment"], as_index=False)
        .agg(
            n_days=("Revenue", "size"),
            Revenue_mean=("Revenue", "mean"),
            COGS_mean=("COGS", "mean"),
            Revenue_pct_vs_year_mean=("Revenue_pct_vs_year_mean", "mean"),
            COGS_pct_vs_year_mean=("COGS_pct_vs_year_mean", "mean"),
        )
        .sort_values(["segment"])
    )
    overall.insert(0, "scope", "overall")
    overall.insert(1, "year", np.nan)

    out = pd.concat([by_year, overall], ignore_index=True)
    out.to_csv(os.path.join(OUT_TABLES, "forecast_024_tet_feature_audit.csv"), index=False)

    # Figure: mean profile by diff + segment bars
    prof = df.groupby("tet_days_diff")[["Revenue", "COGS"]].mean().sort_index()
    roll = prof.rolling(7, center=True, min_periods=1).mean()

    plt.figure(figsize=(14, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(roll.index, roll["Revenue"], label="Revenue (7d roll)")
    ax1.plot(roll.index, roll["COGS"], label="COGS (7d roll)")
    ax1.axvline(0, color="black", lw=1)
    ax1.set_title("Avg Revenue/COGS around Tết (±45d)")
    ax1.set_xlabel("tet_days_diff")
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    seg_order = [
        "pre_tet_-45_-15",
        "pre_tet_-14_-1",
        "tet_day_0",
        "post_tet_1_14",
        "post_tet_15_30",
        "post_tet_31_45",
    ]
    overall_plot = overall.set_index("segment").reindex(seg_order)
    overall_plot[["Revenue_mean", "COGS_mean"]].plot(kind="bar", ax=ax2)
    ax2.set_title("Segment means (overall)")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "forecast_024_tet_window_historical_effect.png"))
    plt.close()

    return out


def _apply_variant(feat: pd.DataFrame, variant: str) -> pd.DataFrame:
    f = feat.copy()
    v = variant
    if v in {"full", "full_reference"}:
        return f

    tet_cols = [c for c in f.columns if c.startswith("tet_") or c.startswith("post_tet_")]
    if v in {"no_tet", "no_tet_all", "no_lunar_distance_keep_month"}:
        return f.drop(columns=tet_cols, errors="ignore")

    if v in {"soft_tet", "soft_tet_only"}:
        drop = ["tet_days_diff", "tet_on", "tet_in_7"]
        return f.drop(columns=[c for c in drop if c in f.columns], errors="ignore")

    if v in {"post_tet_only", "post_tet"}:
        # Broad recovery features derived from tet_days_diff; deterministic for future dates.
        if "tet_days_diff" in f.columns:
            d = pd.Series(f["tet_days_diff"].values)
            f["post_tet_14"] = ((d >= 1) & (d <= 14)).astype(int).values
            f["post_tet_30"] = ((d >= 1) & (d <= 30)).astype(int).values
        drop = [
            "tet_days_diff",
            "tet_in_7",
            "tet_in_14",
            "tet_before_7",
            "tet_on",
        ]
        f = f.drop(columns=[c for c in drop if c in f.columns], errors="ignore")
        return f

    raise ValueError(f"Unknown variant: {variant}")


def _align_columns(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> pd.DataFrame:
    return X_te.reindex(columns=X_tr.columns, fill_value=0.0)


def _lgb_train(y: pd.Series, Xtr: pd.DataFrame, Xte: pd.DataFrame, dtr: pd.DatetimeIndex, qb=None):
    w = get_weights(dtr, qb)
    split = max(len(Xtr) - 180, 180)
    dt = lgb.Dataset(Xtr.iloc[:split], np.log(y.iloc[:split]), weight=w[:split])
    dv = lgb.Dataset(Xtr.iloc[split:], np.log(y.iloc[split:]), weight=w[split:], reference=dt)
    m = lgb.train(
        LGB_PARAMS,
        dt,
        2000,
        valid_sets=[dt, dv],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    m2 = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, np.log(y), weight=w), m.best_iteration)
    return np.exp(m2.predict(Xte)), m2


def _ridge_train(y: pd.Series, Xtr: pd.DataFrame, Xte: pd.DataFrame):
    sc = StandardScaler()
    m = Ridge(alpha=3.0).fit(sc.fit_transform(Xtr), np.log(y))
    return np.exp(m.predict(sc.transform(Xte))), m, sc


def _prophet_train(y: pd.Series, dtr: pd.DatetimeIndex, dte: pd.DatetimeIndex, Xtr: pd.DataFrame, Xte: pd.DataFrame):
    mask = dtr >= "2020-01-01"
    pcols = [f"promo_{p['name']}" for p in PROMOS]
    df = pd.DataFrame({"ds": dtr[mask], "y": np.log(y[mask].values)})
    for c in pcols:
        df[c] = Xtr.loc[mask, c].values
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    for c in pcols:
        m.add_regressor(c)
    m.fit(df)
    df_f = pd.DataFrame({"ds": dte})
    for c in pcols:
        df_f[c] = Xte[c].values
    return np.exp(m.predict(df_f)["yhat"].values)


def _tet_window_mask(feat: pd.DataFrame) -> np.ndarray:
    # Task-focused window: includes pre (-14..-1), tet day, and post recovery through +30.
    if "tet_days_diff" not in feat.columns:
        return np.zeros(len(feat), dtype=bool)
    d = feat["tet_days_diff"].values.astype(float)
    return (d >= -14) & (d <= 30)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": mae, "Bias": bias}


def _eval_masks(dates: pd.DatetimeIndex, tet_mask: np.ndarray) -> dict:
    q1 = (dates.quarter == 1)
    q1_non_tet = q1 & (~tet_mask)
    return {
        "all": np.ones(len(dates), dtype=bool),
        "q1": q1,
        "tet_window": tet_mask,
        "q1_non_tet": q1_non_tet,
    }


def _run_reference_architecture(
    yR: pd.Series,
    yC: pd.Series,
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    dtr: pd.DatetimeIndex,
    dte: pd.DatetimeIndex,
    prophet_cache=None,
):
    # Ridge
    rRidgeRev, _, _ = _ridge_train(yR, Xtr, Xte)
    rRidgeCogs, _, _ = _ridge_train(yC, Xtr, Xte)
    # LGB
    rLgbRev, _ = _lgb_train(yR, Xtr, Xte, dtr)
    rLgbCogs, _ = _lgb_train(yC, Xtr, Xte, dtr)
    # Prophet (cache-safe)
    if prophet_cache is not None and "rev" in prophet_cache and "cog" in prophet_cache:
        rProRev, rProCogs = prophet_cache["rev"], prophet_cache["cog"]
    else:
        rProRev = _prophet_train(yR, dtr, dte, Xtr, Xte)
        rProCogs = _prophet_train(yC, dtr, dte, Xtr, Xte)
        if prophet_cache is not None:
            prophet_cache["rev"], prophet_cache["cog"] = rProRev, rProCogs

    # Q-specialists
    rQRev = np.zeros(len(Xte))
    rQCogs = np.zeros(len(Xte))
    for q in [1, 2, 3, 4]:
        pR, _ = _lgb_train(yR, Xtr, Xte, dtr, qb=q)
        pC, _ = _lgb_train(yC, Xtr, Xte, dtr, qb=q)
        mk = (dte.quarter == q)
        rQRev[mk] = pR[mk]
        rQCogs[mk] = pC[mk]

    blendR = ALPHA * rQRev + (1 - ALPHA) * rLgbRev
    blendC = ALPHA * rQCogs + (1 - ALPHA) * rLgbCogs
    rawR = 0.10 * rRidgeRev + 0.10 * rProRev + 0.80 * blendR
    rawC = 0.10 * rRidgeCogs + 0.10 * rProCogs + 0.80 * blendC

    return {
        "Ridge": (rRidgeRev, rRidgeCogs),
        "LGB": (rLgbRev, rLgbCogs),
        "Prophet": (rProRev, rProCogs),
        "QSpec": (rQRev, rQCogs),
        "Ensemble": (rawR, rawC),
    }


def _clip_preds(rev: np.ndarray, cogs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(rev, 1.0), np.maximum(cogs, 0.0)


def _write_submission(path: str, df: pd.DataFrame) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    df.to_csv(path, index=False)


def _load_or_write_submission(path: str, expected_df: pd.DataFrame, fdates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a submission once; on reruns, load+validate instead of overwriting."""

    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Date" not in df.columns or "Revenue" not in df.columns or "COGS" not in df.columns:
            raise AssertionError(f"Existing submission missing required columns: {path}")
        df["Date"] = pd.to_datetime(df["Date"])
        if len(df) != 548:
            raise AssertionError(f"Existing submission rowcount != 548: {path}")
        if not (pd.DatetimeIndex(df["Date"]) == fdates).all():
            raise AssertionError(f"Existing submission Date order mismatch vs sample: {path}")
        if df.isna().sum().sum() != 0:
            raise AssertionError(f"Existing submission has NaN: {path}")
        if not (df["Revenue"] > 0).all() or not (df["COGS"] >= 0).all():
            raise AssertionError(f"Existing submission violates bounds: {path}")
        return df

    expected_df.to_csv(path, index=False)
    return expected_df


def main() -> None:
    print("FORECAST-024: Tết Feature Surgery")
    _ensure_dirs()

    print("Loading inputs...")
    sales_raw = pd.read_csv("data/sales.csv")
    sample_raw = pd.read_csv("sample_submission.csv")
    best_raw = pd.read_csv("artifacts/submissions/submission_forecast_022_cr128_cc132.csv")

    # Validate + canonicalize
    fdates = _validate_inputs(sales_raw, sample_raw, best_raw)
    sales = sales_raw.copy()
    sales["Date"] = pd.to_datetime(sales["Date"])
    sales = sales.sort_values("Date").set_index("Date")
    best = best_raw.copy()
    best["Date"] = pd.to_datetime(best["Date"])
    best = best.set_index("Date")

    print(f"Sales: {sales.index.min().date()} → {sales.index.max().date()} ({len(sales)} rows)")
    print(f"Sample horizon: {fdates.min().date()} → {fdates.max().date()} ({len(fdates)} rows)")

    print("Building reference features...")
    feat_all = build_features(sales.index)
    feat_future = build_features(fdates)
    if feat_all.isna().sum().sum() != 0 or feat_future.isna().sum().sum() != 0:
        raise AssertionError("NaN found in built features")

    print("Running Tết feature audit...")
    tet_audit = _run_tet_audit(sales, feat_all)
    print(f"Tet audit rows: {len(tet_audit)}")

    variants = [
        ("full_reference", "Original reference features"),
        ("no_tet_all", "Drop all tet_* features"),
        ("soft_tet_only", "Drop tet_days_diff/tet_on/tet_in_7; keep broad flags"),
        ("post_tet_only", "Keep only post-Tết recovery features (post_tet_14/post_tet_30 + tet_after_7)"),
    ]

    print("Running fold ablations (reference architecture)...")
    ablation_rows = []
    model_rows = []

    for vname, vdesc in variants:
        print(f"  Variant: {vname}")
        for fold_name, tend, vst, vend in FOLDS:
            mtr = sales.index <= tend
            mva = (sales.index >= vst) & (sales.index <= vend)

            Xtr0, Xva0 = feat_all.loc[mtr], feat_all.loc[mva]
            dtr, dva = sales.index[mtr], sales.index[mva]
            yRtr, yRva = sales.loc[mtr, "Revenue"], sales.loc[mva, "Revenue"]
            yCtr, yCva = sales.loc[mtr, "COGS"], sales.loc[mva, "COGS"]

            # Apply variant consistently to train/val
            Xtr = _apply_variant(Xtr0, vname)
            Xva = _apply_variant(Xva0, vname)
            Xva = _align_columns(Xtr, Xva)
            if list(Xtr.columns) != list(Xva.columns):
                raise AssertionError(f"Feature mismatch after variant {vname} on {fold_name}")

            # Masks for breakdown metrics (use tet_days_diff from original reference features)
            tet_mask = _tet_window_mask(Xva0)
            masks = _eval_masks(dva, tet_mask)

            prophet_cache = {}
            preds = _run_reference_architecture(yRtr, yCtr, Xtr, Xva, dtr, dva, prophet_cache=prophet_cache)

            # Base model metrics table
            for model_name, (raw_rev, raw_cog) in preds.items():
                fin_rev, fin_cog = _clip_preds(raw_rev * CR_BASE, raw_cog * CC_BASE)

                for scope, mk in masks.items():
                    if mk.sum() == 0:
                        continue
                    r = _metrics(yRva.values[mk], fin_rev[mk])
                    c = _metrics(yCva.values[mk], fin_cog[mk])
                    model_rows.append(
                        {
                            "variant": vname,
                            "variant_desc": vdesc,
                            "fold": fold_name,
                            "model": model_name,
                            "scope": scope,
                            "n": int(mk.sum()),
                            "Rev_MAE": r["MAE"],
                            "COGS_MAE": c["MAE"],
                            "Combined_MAE": r["MAE"] + c["MAE"],
                            "Bias_Rev": r["Bias"],
                            "Bias_COGS": c["Bias"],
                            "cr": CR_BASE,
                            "cc": CC_BASE,
                            "alpha": ALPHA,
                            "seed": SEED,
                        }
                    )

            # Ensemble ablation metrics (fold-level, required breakdown)
            raw_rev, raw_cog = preds["Ensemble"]
            fin_rev, fin_cog = _clip_preds(raw_rev * CR_BASE, raw_cog * CC_BASE)

            def _mae(mk):
                if mk.sum() == 0:
                    return np.nan, np.nan
                return (
                    float(mean_absolute_error(yRva.values[mk], fin_rev[mk])),
                    float(mean_absolute_error(yCva.values[mk], fin_cog[mk])),
                )

            rev_all, cog_all = _mae(masks["all"])
            rev_q1, cog_q1 = _mae(masks["q1"])
            rev_tet, cog_tet = _mae(masks["tet_window"])
            rev_q1_nt, cog_q1_nt = _mae(masks["q1_non_tet"])

            ablation_rows.append(
                {
                    "variant": vname,
                    "variant_desc": vdesc,
                    "fold": fold_name,
                    "cr": CR_BASE,
                    "cc": CC_BASE,
                    "alpha": ALPHA,
                    "seed": SEED,
                    "n_val": int(len(dva)),
                    "n_q1": int(masks["q1"].sum()),
                    "n_tet_window": int(masks["tet_window"].sum()),
                    "n_q1_non_tet": int(masks["q1_non_tet"].sum()),
                    "Rev_MAE": rev_all,
                    "COGS_MAE": cog_all,
                    "Combined_MAE": rev_all + cog_all,
                    "Q1_Rev_MAE": rev_q1,
                    "Q1_COGS_MAE": cog_q1,
                    "Q1_Combined_MAE": (rev_q1 + cog_q1) if np.isfinite(rev_q1) and np.isfinite(cog_q1) else np.nan,
                    "TetWin_Rev_MAE": rev_tet,
                    "TetWin_COGS_MAE": cog_tet,
                    "TetWin_Combined_MAE": (rev_tet + cog_tet) if np.isfinite(rev_tet) and np.isfinite(cog_tet) else np.nan,
                    "Q1_NonTet_Rev_MAE": rev_q1_nt,
                    "Q1_NonTet_COGS_MAE": cog_q1_nt,
                    "Q1_NonTet_Combined_MAE": (rev_q1_nt + cog_q1_nt)
                    if np.isfinite(rev_q1_nt) and np.isfinite(cog_q1_nt)
                    else np.nan,
                    "Bias_Rev": float(np.mean(fin_rev - yRva.values)),
                    "Bias_COGS": float(np.mean(fin_cog - yCva.values)),
                }
            )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(OUT_TABLES, "forecast_024_ablation_metrics.csv"), index=False)
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(os.path.join(OUT_TABLES, "forecast_024_model_metrics.csv"), index=False)

    print("Ablation (Ensemble) summary: mean across folds")
    summary = (
        ablation_df.groupby("variant")[
            [
                "Rev_MAE",
                "COGS_MAE",
                "Combined_MAE",
                "Q1_Combined_MAE",
                "TetWin_Combined_MAE",
                "Q1_NonTet_Combined_MAE",
            ]
        ]
        .mean()
        .sort_values("Combined_MAE")
    )
    print(summary.to_string(float_format=lambda x: f"{x:,.1f}"))

    # Train final future models for selected variants
    print("\nTraining final full-history models...")
    final_variants = ["no_tet_all", "soft_tet_only", "post_tet_only"]

    # Prophet doesn't depend on Tết features (promo-only). Train once.
    prophet_cache_full = {}
    prophet_full_rev = _prophet_train(sales["Revenue"], sales.index, fdates, feat_all, feat_future)
    prophet_full_cog = _prophet_train(sales["COGS"], sales.index, fdates, feat_all, feat_future)
    prophet_cache_full["rev"], prophet_cache_full["cog"] = prophet_full_rev, prophet_full_cog

    raw_future = {}
    for vname in final_variants:
        print(f"  Final variant: {vname}")
        Xtr = _apply_variant(feat_all, vname)
        Xte = _apply_variant(feat_future, vname)
        Xte = _align_columns(Xtr, Xte)
        if list(Xtr.columns) != list(Xte.columns):
            raise AssertionError(f"Train/future feature mismatch for variant {vname}")

        preds = _run_reference_architecture(
            sales["Revenue"],
            sales["COGS"],
            Xtr,
            Xte,
            sales.index,
            fdates,
            prophet_cache=prophet_cache_full,
        )
        raw_future[vname] = preds["Ensemble"]

    # Candidate generation (future)
    print("\nGenerating submissions...")
    win_rev = best["Revenue"].values
    win_cog = best["COGS"].values

    # Calibrated & clipped candidate arrays
    candidates = {}
    raw_rev_nt, raw_cog_nt = raw_future["no_tet_all"]
    raw_rev_soft, raw_cog_soft = raw_future["soft_tet_only"]
    raw_rev_post, raw_cog_post = raw_future["post_tet_only"]

    def _cand(rev_raw, cog_raw, cr, cc):
        r, c = _clip_preds(rev_raw * cr, cog_raw * cc)
        return r, c

    candidates["no_tet_cr128_cc132"] = _cand(raw_rev_nt, raw_cog_nt, 1.28, 1.32)
    candidates["no_tet_cr127_cc132"] = _cand(raw_rev_nt, raw_cog_nt, 1.27, 1.32)
    candidates["no_tet_cr129_cc132"] = _cand(raw_rev_nt, raw_cog_nt, 1.29, 1.32)
    candidates["soft_tet_cr128_cc132"] = _cand(raw_rev_soft, raw_cog_soft, 1.28, 1.32)
    candidates["post_tet_only_cr128_cc132"] = _cand(raw_rev_post, raw_cog_post, 1.28, 1.32)
    # Blend uses the actual current best for future horizon (safe LB perturbation)
    r_nt, c_nt = candidates["no_tet_cr128_cc132"]
    candidates["no_tet_blend_current25"] = (
        np.maximum(0.75 * win_rev + 0.25 * r_nt, 1.0),
        np.maximum(0.75 * win_cog + 0.25 * c_nt, 0.0),
    )

    # Save submissions (no overwrite)
    sub_dfs = {}
    for cname, (rev, cog) in candidates.items():
        df = pd.DataFrame({"Date": fdates.values, "Revenue": rev, "COGS": cog})
        if len(df) != 548 or df.isna().sum().sum() != 0:
            raise AssertionError(f"Submission shape/NaN failure for {cname}")
        if not (df["Revenue"] > 0).all() or not (df["COGS"] >= 0).all():
            raise AssertionError(f"Submission bounds failure for {cname}")
        sub_path = os.path.join(OUT_SUBS, f"submission_forecast_024_{cname}.csv")
        df_saved = _load_or_write_submission(sub_path, df, fdates)
        sub_dfs[cname] = df_saved

    # Forecast candidates file (wide)
    out_df = pd.DataFrame({"Date": fdates.values, "Rev_current_best": win_rev, "Cog_current_best": win_cog})
    for cname, df in sub_dfs.items():
        out_df[f"Rev_{cname}"] = df["Revenue"].values
        out_df[f"Cog_{cname}"] = df["COGS"].values
    out_df.to_csv(os.path.join(OUT_FORECASTS, "forecast_024_future_candidates.csv"), index=False)

    # Diff vs current best (daily)
    diff_rows = []
    for cname, df in sub_dfs.items():
        rev = df["Revenue"].values
        cog = df["COGS"].values
        rev_pct = (rev - win_rev) / win_rev * 100
        cog_pct = (cog - win_cog) / win_cog * 100
        for i, dt in enumerate(fdates):
            diff_rows.append(
                {
                    "candidate": cname,
                    "Date": dt,
                    "quarter": int(dt.quarter),
                    "month": int(dt.month),
                    "Revenue": float(rev[i]),
                    "COGS": float(cog[i]),
                    "best_Revenue": float(win_rev[i]),
                    "best_COGS": float(win_cog[i]),
                    "rev_pct_diff_vs_best": float(rev_pct[i]),
                    "cog_pct_diff_vs_best": float(cog_pct[i]),
                    "abs_rev_pct_diff_vs_best": float(abs(rev_pct[i])),
                }
            )
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(OUT_TABLES, "forecast_024_diff_vs_current_best.csv"), index=False)

    # Candidate diagnostics + recommendation
    print("\nBuilding candidate manifest...")
    base_by_variant = ablation_df.groupby(["variant"])[
        [
            "Rev_MAE",
            "COGS_MAE",
            "Combined_MAE",
            "Q1_Combined_MAE",
            "TetWin_Combined_MAE",
            "Q1_NonTet_Combined_MAE",
        ]
    ].mean()

    def _risk_label(mad_daily_rev: float, pct_total_rev: float) -> str:
        if mad_daily_rev < 0.75 and abs(pct_total_rev) < 1.0:
            return "low"
        if mad_daily_rev < 2.0 and abs(pct_total_rev) < 2.0:
            return "medium"
        return "high"

    manifest_rows = []
    best_combined = float(base_by_variant.loc["full_reference", "Combined_MAE"]) if "full_reference" in base_by_variant.index else np.nan

    # Per-candidate fold-avg OOF metrics (blend uses proxy: 75% full_reference + 25% no_tet_all on folds)
    oof_cache = {}
    for vname, _ in variants:
        vdf = ablation_df[ablation_df.variant == vname]
        oof_cache[vname] = vdf.groupby("fold")[
            [
                "Rev_MAE",
                "COGS_MAE",
                "Combined_MAE",
                "Q1_Combined_MAE",
                "TetWin_Combined_MAE",
                "Q1_NonTet_Combined_MAE",
            ]
        ].mean().mean()

    # Candidate ordering required by task
    candidate_order = [
        ("no_tet_cr128_cc132", 1.28, 1.32, "no_tet_all"),
        ("no_tet_cr127_cc132", 1.27, 1.32, "no_tet_all"),
        ("no_tet_cr129_cc132", 1.29, 1.32, "no_tet_all"),
        ("soft_tet_cr128_cc132", 1.28, 1.32, "soft_tet_only"),
        ("post_tet_only_cr128_cc132", 1.28, 1.32, "post_tet_only"),
        ("no_tet_blend_current25", 1.28, 1.32, "blend_proxy"),
    ]

    # Diff summaries
    diff_sum = (
        diff_df.groupby("candidate")
        .agg(
            pct_total_rev_vs_best=("rev_pct_diff_vs_best", "mean"),
            mad_daily_rev_vs_best=("abs_rev_pct_diff_vs_best", "mean"),
            max_abs_daily_rev_vs_best=("abs_rev_pct_diff_vs_best", "max"),
        )
        .reset_index()
        .set_index("candidate")
    )

    totals = (
        diff_df.groupby("candidate")
        .agg(
            total_rev=("Revenue", "sum"),
            total_cog=("COGS", "sum"),
            best_total_rev=("best_Revenue", "sum"),
            best_total_cog=("best_COGS", "sum"),
        )
        .reset_index()
        .set_index("candidate")
    )

    # Q1 monthly diffs (Jan/Feb/Mar) for report & manifest
    q1_months = diff_df[diff_df["month"].isin([1, 2, 3])].copy()
    q1_months["ym"] = q1_months["Date"].dt.strftime("%Y-%m")
    q1_monthly = (
        q1_months.groupby(["candidate", "ym"])
        .apply(
            lambda g: (g["Revenue"].sum() - g["best_Revenue"].sum()) / g["best_Revenue"].sum() * 100
        )
        .reset_index()
        .rename(columns={0: "q1_month_rev_pct_diff"})
    )

    q1_monthly.to_csv(os.path.join(OUT_TABLES, "forecast_024_q1_monthly_diff_vs_best.csv"), index=False)

    for pri, (cname, cr, cc, source) in enumerate(candidate_order, start=1):
        rev = sub_dfs[cname]["Revenue"].values
        cog = sub_dfs[cname]["COGS"].values

        pct_total_rev = (rev.sum() - win_rev.sum()) / win_rev.sum() * 100
        pct_total_cog = (cog.sum() - win_cog.sum()) / win_cog.sum() * 100
        mad_daily = float(np.mean(np.abs(rev - win_rev) / win_rev) * 100)
        max_daily = float(np.max(np.abs(rev - win_rev) / win_rev) * 100)

        if source == "blend_proxy":
            # proxy OOF: blend fold-avg metrics of full_reference + no_tet_all
            oof = 0.75 * oof_cache["full_reference"] + 0.25 * oof_cache["no_tet_all"]
        else:
            oof = oof_cache[source]

        risk = _risk_label(mad_daily, pct_total_rev)

        manifest_rows.append(
            {
                "candidate_file": f"submission_forecast_024_{cname}.csv",
                "candidate_name": cname,
                "variant_source": source,
                "cr": cr,
                "cc": cc,
                "OOF_Revenue_MAE": float(oof["Rev_MAE"]),
                "OOF_COGS_MAE": float(oof["COGS_MAE"]),
                "OOF_Combined_MAE": float(oof["Combined_MAE"]),
                "OOF_Q1_Combined_MAE": float(oof["Q1_Combined_MAE"]),
                "OOF_TetWin_Combined_MAE": float(oof["TetWin_Combined_MAE"]),
                "OOF_Q1_NonTet_Combined_MAE": float(oof["Q1_NonTet_Combined_MAE"]),
                "total_revenue_forecast": float(rev.sum()),
                "total_cogs_forecast": float(cog.sum()),
                "pct_diff_revenue_vs_current_best": float(pct_total_rev),
                "pct_diff_cogs_vs_current_best": float(pct_total_cog),
                "mean_abs_pct_diff_daily_revenue_vs_best": float(mad_daily),
                "max_abs_pct_diff_daily_revenue_vs_best": float(max_daily),
                "expected_public_risk": risk,
                "recommended_priority": pri,
                "submit_or_hold": "hold",
            }
        )

    mf = pd.DataFrame(manifest_rows).sort_values("recommended_priority")

    # Recommend ONE first submission by the task logic
    def _select_recommendation(mf_df: pd.DataFrame) -> str:
        # baseline for improvement comparison
        base = best_combined
        if not np.isfinite(base):
            base = float(mf_df["OOF_Combined_MAE"].min())

        def improved(row):
            return row["OOF_Combined_MAE"] < base

        no_tet = mf_df[mf_df.candidate_name == "no_tet_cr128_cc132"].iloc[0]
        soft = mf_df[mf_df.candidate_name == "soft_tet_cr128_cc132"].iloc[0]
        post = mf_df[mf_df.candidate_name == "post_tet_only_cr128_cc132"].iloc[0]
        blend = mf_df[mf_df.candidate_name == "no_tet_blend_current25"].iloc[0]

        def modest(row):
            return (
                abs(row["pct_diff_revenue_vs_current_best"]) <= 2.0
                and row["mean_abs_pct_diff_daily_revenue_vs_best"] <= 2.0
            )

        if improved(no_tet) and modest(no_tet):
            return "no_tet_cr128_cc132"
        if improved(no_tet) and not modest(no_tet):
            return "no_tet_blend_current25"
        if improved(soft) and modest(soft):
            return "soft_tet_cr128_cc132"
        if improved(post) and modest(post):
            return "post_tet_only_cr128_cc132"
        # fallback safest by diff
        return mf_df.sort_values(["mean_abs_pct_diff_daily_revenue_vs_best"]).iloc[0]["candidate_name"]

    reco = _select_recommendation(mf)
    mf.loc[mf.candidate_name == reco, "submit_or_hold"] = "submit"
    mf.to_csv(os.path.join(OUT_TABLES, "forecast_024_candidate_manifest.csv"), index=False)

    # Figures: candidate profiles + diffs + Q1 profile
    print("\nRendering figures...")
    plt.figure(figsize=(14, 5))
    plt.plot(fdates, win_rev, label="022 best", lw=2, color="black")
    for cname in [
        "no_tet_cr128_cc132",
        "soft_tet_cr128_cc132",
        "post_tet_only_cr128_cc132",
        "no_tet_blend_current25",
    ]:
        plt.plot(fdates, sub_dfs[cname]["Revenue"].values, label=cname, alpha=0.75)
    plt.legend(fontsize=7)
    plt.title("FORECAST-024 Candidate Revenue Profiles")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "forecast_024_candidate_profiles.png"))
    plt.close()

    plt.figure(figsize=(14, 4))
    for cname, df in sub_dfs.items():
        diff_pct = (df["Revenue"].values - win_rev) / win_rev * 100
        plt.plot(fdates, diff_pct, label=cname, alpha=0.8)
    plt.axhline(0, color="black", lw=1)
    plt.title("Daily Revenue % Diff vs Current Best (022)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "forecast_024_diff_vs_current_best.png"))
    plt.close()

    q1m = fdates.month.isin([1, 2, 3])
    plt.figure(figsize=(14, 5))
    plt.plot(fdates[q1m], win_rev[q1m], label="022 best", lw=2, color="black")
    for cname in ["no_tet_cr128_cc132", "soft_tet_cr128_cc132", "post_tet_only_cr128_cc132"]:
        plt.plot(fdates[q1m], sub_dfs[cname]["Revenue"].values[q1m], label=cname, alpha=0.8)
    plt.legend(fontsize=8)
    plt.title("Q1 Revenue Profile Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "forecast_024_q1_profile_comparison.png"))
    plt.close()

    # Report
    print("\nWriting report...")
    # FoldA spotlight
    foldA = ablation_df[ablation_df.fold == "FoldA"].set_index("variant")
    foldA_rev_full = float(foldA.loc["full_reference", "Rev_MAE"]) if "full_reference" in foldA.index else np.nan
    foldA_rev_no_tet = float(foldA.loc["no_tet_all", "Rev_MAE"]) if "no_tet_all" in foldA.index else np.nan
    foldA_delta = (
        (foldA_rev_full - foldA_rev_no_tet) / foldA_rev_full * 100
        if np.isfinite(foldA_rev_full) and np.isfinite(foldA_rev_no_tet)
        else np.nan
    )

    # Variant mean table for report
    var_mean = (
        ablation_df.groupby("variant")[
            [
                "Rev_MAE",
                "COGS_MAE",
                "Combined_MAE",
                "Q1_Combined_MAE",
                "TetWin_Combined_MAE",
                "Q1_NonTet_Combined_MAE",
            ]
        ]
        .mean()
        .sort_values("Combined_MAE")
    )

    # Candidate summary table
    cand_view = mf[[
        "candidate_name",
        "OOF_Combined_MAE",
        "OOF_Q1_Combined_MAE",
        "OOF_TetWin_Combined_MAE",
        "pct_diff_revenue_vs_current_best",
        "mean_abs_pct_diff_daily_revenue_vs_best",
        "expected_public_risk",
        "submit_or_hold",
    ]].copy()

    report_lines = []
    report_lines.append("# FORECAST-024: Tết Feature Surgery Report\n")
    report_lines.append("## Current Best / Recent Failed\n")
    report_lines.append("- Current best public score: **submission_forecast_022_cr128_cc132.csv — 706,621.35616**")
    report_lines.append("- Recently failed: submission_forecast_023_ref_adaptive_qalpha_cr128_cc132.csv — 708,392.43742")
    report_lines.append("- Interpretation: Adaptive alpha failed on public LB; focus shifts to Tết feature surgery.\n")

    report_lines.append("## Why Tết Surgery\n")
    report_lines.append(
        "FORECAST-023 feature ablation found **dropping Tết features improved FoldA MAE materially**. "
        "This suggests misalignment/double-counting vs month/Fourier/promo windows or overfitting around Q1.\n"
    )

    report_lines.append("## Tết Historical Audit (Sales)\n")
    report_lines.append(
        "Audit recomputed `tet_days_diff` exactly as in FORECAST-020 and summarized Revenue/COGS within ±45 days. "
        "See `artifacts/tables/forecast_024_tet_feature_audit.csv` and `artifacts/figures/forecast_024_tet_window_historical_effect.png`.\n"
    )

    report_lines.append("## Ablation Metrics (3 folds, reference architecture, CR=1.28 CC=1.32)\n")
    report_lines.append(_df_to_markdown(var_mean, index=True, floatfmt=",.1f"))
    report_lines.append("\n")
    if np.isfinite(foldA_delta):
        report_lines.append(
            f"FoldA spotlight: full_reference Rev MAE={foldA_rev_full:,.0f} vs no_tet_all Rev MAE={foldA_rev_no_tet:,.0f} "
            f"(**{foldA_delta:+.2f}%**).\n"
        )

    report_lines.append("## Candidates (future horizon)\n")
    report_lines.append(_df_to_markdown(cand_view, index=False, floatfmt=",.3f"))
    report_lines.append("\n")
    report_lines.append(f"## Recommendation\n\nRecommend submitting **submission_forecast_024_{reco}.csv** first.\n")

    report_lines.append("## Risks\n")
    report_lines.append(
        "- If `no_tet_*` shifts totals > ±2% or daily diffs are large, prefer the blend candidate as a safer perturbation.\n"
        "- `post_tet_only` may underfit if the true effect is partly anticipatory/pre-Tết.\n"
    )

    report_lines.append("## Leakage Checklist\n")
    report_lines.append("- [x] No sample_submission Revenue/COGS used (Date-only).")
    report_lines.append("- [x] No external data used.")
    report_lines.append("- [x] All future features are Date/schedule-derived (no lags).")
    report_lines.append("- [x] All folds train-before-validation.")
    report_lines.append("- [x] Current best file unchanged; only read.")
    report_lines.append("- [x] All submissions: 548 rows, Date order == sample_submission, Revenue>0, COGS>=0.\n")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\nManifest:")
    for r in mf.to_dict("records"):
        print(
            f"  {r['candidate_name']:30s}  "
            f"OOF_comb={r['OOF_Combined_MAE']:,.0f}  "
            f"rev_tot_diff={r['pct_diff_revenue_vs_current_best']:+.2f}%  "
            f"mad={r['mean_abs_pct_diff_daily_revenue_vs_best']:.2f}%  "
            f"{r['submit_or_hold']}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
