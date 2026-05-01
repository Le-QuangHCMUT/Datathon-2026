"""FORECAST-009

Leakage-safe forecasting baseline + model comparison for daily Revenue and COGS.

Run from project root:
    python src/datathon/forecasting/forecast_009_baseline_models.py

Constraints honored:
- Uses ONLY historical targets from data/sales.csv up to 2022-12-31.
- Uses sample_submission.csv ONLY for Date order/schema (ignores its Revenue/COGS values).
- Time-based validation only.
- Rolling features are shifted (no same-day leakage).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib (headless-safe)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# Optional sklearn
SKLEARN_AVAILABLE = False
SKLEARN_IMPORT_ERROR = ""
try:
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception as e:  # pragma: no cover
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = repr(e)


# -----------------------------
# Utilities
# -----------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _project_root() -> Path:
    # .../src/datathon/forecasting/forecast_009_baseline_models.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3]


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {label} at {path}")


def _assert_date_index_unique_sorted_continuous(idx: pd.DatetimeIndex, name: str) -> None:
    if idx.isna().any():
        raise ValueError(f"{name}: Date index contains NaT")

    if idx.has_duplicates:
        dups = idx[idx.duplicated()].unique()
        raise ValueError(f"{name}: Date index has duplicates (showing up to 5): {list(dups[:5])}")

    if not idx.is_monotonic_increasing:
        raise ValueError(f"{name}: Date index is not sorted ascending")

    expected = pd.date_range(idx.min(), idx.max(), freq="D")
    if len(expected) != len(idx) or not expected.equals(idx):
        missing = expected.difference(idx)
        extra = idx.difference(expected)
        msg = (
            f"{name}: Date index is not continuous daily. "
            f"missing={len(missing)} extra={len(extra)} "
            f"range=[{idx.min().date()}..{idx.max().date()}]"
        )
        if len(missing) > 0:
            msg += f" | first_missing={missing[0].date()}"
        if len(extra) > 0:
            msg += f" | first_extra={extra[0].date()}"
        raise ValueError(msg)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _postprocess_target_predictions(pred: np.ndarray, target: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if target.lower() == "revenue":
        return np.maximum(pred, 1e-3)
    if target.lower() == "cogs":
        return np.maximum(pred, 0.0)
    return pred


def _cap_cogs_to_revenue(
    revenue_pred: np.ndarray,
    cogs_pred: np.ndarray,
    ratio_cap: float,
) -> Tuple[np.ndarray, int]:
    revenue_pred = np.asarray(revenue_pred, dtype=float)
    cogs_pred = np.asarray(cogs_pred, dtype=float)
    cap_values = revenue_pred * ratio_cap
    capped = np.minimum(cogs_pred, cap_values)
    n_capped = int(np.sum(cogs_pred > cap_values))
    return capped, n_capped


def _safe_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.2f}%"


# -----------------------------
# Feature engineering
# -----------------------------


@dataclass(frozen=True)
class FeatureSpec:
    lags: Tuple[int, ...] = (1, 7, 14, 28, 30, 56, 91, 182, 364, 365)
    roll_mean_windows: Tuple[int, ...] = (7, 14, 28, 56, 91, 365)
    roll_std_windows: Tuple[int, ...] = (28, 91)


CAL_FEATURES: Tuple[str, ...] = (
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "iso_week",
    "quarter",
    "is_month_start",
    "is_month_end",
    "days_since_start",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
)


def _calendar_features(index: pd.DatetimeIndex, global_start: pd.Timestamp) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["year"] = index.year
    df["month"] = index.month
    df["day"] = index.day
    df["dayofweek"] = index.dayofweek
    df["dayofyear"] = index.dayofyear
    # ISO week (1..53)
    isocal = index.isocalendar()
    df["iso_week"] = isocal.week.astype(int)
    df["quarter"] = index.quarter
    df["is_month_start"] = index.is_month_start.astype(int)
    df["is_month_end"] = index.is_month_end.astype(int)
    df["days_since_start"] = (index - global_start).days

    # Cyclical encodings
    doy = df["dayofyear"].astype(float).to_numpy()
    dow = df["dayofweek"].astype(float).to_numpy()
    month = df["month"].astype(float).to_numpy()

    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    return df


def _build_feature_frame_from_series(
    y: pd.Series, spec: FeatureSpec, global_start: pd.Timestamp
) -> pd.DataFrame:
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must be indexed by DatetimeIndex")

    x = _calendar_features(y.index, global_start=global_start)

    for lag in spec.lags:
        x[f"lag_{lag}"] = y.shift(lag)

    shifted = y.shift(1)
    for w in spec.roll_mean_windows:
        x[f"rolling_mean_{w}"] = shifted.rolling(window=w).mean()

    for w in spec.roll_std_windows:
        x[f"rolling_std_{w}"] = shifted.rolling(window=w).std(ddof=0)

    return x


def _feature_columns(spec: FeatureSpec) -> List[str]:
    cols: List[str] = list(CAL_FEATURES)
    cols += [f"lag_{lag}" for lag in spec.lags]
    cols += [f"rolling_mean_{w}" for w in spec.roll_mean_windows]
    cols += [f"rolling_std_{w}" for w in spec.roll_std_windows]
    return cols


def _build_feature_row_for_date(
    d: pd.Timestamp,
    history: pd.Series,
    spec: FeatureSpec,
    global_start: pd.Timestamp,
) -> Dict[str, float]:
    # Calendar
    idx = pd.DatetimeIndex([d])
    cal = _calendar_features(idx, global_start=global_start).iloc[0].to_dict()

    out: Dict[str, float] = {k: float(v) for k, v in cal.items()}

    # Lags
    for lag in spec.lags:
        out[f"lag_{lag}"] = float(history.loc[d - pd.Timedelta(days=lag)])

    # Rolling statistics (shifted by 1 => end at d-1)
    end = d - pd.Timedelta(days=1)
    for w in spec.roll_mean_windows:
        start = d - pd.Timedelta(days=w)
        window_idx = pd.date_range(start, end, freq="D")
        vals = history.reindex(window_idx).to_numpy(dtype=float)
        out[f"rolling_mean_{w}"] = float(np.mean(vals))

    for w in spec.roll_std_windows:
        start = d - pd.Timedelta(days=w)
        window_idx = pd.date_range(start, end, freq="D")
        vals = history.reindex(window_idx).to_numpy(dtype=float)
        out[f"rolling_std_{w}"] = float(np.std(vals))

    return out


# -----------------------------
# Model definitions
# -----------------------------


class BaseModel:
    name: str

    def fit(self, y_train: pd.Series, global_start: pd.Timestamp) -> "BaseModel":
        raise NotImplementedError

    def predict_recursive(
        self,
        history: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        global_start: pd.Timestamp,
        spec: FeatureSpec,
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        """Return (predictions, features_used)"""
        raise NotImplementedError

    def is_ml(self) -> bool:
        return False


class NaiveLastValueModel(BaseModel):
    name = "naive_last"

    def fit(self, y_train: pd.Series, global_start: pd.Timestamp) -> "NaiveLastValueModel":
        self.last_value_ = float(y_train.iloc[-1])
        return self

    def predict_recursive(
        self,
        history: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        global_start: pd.Timestamp,
        spec: FeatureSpec,
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        # Constant forecast equal to last observed
        preds = np.full(len(forecast_dates), float(history.iloc[-1]), dtype=float)
        return preds, None


class SeasonalNaiveModel(BaseModel):
    def __init__(self, lag_days: int):
        self.lag_days = int(lag_days)
        self.name = f"seasonal_naive_{self.lag_days}"

    def fit(self, y_train: pd.Series, global_start: pd.Timestamp) -> "SeasonalNaiveModel":
        # No learned params
        return self

    def predict_recursive(
        self,
        history: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        global_start: pd.Timestamp,
        spec: FeatureSpec,
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        hist = history.copy()
        preds: List[float] = []
        for d in forecast_dates:
            ref = d - pd.Timedelta(days=self.lag_days)
            if ref not in hist.index:
                raise ValueError(
                    f"{self.name}: missing seasonal reference date {ref.date()} for forecast date {d.date()}"
                )
            p = float(hist.loc[ref])
            preds.append(p)
            hist.loc[d] = p
        return np.asarray(preds, dtype=float), None


class CalendarProfileModel(BaseModel):
    name = "calendar_profile"

    def fit(self, y_train: pd.Series, global_start: pd.Timestamp) -> "CalendarProfileModel":
        idx = y_train.index
        df = pd.DataFrame({"y": y_train.values}, index=idx)

        self.overall_mean_ = float(df["y"].mean())
        self.mean_by_doy_ = df.groupby(idx.dayofyear)["y"].mean().to_dict()
        self.mean_by_month_dow_ = df.groupby([idx.month, idx.dayofweek])["y"].mean().to_dict()
        self.mean_by_dow_ = df.groupby(idx.dayofweek)["y"].mean().to_dict()
        return self

    def predict_recursive(
        self,
        history: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        global_start: pd.Timestamp,
        spec: FeatureSpec,
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        preds: List[float] = []
        for d in forecast_dates:
            doy = int(d.dayofyear)
            m = int(d.month)
            dow = int(d.dayofweek)

            if doy in self.mean_by_doy_:
                p = float(self.mean_by_doy_[doy])
            elif (m, dow) in self.mean_by_month_dow_:
                p = float(self.mean_by_month_dow_[(m, dow)])
            elif dow in self.mean_by_dow_:
                p = float(self.mean_by_dow_[dow])
            else:
                p = float(self.overall_mean_)

            preds.append(p)
        return np.asarray(preds, dtype=float), None


class SklearnLagModel(BaseModel):
    def __init__(self, name: str, estimator: Any):
        self.name = name
        self.estimator = estimator
        self._feature_cols: List[str] = []

    def is_ml(self) -> bool:
        return True

    def fit(self, y_train: pd.Series, global_start: pd.Timestamp) -> "SklearnLagModel":
        spec = FeatureSpec()
        x_full = _build_feature_frame_from_series(y_train, spec=spec, global_start=global_start)
        df = x_full.copy()
        df["y"] = y_train
        df = df.dropna(axis=0)

        self._feature_cols = _feature_columns(spec)
        x_train = df[self._feature_cols]
        y = df["y"].astype(float)

        self.estimator.fit(x_train, y)
        return self

    def predict_recursive(
        self,
        history: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        global_start: pd.Timestamp,
        spec: FeatureSpec,
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        hist = history.copy()
        rows: List[Dict[str, float]] = []
        preds: List[float] = []

        cols = _feature_columns(spec)
        self._feature_cols = cols

        for d in forecast_dates:
            row = _build_feature_row_for_date(d, history=hist, spec=spec, global_start=global_start)
            x = pd.DataFrame([row], columns=cols)
            p = float(self.estimator.predict(x)[0])
            rows.append(row)
            preds.append(p)
            hist.loc[d] = p

        return np.asarray(preds, dtype=float), pd.DataFrame(rows, index=forecast_dates)


def _candidate_models() -> List[BaseModel]:
    models: List[BaseModel] = [
        NaiveLastValueModel(),
        SeasonalNaiveModel(364),
        SeasonalNaiveModel(365),
        CalendarProfileModel(),
    ]

    if not SKLEARN_AVAILABLE:
        logging.warning("sklearn not available; ML models will be skipped: %s", SKLEARN_IMPORT_ERROR)
        return models

    # Linear
    ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    enet = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.2,
                    random_state=SEED,
                    max_iter=20000,
                ),
            ),
        ]
    )

    # Trees / boosting
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        max_depth=18,
        min_samples_leaf=2,
    )
    et = ExtraTreesRegressor(
        n_estimators=400,
        random_state=SEED,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )

    # HistGradientBoosting is typically strong and fast
    hgb = HistGradientBoostingRegressor(
        random_state=SEED,
        learning_rate=0.05,
        max_depth=8,
        max_iter=600,
        l2_regularization=0.0,
    )

    gbr = GradientBoostingRegressor(
        random_state=SEED,
        learning_rate=0.05,
        n_estimators=800,
        max_depth=3,
    )

    models += [
        SklearnLagModel("ridge", ridge),
        SklearnLagModel("elasticnet", enet),
        SklearnLagModel("random_forest", rf),
        SklearnLagModel("extra_trees", et),
        SklearnLagModel("hist_gbdt", hgb),
        SklearnLagModel("gbr", gbr),
    ]

    return models


# -----------------------------
# Evaluation / selection
# -----------------------------


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    @property
    def val_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.val_start, self.val_end, freq="D")

    @property
    def horizon(self) -> int:
        return len(self.val_dates)


def _build_splits(last_date: pd.Timestamp) -> List[SplitSpec]:
    # Primary split: last 548 days
    def holdout(name: str, horizon: int) -> SplitSpec:
        train_end = last_date - pd.Timedelta(days=horizon)
        val_start = train_end + pd.Timedelta(days=1)
        val_end = last_date
        return SplitSpec(name=name, train_end=train_end, val_start=val_start, val_end=val_end)

    splits: List[SplitSpec] = [holdout("holdout_548", 548), holdout("holdout_365", 365)]

    # Rolling-origin year splits (optional but cheap)
    for year in (2020, 2021, 2022):
        val_start = pd.Timestamp(f"{year}-01-01")
        val_end = pd.Timestamp(f"{year}-12-31")
        train_end = val_start - pd.Timedelta(days=1)
        splits.append(SplitSpec(name=f"ro_{year}", train_end=train_end, val_start=val_start, val_end=val_end))

    return splits


def _evaluate_model_on_split(
    model: BaseModel,
    y: pd.Series,
    split: SplitSpec,
    target: str,
    global_start: pd.Timestamp,
    spec: FeatureSpec,
) -> Tuple[Dict[str, Any], np.ndarray, Optional[pd.DataFrame]]:
    y_train = y.loc[: split.train_end]
    y_val = y.loc[split.val_dates]

    if len(y_train) < 365 + 10:
        raise ValueError(f"Not enough training history for {split.name}: {len(y_train)} rows")

    model.fit(y_train=y_train, global_start=global_start)

    history = y.loc[: split.train_end].copy()
    preds, feats = model.predict_recursive(
        history=history,
        forecast_dates=split.val_dates,
        global_start=global_start,
        spec=spec,
    )
    preds = _postprocess_target_predictions(preds, target=target)

    y_true = y_val.to_numpy(dtype=float)
    metrics = {
        "split": split.name,
        "horizon": split.horizon,
        "model": model.name,
        "target": target,
        "mae": _mae(y_true, preds),
        "rmse": _rmse(y_true, preds),
        "r2": _r2(y_true, preds),
        "mape": _mape(y_true, preds),
    }

    return metrics, preds, feats


def _select_best_model(
    metrics_df: pd.DataFrame,
    target: str,
    split_name: str = "holdout_548",
) -> str:
    df = metrics_df[(metrics_df["target"] == target) & (metrics_df["split"] == split_name)].copy()
    if df.empty:
        raise ValueError(f"No metrics to select model for target={target} split={split_name}")

    # Primary: RMSE then MAE
    df = df.sort_values(["rmse", "mae"], ascending=True)
    return str(df.iloc[0]["model"])


# -----------------------------
# Explainability
# -----------------------------


def _extract_estimator_and_names(model: Any) -> Tuple[Any, str]:
    # Return (estimator, estimator_name)
    if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
        # Pipeline
        if "model" in model.named_steps:
            return model, "pipeline"
        return model, "pipeline"
    return model, type(model).__name__


def _compute_feature_importance(
    estimator: Any,
    feature_names: Sequence[str],
    x_eval: pd.DataFrame,
    y_eval: np.ndarray,
    target: str,
) -> pd.DataFrame:
    # Prefer built-in feature_importances_ for tree models, then coef_ for linear, else permutation importance.
    est, _ = _extract_estimator_and_names(estimator)

    importances: Optional[np.ndarray] = None
    method = ""

    # Tree-based
    if hasattr(est, "feature_importances_"):
        try:
            importances = np.asarray(est.feature_importances_, dtype=float)
            method = "model.feature_importances_"
        except Exception:
            importances = None

    # Linear
    if importances is None and hasattr(est, "named_steps"):
        last = None
        try:
            last = est.named_steps.get("model")
        except Exception:
            last = None
        if last is not None and hasattr(last, "coef_"):
            importances = np.abs(np.asarray(last.coef_, dtype=float))
            method = "abs(coef_)"

    if importances is None and hasattr(est, "coef_"):
        importances = np.abs(np.asarray(est.coef_, dtype=float))
        method = "abs(coef_)"

    # Permutation importance
    if importances is None:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Permutation importance requires sklearn")
        result = permutation_importance(
            est,
            x_eval,
            y_eval,
            n_repeats=10,
            random_state=SEED,
            scoring="neg_mean_absolute_error",
        )
        importances = np.asarray(result.importances_mean, dtype=float)
        method = "permutation_importance_neg_mae"

    if importances.shape[0] != len(feature_names):
        raise ValueError(
            f"Feature importance length mismatch: {importances.shape[0]} vs {len(feature_names)}"
        )

    out = pd.DataFrame(
        {
            "target": target,
            "feature": list(feature_names),
            "importance": importances,
            "method": method,
        }
    )
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# Error analysis / plotting
# -----------------------------


def _error_slices(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # df must include columns: Date, actual_{target}, pred_{target}
    t = target.lower()
    actual_col = f"{t}_actual"
    pred_col = f"{t}_pred"
    if actual_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Missing columns for error analysis: {actual_col}, {pred_col}")

    dti = pd.to_datetime(df["Date"])
    actual = df[actual_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    err = pred - actual
    abs_err = np.abs(err)

    frame = pd.DataFrame(
        {
            "Date": dti,
            "target": target,
            "actual": actual,
            "pred": pred,
            "error": err,
            "abs_error": abs_err,
            "month": dti.dt.month,
            "dayofweek": dti.dt.dayofweek,
            "year_month": dti.dt.strftime("%Y-%m"),
        }
    )

    # High vs normal (based on validation actual distribution)
    thresh = float(np.quantile(actual, 0.9))
    frame["revenue_bucket" if t == "revenue" else "bucket"] = np.where(
        actual >= thresh, "high_p90+", "normal"
    )

    def summarize(group_col: str, slice_name: str) -> pd.DataFrame:
        rows = []
        for key, g in frame.groupby(group_col):
            y_true = g["actual"].to_numpy(dtype=float)
            y_pred = g["pred"].to_numpy(dtype=float)
            rows.append(
                {
                    "target": target,
                    "slice": slice_name,
                    "group": str(key),
                    "n": int(len(g)),
                    "mae": _mae(y_true, y_pred),
                    "rmse": _rmse(y_true, y_pred),
                    "mape": _mape(y_true, y_pred),
                    "bias_mean_error": float(np.mean(y_pred - y_true)),
                    "mean_actual": float(np.mean(y_true)),
                    "mean_pred": float(np.mean(y_pred)),
                }
            )
        return pd.DataFrame(rows)

    out = pd.concat(
        [
            summarize("month", "month"),
            summarize("dayofweek", "dayofweek"),
            summarize("year_month", "year_month"),
            summarize("revenue_bucket" if t == "revenue" else "bucket", "bucket"),
        ],
        axis=0,
        ignore_index=True,
    )

    return out.sort_values(["slice", "mae"], ascending=[True, False]).reset_index(drop=True)


def _plot_validation(
    val_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    dates = pd.to_datetime(val_df["Date"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(dates, val_df["revenue_actual"], label="Revenue actual", linewidth=1.2)
    axes[0].plot(dates, val_df["revenue_pred"], label="Revenue pred", linewidth=1.2)
    axes[0].set_title("Revenue: actual vs predicted")
    axes[0].legend(loc="upper right")

    axes[1].plot(dates, val_df["cogs_actual"], label="COGS actual", linewidth=1.2)
    axes[1].plot(dates, val_df["cogs_pred"], label="COGS pred", linewidth=1.2)
    axes[1].set_title("COGS: actual vs predicted")
    axes[1].legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_residuals(val_df: pd.DataFrame, out_path: Path, title: str) -> None:
    dates = pd.to_datetime(val_df["Date"])
    rev_resid = val_df["revenue_pred"].to_numpy(dtype=float) - val_df["revenue_actual"].to_numpy(dtype=float)
    cogs_resid = val_df["cogs_pred"].to_numpy(dtype=float) - val_df["cogs_actual"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].plot(dates, rev_resid, linewidth=1.0)
    axes[0].set_title("Revenue residuals (pred - actual)")

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].plot(dates, cogs_resid, linewidth=1.0)
    axes[1].set_title("COGS residuals (pred - actual)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(fi_df: pd.DataFrame, out_path: Path, title: str) -> None:
    # Plot top 20 (Revenue preferred)
    if fi_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Prefer revenue rows
    plot_df = fi_df.copy()
    if (plot_df["target"] == "Revenue").any():
        plot_df = plot_df[plot_df["target"] == "Revenue"].copy()

    plot_df = plot_df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(plot_df["feature"], plot_df["importance"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_future_forecast(future_df: pd.DataFrame, out_path: Path, title: str) -> None:
    dates = pd.to_datetime(future_df["Date"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(dates, future_df["Revenue"], linewidth=1.2)
    axes[0].set_title("Forecast Revenue")

    axes[1].plot(dates, future_df["COGS"], linewidth=1.2)
    axes[1].set_title("Forecast COGS")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    _setup_logging()

    root = _project_root()
    logging.info("Project root: %s", root)

    # Inputs
    sales_path = root / "data" / "sales.csv"
    sample_path = root / "sample_submission.csv"

    _require_file(sales_path, "data/sales.csv")
    _require_file(sample_path, "sample_submission.csv")

    sales = pd.read_csv(sales_path, parse_dates=["Date"])
    sample = pd.read_csv(sample_path, parse_dates=["Date"])

    # Validate columns
    for col in ("Date", "Revenue", "COGS"):
        if col not in sales.columns:
            raise ValueError(f"sales.csv missing required column: {col}")

    if "Date" not in sample.columns:
        raise ValueError("sample_submission.csv missing required column: Date")

    # Ignore placeholders (but warn if non-zero)
    if "Revenue" in sample.columns and "COGS" in sample.columns:
        nonzero = int(((sample["Revenue"].fillna(0) != 0) | (sample["COGS"].fillna(0) != 0)).sum())
        if nonzero > 0:
            logging.warning(
                "sample_submission.csv contains non-zero Revenue/COGS in %s rows; treating as placeholders ONLY.",
                nonzero,
            )

    # IMPORTANT: Do NOT sort. We must *verify* uniqueness/sortedness/continuity as-is.
    sales = sales[["Date", "Revenue", "COGS"]].copy()
    sales["Date"] = pd.to_datetime(sales["Date"])

    sales_idx = pd.DatetimeIndex(sales["Date"], name="Date")
    _assert_date_index_unique_sorted_continuous(sales_idx, name="sales")

    sample = sample[["Date"]].copy()
    sample["Date"] = pd.to_datetime(sample["Date"])

    sample_idx = pd.DatetimeIndex(sample["Date"], name="Date")
    _assert_date_index_unique_sorted_continuous(sample_idx, name="sample_submission")

    # Horizon checks
    expected_start = pd.Timestamp("2023-01-01")
    expected_end = pd.Timestamp("2024-07-01")
    if sample_idx.min() != expected_start or sample_idx.max() != expected_end:
        raise ValueError(
            f"sample_submission Date range mismatch: got [{sample_idx.min().date()}..{sample_idx.max().date()}], "
            f"expected [{expected_start.date()}..{expected_end.date()}]"
        )
    if len(sample_idx) != 548:
        raise ValueError(f"sample_submission row count mismatch: got {len(sample_idx)}, expected 548")

    # Training cutoff enforcement
    sales_last = sales_idx.max()
    if sales_last != pd.Timestamp("2022-12-31"):
        logging.warning("sales last date is %s (expected 2022-12-31)", sales_last.date())

    # Make sure we never train after 2022-12-31
    train_cutoff = pd.Timestamp("2022-12-31")
    sales = sales[sales["Date"] <= train_cutoff].copy()
    sales_idx = pd.DatetimeIndex(sales["Date"], name="Date")
    _assert_date_index_unique_sorted_continuous(sales_idx, name="sales<=2022-12-31")

    global_start = sales_idx.min()

    # Series
    revenue = pd.Series(sales["Revenue"].to_numpy(dtype=float), index=sales_idx, name="Revenue")
    cogs = pd.Series(sales["COGS"].to_numpy(dtype=float), index=sales_idx, name="COGS")

    # Output dirs
    out_docs = root / "docs" / "model_log"
    out_forecasts = root / "artifacts" / "forecasts"
    out_submissions = root / "artifacts" / "submissions"
    out_tables = root / "artifacts" / "tables"
    out_figures = root / "artifacts" / "figures"

    for d in (out_docs, out_forecasts, out_submissions, out_tables, out_figures):
        d.mkdir(parents=True, exist_ok=True)

    spec = FeatureSpec()

    # Compute ratio cap from full training history
    ratio = (cogs / revenue).replace([np.inf, -np.inf], np.nan)
    ratio = ratio[(revenue > 0) & ratio.notna()]
    ratio_p99 = float(np.quantile(ratio.to_numpy(dtype=float), 0.99)) if len(ratio) > 0 else 1.0
    ratio_cap = float(min(max(ratio_p99, 1.05), 2.0))

    logging.info("Historical COGS/Revenue ratio p99=%.4f -> cap=%.4f", ratio_p99, ratio_cap)

    # Splits and candidates
    splits = _build_splits(last_date=sales_idx.max())
    candidates = _candidate_models()

    logging.info("Evaluating %d models across %d splits", len(candidates), len(splits))

    metrics_rows: List[Dict[str, Any]] = []

    # Keep primary split predictions for later diagnostics (holdout_548)
    holdout_name = "holdout_548"
    holdout_split = next(s for s in splits if s.name == holdout_name)
    holdout_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for model in candidates:
        for split in splits:
            for target_name, series in ("Revenue", revenue), ("COGS", cogs):
                try:
                    row, preds, feats = _evaluate_model_on_split(
                        model=model,
                        y=series,
                        split=split,
                        target=target_name,
                        global_start=global_start,
                        spec=spec,
                    )
                    metrics_rows.append(row)

                    if split.name == holdout_name:
                        holdout_cache[(model.name, target_name)] = {
                            "preds": preds,
                            "feats": feats,
                            "row": row,
                        }
                except Exception as e:
                    logging.warning(
                        "Skipping model=%s target=%s split=%s due to error: %s",
                        model.name,
                        target_name,
                        split.name,
                        repr(e),
                    )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise RuntimeError("No models were successfully evaluated; cannot proceed")

    metrics_out = out_tables / "forecast_009_model_metrics.csv"
    metrics_df.to_csv(metrics_out, index=False)
    logging.info("Wrote metrics: %s", metrics_out)

    # Select models based on primary split
    best_revenue_model_name = _select_best_model(metrics_df, target="Revenue", split_name=holdout_name)
    best_cogs_model_name = _select_best_model(metrics_df, target="COGS", split_name=holdout_name)

    logging.info("Selected revenue model: %s", best_revenue_model_name)
    logging.info("Selected cogs model: %s", best_cogs_model_name)

    # Re-instantiate selected models (fresh) for training/importance. Do NOT reuse the same
    # instance for both targets (if names match) or the estimator will be overwritten.
    def fresh_model_by_name(model_name: str) -> BaseModel:
        fresh = {m.name: m for m in _candidate_models()}
        if model_name not in fresh:
            raise RuntimeError(f"Selected model '{model_name}' missing from candidates (unexpected)")
        return fresh[model_name]

    revenue_model = fresh_model_by_name(best_revenue_model_name)
    cogs_model = fresh_model_by_name(best_cogs_model_name)

    # Validation predictions (primary split) for error analysis
    # Fit on train period, predict recursively for validation horizon
    def fit_and_forecast_primary(model: BaseModel, series: pd.Series, target_name: str) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        y_train = series.loc[: holdout_split.train_end]
        model.fit(y_train=y_train, global_start=global_start)
        history = series.loc[: holdout_split.train_end].copy()
        preds, feats = model.predict_recursive(
            history=history,
            forecast_dates=holdout_split.val_dates,
            global_start=global_start,
            spec=spec,
        )
        preds = _postprocess_target_predictions(preds, target=target_name)
        return preds, feats

    rev_val_pred, rev_val_feats = fit_and_forecast_primary(revenue_model, revenue, "Revenue")
    cogs_val_pred, cogs_val_feats = fit_and_forecast_primary(cogs_model, cogs, "COGS")

    val_dates = holdout_split.val_dates
    val_df = pd.DataFrame(
        {
            "Date": val_dates,
            "revenue_actual": revenue.loc[val_dates].to_numpy(dtype=float),
            "revenue_pred": rev_val_pred,
            "cogs_actual": cogs.loc[val_dates].to_numpy(dtype=float),
            "cogs_pred": cogs_val_pred,
            "revenue_model": best_revenue_model_name,
            "cogs_model": best_cogs_model_name,
            "split": holdout_name,
        }
    )

    val_out = out_forecasts / "forecast_009_validation_predictions.csv"
    val_df.to_csv(val_out, index=False)
    logging.info("Wrote validation predictions: %s", val_out)

    # Error analysis
    err_rev = _error_slices(val_df.rename(columns={"revenue_actual": "revenue_actual", "revenue_pred": "revenue_pred"}), "Revenue")
    err_cogs = _error_slices(val_df.rename(columns={"cogs_actual": "cogs_actual", "cogs_pred": "cogs_pred"}), "COGS")
    err_df = pd.concat([err_rev, err_cogs], axis=0, ignore_index=True)
    err_out = out_tables / "forecast_009_error_analysis.csv"
    err_df.to_csv(err_out, index=False)
    logging.info("Wrote error analysis: %s", err_out)

    # Plots
    _plot_validation(
        val_df,
        out_path=out_figures / "forecast_009_validation_actual_vs_pred.png",
        title=f"FORECAST-009 Holdout Validation ({holdout_name})",
    )
    _plot_residuals(
        val_df,
        out_path=out_figures / "forecast_009_validation_residuals.png",
        title=f"FORECAST-009 Residuals ({holdout_name})",
    )

    # Explainability (feature importance)
    fi_rows: List[pd.DataFrame] = []

    def _maybe_compute_fi(
        target_name: str,
        model_obj: BaseModel,
        feats: Optional[pd.DataFrame],
        y_true: pd.Series,
    ) -> None:
        if feats is None or not model_obj.is_ml():
            # Baseline explainability placeholder
            fi_rows.append(
                pd.DataFrame(
                    {
                        "target": [target_name],
                        "feature": ["baseline_rule"],
                        "importance": [1.0],
                        "method": ["baseline_explanation"],
                    }
                )
            )
            return

        if not SKLEARN_AVAILABLE:
            fi_rows.append(
                pd.DataFrame(
                    {
                        "target": [target_name],
                        "feature": ["sklearn_missing"],
                        "importance": [np.nan],
                        "method": ["unavailable"],
                    }
                )
            )
            return

        # If model is SklearnLagModel, estimator lives in model_obj.estimator
        est = getattr(model_obj, "estimator", None)
        if est is None:
            # Should not happen for ML models here
            fi_rows.append(
                pd.DataFrame(
                    {
                        "target": [target_name],
                        "feature": ["unknown_model"],
                        "importance": [np.nan],
                        "method": ["unavailable"],
                    }
                )
            )
            return

        feature_names = list(feats.columns)
        x_eval = feats.copy()
        y_eval = y_true.to_numpy(dtype=float)

        try:
            fi = _compute_feature_importance(est, feature_names, x_eval, y_eval, target=target_name)
            fi_rows.append(fi)
        except Exception as e:
            logging.warning("Feature importance failed for %s: %s", target_name, repr(e))
            fi_rows.append(
                pd.DataFrame(
                    {
                        "target": [target_name],
                        "feature": ["feature_importance_failed"],
                        "importance": [np.nan],
                        "method": [repr(e)],
                    }
                )
            )

    _maybe_compute_fi("Revenue", revenue_model, rev_val_feats, revenue.loc[val_dates])
    _maybe_compute_fi("COGS", cogs_model, cogs_val_feats, cogs.loc[val_dates])

    fi_df = pd.concat(fi_rows, axis=0, ignore_index=True)
    fi_out = out_tables / "forecast_009_feature_importance.csv"
    fi_df.to_csv(fi_out, index=False)
    logging.info("Wrote feature importance: %s", fi_out)

    _plot_feature_importance(
        fi_df,
        out_path=out_figures / "forecast_009_feature_importance.png",
        title="FORECAST-009 Feature Importance (top 20)",
    )

    # Train on all data (through 2022-12-31) and forecast future horizon
    full_end = sales_idx.max()

    def fit_full_and_forecast(model: BaseModel, series: pd.Series, target_name: str) -> np.ndarray:
        model.fit(y_train=series.loc[:full_end], global_start=global_start)
        history = series.loc[:full_end].copy()
        preds, _ = model.predict_recursive(
            history=history,
            forecast_dates=sample_idx,
            global_start=global_start,
            spec=spec,
        )
        preds = _postprocess_target_predictions(preds, target=target_name)
        return preds

    future_rev = fit_full_and_forecast(revenue_model, revenue, "Revenue")
    future_cogs = fit_full_and_forecast(cogs_model, cogs, "COGS")

    # Apply sanity cap to COGS based on predicted Revenue
    future_cogs, n_capped = _cap_cogs_to_revenue(future_rev, future_cogs, ratio_cap=ratio_cap)
    if n_capped > 0:
        logging.warning("Capped COGS on %d future dates using ratio_cap=%.4f", n_capped, ratio_cap)

    future_df = pd.DataFrame(
        {
            "Date": sample_idx,
            "Revenue": future_rev,
            "COGS": future_cogs,
            "revenue_model": best_revenue_model_name,
            "cogs_model": best_cogs_model_name,
        }
    )

    fut_out = out_forecasts / "forecast_009_future_predictions.csv"
    future_df.to_csv(fut_out, index=False)
    logging.info("Wrote future predictions: %s", fut_out)

    _plot_future_forecast(
        future_df,
        out_path=out_figures / "forecast_009_future_forecast.png",
        title="FORECAST-009 Future Forecast (submission horizon)",
    )

    # Submission file (exact schema/order)
    submission = future_df[["Date", "Revenue", "COGS"]].copy()

    # Validate order vs sample_submission
    if len(submission) != len(sample_idx):
        raise ValueError("Submission row count does not match sample_submission")
    if not pd.DatetimeIndex(submission["Date"]).equals(sample_idx):
        raise ValueError("Submission Date order does not exactly match sample_submission")

    # Required constraints
    if submission[["Revenue", "COGS"]].isna().any().any():
        raise ValueError("Submission contains nulls")
    if not (submission["Revenue"] > 0).all():
        bad = int((submission["Revenue"] <= 0).sum())
        raise ValueError(f"Submission Revenue must be >0 for all rows; bad_rows={bad}")
    if not (submission["COGS"] >= 0).all():
        bad = int((submission["COGS"] < 0).sum())
        raise ValueError(f"Submission COGS must be >=0 for all rows; bad_rows={bad}")

    sub_out = out_submissions / "submission_forecast_009.csv"
    submission.to_csv(sub_out, index=False)
    logging.info("Wrote submission: %s", sub_out)

    # Report
    report_path = out_docs / "forecast_009_baseline_report.md"

    # Metrics table for primary split only (compact)
    m_primary = metrics_df[metrics_df["split"] == holdout_name].copy()
    m_primary = m_primary.sort_values(["target", "rmse", "mae"], ascending=[True, True, True])

    def _md_escape(val: str) -> str:
        # Minimal escaping for pipe tables
        return val.replace("|", "\\|")

    def _df_to_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"

        cols = list(df.columns)
        # stringify
        rows = [[_md_escape(str(v)) for v in r] for r in df.to_numpy().tolist()]
        header = [_md_escape(str(c)) for c in cols]

        # widths
        widths = [len(h) for h in header]
        for r in rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))

        def fmt_row(values: List[str]) -> str:
            padded = [values[i].ljust(widths[i]) for i in range(len(values))]
            return "| " + " | ".join(padded) + " |"

        out_lines = [fmt_row(header), "| " + " | ".join(["-" * w for w in widths]) + " |"]
        out_lines += [fmt_row(r) for r in rows]
        return "\n".join(out_lines)

    def to_md_table(df: pd.DataFrame) -> str:
        cols = ["model", "target", "mae", "rmse", "r2", "mape"]
        view = df[cols].copy()
        view["mae"] = view["mae"].map(lambda x: f"{x:,.2f}")
        view["rmse"] = view["rmse"].map(lambda x: f"{x:,.2f}")
        view["r2"] = view["r2"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
        view["mape"] = view["mape"].map(_safe_pct)
        return _df_to_markdown_table(view)

    top_fi_rev = fi_df[fi_df["target"] == "Revenue"].head(10)
    top_fi_cogs = fi_df[fi_df["target"] == "COGS"].head(10)

    report_lines: List[str] = []
    report_lines.append(f"# FORECAST-009 — Leakage-Safe Forecasting Baseline Report\n")
    report_lines.append("## 1. Objective and target definition\n")
    report_lines.append(
        "Forecast daily **Revenue** and **COGS** for the submission horizon (2023-01-01 to 2024-07-01) "
        "using only historical data up to 2022-12-31.\n"
    )

    report_lines.append("## 2. Data used\n")
    report_lines.append("- Training time series: `data/sales.csv` (Date, Revenue, COGS)\n")
    report_lines.append("- Submission template: `sample_submission.csv` (Date order only; Revenue/COGS ignored)\n")
    report_lines.append(
        f"- Sales history: {sales_idx.min().date()} to {sales_idx.max().date()} "
        f"({len(sales_idx):,} daily rows; continuous)\n"
    )
    report_lines.append(
        f"- Submission horizon: {sample_idx.min().date()} to {sample_idx.max().date()} "
        f"({len(sample_idx):,} daily rows; continuous)\n"
    )

    report_lines.append("## 3. Leakage checklist\n")
    report_lines.append("- No external data used.\n")
    report_lines.append("- `sample_submission.csv` Revenue/COGS not used as truth or features (Date only).\n")
    report_lines.append("- Rolling features are shifted by 1 day (no same-day target leakage).\n")
    report_lines.append("- Validation is time-based (no random split).\n")
    report_lines.append("- Recursive forecasting used for models with lags/rolling features.\n")

    report_lines.append("## 4. Validation design\n")
    report_lines.append(
        "Primary: holdout last **548** days of sales history (matches submission horizon length).\n"
    )
    report_lines.append("Secondary (computed for robustness): holdout last 365 days, plus rolling-origin year splits for 2020/2021/2022.\n")

    report_lines.append("## 5. Model candidates\n")
    report_lines.append(
        "- Naive last value\n"
        "- Seasonal naive (364-day)\n"
        "- Seasonal naive (365-day)\n"
        "- Calendar profile (day-of-year/month/day-of-week averages)\n"
        "- ML lag models (if sklearn available): Ridge, ElasticNet, RandomForest, ExtraTrees, HistGradientBoosting, GradientBoosting\n"
    )

    report_lines.append("## 6. Metrics table (primary holdout_548)\n")
    report_lines.append(to_md_table(m_primary))
    report_lines.append("\n")

    report_lines.append("## 7. Selected model and rationale\n")
    report_lines.append(
        f"- Revenue model: **{best_revenue_model_name}** (selected by lowest RMSE then MAE on holdout_548)\n"
    )
    report_lines.append(
        f"- COGS model: **{best_cogs_model_name}** (selected by lowest RMSE then MAE on holdout_548)\n"
    )
    report_lines.append(
        "Selection prioritizes Revenue error (competition focus) while producing a reasonable COGS forecast for submission schema.\n"
    )

    report_lines.append("## 8. Explainability summary\n")
    report_lines.append(
        "Feature importance is derived from model-native importance (trees), absolute coefficients (linear), or permutation importance (fallback).\n"
    )

    def fi_block(df: pd.DataFrame, header: str) -> None:
        report_lines.append(f"### {header}\n")
        if df.empty:
            report_lines.append("(No feature importance available.)\n")
            return
        view = df[["feature", "importance", "method"]].copy()
        view["importance"] = view["importance"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "NA")
        report_lines.append(_df_to_markdown_table(view))
        report_lines.append("\n")

    fi_block(top_fi_rev, "Top features — Revenue")
    fi_block(top_fi_cogs, "Top features — COGS")

    report_lines.append("## 9. Error analysis\n")
    report_lines.append(
        "Error slices were computed on the primary holdout validation by month, day-of-week, year-month, and high vs normal days. "
        "See `artifacts/tables/forecast_009_error_analysis.csv`.\n"
    )

    report_lines.append("## 10. Submission path and format validation\n")
    report_lines.append(f"- Submission written to: `artifacts/submissions/submission_forecast_009.csv`\n")
    report_lines.append("- Schema: Date, Revenue, COGS\n")
    report_lines.append("- Row order: Date matches sample_submission exactly (asserted in code).\n")
    report_lines.append(
        f"- Sanity checks: Revenue>0, COGS>=0, and COGS capped to Revenue*{ratio_cap:.4f} (historical p99 ratio).\n"
    )

    report_lines.append("## 11. Known limitations\n")
    report_lines.append(
        "- Univariate forecasting only (no exogenous regressors beyond calendar).\n"
        "- Recursive multi-step forecasts can accumulate error across the horizon.\n"
        "- No hyperparameter tuning beyond reasonable defaults (to reduce overfitting risk).\n"
    )

    report_lines.append("## 12. Next improvement ideas\n")
    report_lines.append(
        "- Try log-transform targets (log1p) with inverse transform to stabilize variance.\n"
        "- Add robust seasonal components (e.g., STL decomposition) and model residuals.\n"
        "- Evaluate simple ensembling (e.g., average of top 2–3 models) with time-based selection.\n"
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logging.info("Wrote report: %s", report_path)

    # Final assertions for expected artifacts
    expected_files = [
        report_path,
        val_out,
        fut_out,
        sub_out,
        metrics_out,
        fi_out,
        err_out,
        out_figures / "forecast_009_validation_actual_vs_pred.png",
        out_figures / "forecast_009_validation_residuals.png",
        out_figures / "forecast_009_feature_importance.png",
        out_figures / "forecast_009_future_forecast.png",
    ]

    missing = [p for p in expected_files if not p.exists()]
    if missing:
        raise RuntimeError(f"Some expected artifacts were not created: {missing}")

    logging.info("All expected FORECAST-009 artifacts created successfully.")


if __name__ == "__main__":
    main()
