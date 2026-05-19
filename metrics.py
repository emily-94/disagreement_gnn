
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def _check_finite(arr, name: str):
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))[0][:5]
        raise ValueError(f"{name} has non-finite values at indices {bad}")
    return arr


def regression_metrics_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    y_true = _check_finite(y_true, "y_true")
    y_pred = _check_finite(y_pred, "y_pred")

    err  = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2     = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    # Spearman via rank correlation (average ranks for ties)
    rt = pd.Series(y_true).rank(method="average").to_numpy()
    rp = pd.Series(y_pred).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rt, rp)[0, 1]) if len(rt) > 1 else np.nan

    return {"N": int(len(y_true)), "MAE": mae, "RMSE": rmse, "R2": r2, "Spearman": spearman}


def regression_metrics_df(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Dict[str, float]:
    y_true = pd.to_numeric(df[y_true_col], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(df[y_pred_col], errors="coerce").to_numpy()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return regression_metrics_arrays(y_true[m], y_pred[m])
