"""Time series: univariate characterization and simple ARIMA-style forecast (statsmodels)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
except ImportError:  # pragma: no cover
    ARIMA = None  # type: ignore[misc, assignment]


def recommend_time_series_columns(profile: dict) -> dict[str, Any]:
    """Suggest datetime index and target from profile."""
    dt_cols = [c["name"] for c in profile.get("columns", []) if c.get("inferred_type") == "date_or_datetime"]
    numeric = [
        c["name"]
        for c in profile.get("columns", [])
        if c.get("inferred_type") == "continuous_numeric"
    ]
    return {"datetime_columns": dt_cols, "suggested_numeric_targets": numeric[:5]}


def run_time_series_summary(series: pd.Series, max_lags: int = 20) -> dict[str, Any]:
    """Stationarity check and ACF/PACF at a small lag window."""
    y = pd.to_numeric(series, errors="coerce").dropna()
    if len(y) < 12:
        raise ValueError("Need at least 12 observations for a minimal time series summary.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, *_ = adfuller(y.values, autolag="AIC")

    max_lags = min(max_lags, len(y) // 2 - 1, 40)
    max_lags = max(5, max_lags)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf_vals = acf(y.values, nlags=max_lags, fft=True)
        pacf_vals = pacf(y.values, nlags=max_lags, method="ywm")

    return {
        "model_type": "Time series characterization",
        "n_obs": int(len(y)),
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "stationarity_hint": "likely stationary" if adf_p < 0.05 else "unit root not rejected at 5%",
        "acf_lags": list(range(len(acf_vals))),
        "acf": [float(x) for x in acf_vals],
        "pacf": [float(x) for x in pacf_vals],
    }


def run_arima_forecast(
    series: pd.Series,
    order: tuple[int, int, int],
    steps: int = 5,
) -> dict[str, Any]:
    """Fit ARIMA(order) and return in-sample summary + forecast."""
    if ARIMA is None:
        raise ImportError("statsmodels ARIMA is required for forecasting.")
    y = pd.to_numeric(series, errors="coerce").dropna()
    if len(y) < max(15, sum(order) + 5):
        raise ValueError("Insufficient history for the requested ARIMA order.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y.values, order=order).fit()
    fc = model.get_forecast(steps=steps)
    mean_fc = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)

    return {
        "model_type": f"ARIMA{order}",
        "order": order,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "summary_tail": model.summary().as_text()[-4000:],
        "forecast_mean": [float(x) for x in mean_fc],
        "forecast_ci_lower": [float(x) for x in conf[:, 0]],
        "forecast_ci_upper": [float(x) for x in conf[:, 1]],
        "_fitted": model,
    }
