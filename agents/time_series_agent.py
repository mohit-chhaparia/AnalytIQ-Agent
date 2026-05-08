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


