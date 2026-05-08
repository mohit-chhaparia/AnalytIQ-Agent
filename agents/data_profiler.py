"""Data profiling: types, missingness, duplicates, outliers, suspicious categories."""

from __future__ import annotations

import pandas as pd


def infer_variable_type(series: pd.Series) -> str:
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return "unknown"
    unique_count = non_missing.nunique()
    if unique_count <= 1:
        return "constant_or_single_value"
    if unique_count == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        if unique_count <= 10:
            return "numeric_discrete_or_categorical"
        return "continuous_numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date_or_datetime"
    parsed_dates = pd.to_datetime(non_missing, errors="coerce")
    if parsed_dates.notna().mean() > 0.8:
        return "date_or_datetime"
    if unique_count <= 20:
        return "categorical"
    return "text_or_high_cardinality_categorical"


def numeric_summary(series: pd.Series) -> dict:
    x = series.dropna()
    if len(x) == 0:
        return {}
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((x < lower) | (x > upper)).sum())
    return {
        "mean": round(float(x.mean()), 4),
        "std": round(float(x.std(ddof=1)), 4) if len(x) > 1 else 0.0,
        "min": round(float(x.min()), 4),
        "q1": round(float(q1), 4),
        "median": round(float(x.median()), 4),
        "q3": round(float(q3), 4),
        "max": round(float(x.max()), 4),
        "outlier_count_iqr": outlier_count,
    }
