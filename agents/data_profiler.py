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
