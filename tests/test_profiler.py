"""
Tests for agents/data_profiler.py

Covers: infer_variable_type, numeric_summary, profile_dataframe
(missing values, duplicates, outlier detection, type inference).
"""

import pandas as pd
import numpy as np
import pytest

from agents.data_profiler import infer_variable_type, numeric_summary, profile_dataframe


# infer_variable_type

class TestInferVariableType:

    def test_binary_int(self):
        assert infer_variable_type(pd.Series([0, 1, 0, 1])) == "binary"

    def test_binary_string(self):
        assert infer_variable_type(pd.Series(["yes", "no", "yes"])) == "binary"

    def test_constant_returns_constant(self):
        assert infer_variable_type(pd.Series([1, 1, 1])) == "constant_or_single_value"

    def test_all_missing_returns_unknown(self):
        assert infer_variable_type(pd.Series([np.nan, np.nan])) == "unknown"

    def test_continuous_numeric(self):
        s = pd.Series(np.linspace(0, 100, 50))
        assert infer_variable_type(s) == "continuous_numeric"

    def test_low_cardinality_numeric_is_discrete(self):
        s = pd.Series([1, 2, 3, 4, 5, 1, 2, 3] * 5)
        assert infer_variable_type(s) == "numeric_discrete_or_categorical"

    def test_categorical_string(self):
        s = pd.Series(["A", "B", "C", "A", "B"] * 4)
        assert infer_variable_type(s) == "categorical"

    def test_high_cardinality_text(self):
        s = pd.Series([f"user_{i}" for i in range(100)])
        assert infer_variable_type(s) == "text_or_high_cardinality_categorical"

    def test_date_column_detected(self):
        s = pd.Series(pd.date_range("2020-01-01", periods=20, freq="D"))
        assert infer_variable_type(s) == "date_or_datetime"

    def test_string_dates_detected(self):
        s = pd.Series(["2020-01-01", "2020-01-02", "2020-01-03"] * 5)
        assert infer_variable_type(s) == "date_or_datetime"


# numeric_summary

class TestNumericSummary:
    REQUIRED_KEYS = ("mean", "std", "min", "q1", "median", "q3", "max", "outlier_count_iqr")

    def test_returns_required_keys(self):
        s = pd.Series(range(20), dtype=float)
        result = numeric_summary(s)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_mean_correct(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert numeric_summary(s)["mean"] == pytest.approx(3.0)

    def test_median_correct(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert numeric_summary(s)["median"] == pytest.approx(3.0)

    def test_outlier_count_correct(self):
        # IQR method: values far outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are outliers
        s = pd.Series([1.0] * 20 + [1000.0])   # 1000 is a clear outlier
        result = numeric_summary(s)
        assert result["outlier_count_iqr"] >= 1

    def test_no_outliers_in_uniform_data(self):
        s = pd.Series(np.linspace(0, 10, 30))
        result = numeric_summary(s)
        assert result["outlier_count_iqr"] == 0

    def test_empty_series_returns_empty(self):
        result = numeric_summary(pd.Series([], dtype=float))
        assert result == {}

    def test_missing_values_ignored(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = numeric_summary(s)
        assert result["mean"] == pytest.approx(3.0)


