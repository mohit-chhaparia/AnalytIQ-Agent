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


