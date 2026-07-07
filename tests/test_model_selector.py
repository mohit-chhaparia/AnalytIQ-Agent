"""
Tests for agents/model_selector.py

Verifies recommendations for binary, continuous, count, and
time-series outcomes, plus error handling.
"""

import pandas as pd
import pytest

from agents.data_profiler import profile_dataframe
from agents.model_selector import recommend_models


# Fixtures

@pytest.fixture
def binary_profile():
    df = pd.DataFrame({
        "churn":   [0, 1] * 25,
        "tenure":  range(50),
        "charges": [float(i) for i in range(50)],
    })
    return profile_dataframe(df)


@pytest.fixture
def continuous_profile():
    df = pd.DataFrame({
        "price": [float(i) * 2.0 for i in range(50)],
        "sqft":  range(50),
        "beds":  [2, 3, 4] * 16 + [2, 3],
    })
    return profile_dataframe(df)


@pytest.fixture
def count_profile():
    df = pd.DataFrame({
        "claims": list(range(10)) * 5,          # 0–9 repeated → discrete
        "age":    list(range(20, 70))[:50],
    })
    return profile_dataframe(df)


@pytest.fixture
def ts_profile():
    """Profile where goal mentions 'forecast' / 'time series'."""
    import numpy as np
    df = pd.DataFrame({
        "sales":  [float(i) + 0.5 * i for i in range(50)],
        "month":  range(50),
    })
    return profile_dataframe(df)


# Output contract

class TestOutputContract:

    def test_returns_dict(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        assert isinstance(result, dict)

    def test_no_error_key_for_valid_input(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        assert "error" not in result

    def test_error_returned_for_missing_outcome(self, binary_profile):
        result = recommend_models(binary_profile, "nonexistent_col", "predict churn")
        assert "error" in result

    def test_recommendations_is_list(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        assert isinstance(result["recommendations"], list)

    def test_each_recommendation_has_required_keys(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        for rec in result["recommendations"]:
            for key in ("model", "reason", "python_engine", "r_engine"):
                assert key in rec, f"Recommendation missing '{key}': {rec}"

    def test_outcome_type_present(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        assert "outcome_type" in result


