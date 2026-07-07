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


# Binary outcome

class TestBinaryOutcome:

    def test_outcome_type_is_binary(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        assert result["outcome_type"] == "binary"

    def test_logistic_recommended(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        models = [r["model"] for r in result["recommendations"]]
        assert any("Logistic" in m for m in models), f"Expected logistic in {models}"

    def test_at_least_two_recommendations(self, binary_profile):
        result = recommend_models(binary_profile, "churn", "predict churn")
        assert len(result["recommendations"]) >= 2


# Continuous outcome

class TestContinuousOutcome:

    def test_outcome_type_is_continuous(self, continuous_profile):
        result = recommend_models(continuous_profile, "price", "predict house price")
        assert result["outcome_type"] == "continuous_numeric"

    def test_linear_regression_recommended(self, continuous_profile):
        result = recommend_models(continuous_profile, "price", "predict house price")
        models = [r["model"] for r in result["recommendations"]]
        assert any("Linear" in m for m in models), f"Expected linear in {models}"

    def test_at_least_two_recommendations(self, continuous_profile):
        result = recommend_models(continuous_profile, "price", "predict house price")
        assert len(result["recommendations"]) >= 2


# Count outcome

class TestCountOutcome:

    def test_poisson_recommended_for_discrete_outcome(self, count_profile):
        result = recommend_models(count_profile, "claims", "model count of claims")
        models = [r["model"] for r in result["recommendations"]]
        assert any("Poisson" in m for m in models), f"Expected Poisson in {models}"


# Time-series keyword routing

class TestTimeSeriesKeywords:

    def test_arima_recommended_when_goal_mentions_forecast(self, ts_profile):
        result = recommend_models(ts_profile, "sales", "forecast monthly sales time series")
        models = [r["model"] for r in result["recommendations"]]
        assert any("ARIMA" in m or "time" in m.lower() for m in models), (
            f"Expected time-series model in {models}"
        )
