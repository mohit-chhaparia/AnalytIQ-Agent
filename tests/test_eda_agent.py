"""
Tests for agents/eda_agent.py

Verifies that recommend_eda_plots returns well-structured, useful
recommendations for different outcome types.
"""

import pandas as pd
import numpy as np
import pytest

from agents.data_profiler import profile_dataframe
from agents.eda_agent import recommend_eda_plots


# Fixtures

@pytest.fixture
def binary_profile_and_outcome():
    df = pd.DataFrame({
        "churn":    ([0, 1] * 25),
        "tenure":   range(50),
        "charges":  [float(i) * 1.5 for i in range(50)],
        "contract": (["Month"] * 25 + ["Year"] * 25),
    })
    return profile_dataframe(df), "churn"


@pytest.fixture
def continuous_profile_and_outcome():
    df = pd.DataFrame({
        "price":    [float(i) * 2.0 for i in range(50)],
        "sqft":     range(50),
        "bedrooms": ([2, 3, 4] * 16 + [2, 3]),
        "location": (["A", "B", "C", "D", "E"] * 10),
    })
    return profile_dataframe(df), "price"


@pytest.fixture
def count_profile_and_outcome():
    np.random.seed(1)
    df = pd.DataFrame({
        "claims": np.random.poisson(3, 50),
        "age":    np.random.randint(20, 70, 50),
        "region": (["N", "S", "E", "W"] * 12 + ["N", "S"]),
    })
    return profile_dataframe(df), "claims"


# Tests

class TestRecommendEdaPlots:

    def test_returns_a_list(self, binary_profile_and_outcome):
        profile, outcome = binary_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        assert isinstance(result, list)

    def test_at_least_one_recommendation(self, binary_profile_and_outcome):
        profile, outcome = binary_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        assert len(result) >= 1

    def test_each_item_has_plot_key(self, binary_profile_and_outcome):
        profile, outcome = binary_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        for item in result:
            assert "plot" in item, f"Recommendation missing 'plot' key: {item}"

    def test_plot_values_are_strings(self, binary_profile_and_outcome):
        profile, outcome = binary_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        for item in result:
            assert isinstance(item["plot"], str), f"'plot' should be a str, got {type(item['plot'])}"

    def test_binary_outcome_recommendations_non_empty(self, binary_profile_and_outcome):
        profile, outcome = binary_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        assert len(result) >= 1

    def test_continuous_outcome_recommendations_non_empty(self, continuous_profile_and_outcome):
        profile, outcome = continuous_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        assert len(result) >= 1

    def test_count_outcome_recommendations_non_empty(self, count_profile_and_outcome):
        profile, outcome = count_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        assert len(result) >= 1

    def test_outcome_not_suggested_as_own_predictor(self, binary_profile_and_outcome):
        """The outcome column shouldn't appear as both x and y in the same recommendation."""
        profile, outcome = binary_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        for item in result:
            x, y = item.get("x"), item.get("y")
            assert not (x == outcome and y == outcome), (
                f"Outcome '{outcome}' should not be both x and y: {item}"
            )

    def test_no_duplicate_plot_recommendations(self, continuous_profile_and_outcome):
        """The same (plot, x, y) triple shouldn't be recommended twice."""
        profile, outcome = continuous_profile_and_outcome
        result = recommend_eda_plots(profile, outcome)
        seen = set()
        for item in result:
            key = (item.get("plot"), item.get("x"), item.get("y"))
            assert key not in seen, f"Duplicate recommendation: {item}"
            seen.add(key)
