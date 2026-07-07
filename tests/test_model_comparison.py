"""
Tests for agents/model_comparison_agent.py

Verifies ranking logic, output structure, and edge-case handling.
"""

import pytest
from agents.model_comparison_agent import compare_models


# Sample result fixtures

LINEAR_HIGH = {
    "model_type": "Linear Regression",
    "aic": 120.0, "bic": 125.0,
    "r_squared": 0.91, "adj_r_squared": 0.90,
}

LINEAR_LOW = {
    "model_type": "Linear Regression",
    "aic": 210.0, "bic": 215.0,
    "r_squared": 0.45, "adj_r_squared": 0.44,
}

LOGISTIC_GOOD = {
    "model_type": "Logistic Regression",
    "aic": 85.0, "bic": 90.0,
    "metrics": {"accuracy": 0.88, "auc": 0.93, "sensitivity_recall": 0.85, "precision": 0.87},
}

LOGISTIC_POOR = {
    "model_type": "Logistic Regression",
    "aic": 155.0, "bic": 160.0,
    "metrics": {"accuracy": 0.61, "auc": 0.58, "sensitivity_recall": 0.55, "precision": 0.60},
}

POISSON_OK = {
    "model_type": "Poisson Regression",
    "aic": 310.0, "bic": 315.0,
    "dispersion": 1.1, "overdispersion_flag": False,
}


# Tests

class TestCompareModels:

    def test_empty_list_returns_gracefully(self):
        result = compare_models([])
        assert "ranked" in result
        assert result["ranked"] == []
        assert "best_model" in result

    def test_single_model_is_best(self):
        result = compare_models([LINEAR_HIGH])
        assert result["best_model"] == "Linear Regression"
        assert len(result["ranked"]) == 1

    def test_returns_all_required_keys(self):
        result = compare_models([LINEAR_HIGH, POISSON_OK])
        for key in ("ranked", "best_model", "rationale", "comparison_table"):
            assert key in result, f"Missing key: '{key}'"

    def test_better_logistic_ranked_first(self):
        result = compare_models([LOGISTIC_POOR, LOGISTIC_GOOD])
        assert result["ranked"][0]["aic"] == pytest.approx(85.0), (
            "Lower-AIC (better) logistic model should rank first"
        )

    def test_ranked_length_matches_inputs(self):
        result = compare_models([LINEAR_HIGH, LOGISTIC_GOOD, POISSON_OK])
        assert len(result["ranked"]) == 3

    
