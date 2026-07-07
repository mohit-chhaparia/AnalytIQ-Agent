"""
Tests for agents/model_runner.py

Covers the three deterministic Python engines:
  run_linear_regression, run_logistic_regression, run_poisson_regression
"""

import numpy as np
import pandas as pd
import pytest

from agents.model_runner import (
    run_linear_regression,
    run_logistic_regression,
    run_poisson_regression,
)


# Fixtures

@pytest.fixture
def linear_df():
    """30-row dataset with a clear linear signal (R² should be > 0.95)."""
    np.random.seed(42)
    x = np.arange(1, 31, dtype=float)
    y = 3.0 * x + 5.0 + np.random.normal(0, 1.5, 30)
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def logistic_df():
    """60-row binary dataset with a strong but non-perfect predictor."""
    np.random.seed(42)
    x = np.linspace(-3, 3, 60)
    prob = 1 / (1 + np.exp(-2 * x))
    y = np.random.binomial(1, prob)
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def logistic_string_df():
    """Logistic dataset where the outcome is a string ('no' / 'yes').

    Uses a list comprehension (not np.where) so pandas stores the column
    as dtype=object — matching what pd.read_csv() produces in practice.
    np.where returns dtype <U3 (numpy unicode) which on Python 3.11 +
    numpy 2.x can bypass the model_runner dtype==object encoding check.
    """
    np.random.seed(7)
    x = np.linspace(-2, 2, 48)
    y = ["yes" if xi > 0 else "no" for xi in x]   # Python strings -> object dtype
    return pd.DataFrame({"outcome": y, "x": x})


@pytest.fixture
def poisson_df():
    """30-row count dataset — well-specified Poisson."""
    np.random.seed(42)
    x = np.linspace(0, 2, 30)
    y = np.random.poisson(np.exp(0.5 + 0.8 * x))
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def overdispersed_df():
    """Count data drawn from a negative binomial — almost always overdispersed."""
    np.random.seed(0)
    x = np.linspace(0, 1, 50)
    y = np.random.negative_binomial(1, 0.25, 50)
    return pd.DataFrame({"y": y, "x": x})


# Linear regression

class TestLinearRegression:
    REQUIRED_KEYS = (
        "model_type", "formula", "summary",
        "aic", "bic", "r_squared", "adj_r_squared",
        "residuals", "fitted_values",
    )

    def test_returns_required_keys(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_model_type_label(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert result["model_type"] == "Linear Regression"

    def test_formula_preserved(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert result["formula"] == "y ~ x"

    def test_r_squared_in_unit_interval(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_r_squared_high_for_linear_data(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert result["r_squared"] > 0.95, (
            f"Expected R² > 0.95 for clean linear data, got {result['r_squared']:.4f}"
        )

    def test_adj_r_squared_le_r_squared(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert result["adj_r_squared"] <= result["r_squared"]

    def test_residuals_length_matches_rows(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert len(result["residuals"]) == len(linear_df)

    def test_fitted_values_length_matches_rows(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert len(result["fitted_values"]) == len(linear_df)

    def test_aic_and_bic_are_finite(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert np.isfinite(result["aic"]), "AIC should be finite"
        assert np.isfinite(result["bic"]), "BIC should be finite"

    def test_residuals_sum_near_zero(self, linear_df):
        """OLS residuals must sum to (approx) zero."""
        result = run_linear_regression(linear_df, "y ~ x")
        assert abs(sum(result["residuals"])) < 1e-8

    def test_summary_is_nonempty_string(self, linear_df):
        result = run_linear_regression(linear_df, "y ~ x")
        assert isinstance(result["summary"], str) and len(result["summary"]) > 0


# Logistic regression

class TestLogisticRegression:
    REQUIRED_KEYS = (
        "model_type", "formula", "summary",
        "aic", "bic", "metrics", "predicted_probabilities",
    )
    REQUIRED_METRICS = ("accuracy", "sensitivity_recall", "precision", "auc")

    def test_returns_required_keys(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: '{key}'"

    def test_model_type_label(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        assert result["model_type"] == "Logistic Regression"

    def test_metrics_keys_present(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        for key in self.REQUIRED_METRICS:
            assert key in result["metrics"], f"Missing metric: '{key}'"

    def test_auc_in_unit_interval(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        assert 0.0 <= result["metrics"]["auc"] <= 1.0

    def test_auc_above_chance_for_good_predictor(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        assert result["metrics"]["auc"] > 0.7, (
            f"Expected AUC > 0.7 for clear signal, got {result['metrics']['auc']:.4f}"
        )

    def test_accuracy_in_unit_interval(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    def test_predicted_probabilities_in_unit_interval(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        probs = result["predicted_probabilities"]
        assert all(0.0 <= p <= 1.0 for p in probs), "All probs must be in [0, 1]"

    def test_predicted_probabilities_length(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        assert len(result["predicted_probabilities"]) == len(logistic_df)

    def test_string_outcome_encoded_without_error(self, logistic_string_df):
        """String 'yes'/'no' outcomes should be encoded automatically."""
        result = run_logistic_regression(logistic_string_df, "outcome ~ x", "outcome")
        assert result["model_type"] == "Logistic Regression"
        assert 0.0 <= result["metrics"]["auc"] <= 1.0

    def test_aic_is_finite(self, logistic_df):
        result = run_logistic_regression(logistic_df, "y ~ x", "y")
        assert np.isfinite(result["aic"])


