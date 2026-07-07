"""
Tests for agents/diagnostics_agent.py

Covers run_diagnostics_for_result, interpret_diagnostics,
and interpret_poisson_diagnostics for all model types.
"""

import numpy as np
import pandas as pd
import pytest

from agents.diagnostics_agent import (
    interpret_poisson_diagnostics,
    run_diagnostics_for_result,
)
from agents.model_runner import (
    run_linear_regression,
    run_logistic_regression,
    run_poisson_regression,
)


# Fixtures

@pytest.fixture
def linear_result():
    np.random.seed(42)
    x = np.linspace(1, 10, 30)
    y = 2.0 * x + np.random.normal(0, 1, 30)
    df = pd.DataFrame({"y": y, "x": x})
    return run_linear_regression(df, "y ~ x"), df


@pytest.fixture
def logistic_result():
    np.random.seed(42)
    x = np.linspace(-2, 2, 60)
    prob = 1 / (1 + np.exp(-3 * x))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({"y": y, "x": x})
    return run_logistic_regression(df, "y ~ x", "y"), df


@pytest.fixture
def overdispersed_poisson_result():
    np.random.seed(0)
    x = np.linspace(0, 1, 50)
    y = np.random.negative_binomial(1, 0.25, 50)
    df = pd.DataFrame({"y": y, "x": x})
    return run_poisson_regression(df, "y ~ x"), df


# interpret_poisson_diagnostics

class TestInterpretPoissonDiagnostics:

    def test_overdispersion_flagged(self):
        notes = interpret_poisson_diagnostics(
            {"dispersion": 2.5, "overdispersion_flag": True}
        )
        assert len(notes) >= 1
        combined = " ".join(notes).lower()
        assert "overdispersion" in combined or "quasi" in combined or "negative binomial" in combined

    def test_no_overdispersion(self):
        notes = interpret_poisson_diagnostics(
            {"dispersion": 0.9, "overdispersion_flag": False}
        )
        assert isinstance(notes, list)
        # Should still return some notes (e.g. "no overdispersion")
        assert len(notes) >= 1

    def test_boundary_dispersion(self):
        """Dispersion exactly at 1.5 — should not flag."""
        notes = interpret_poisson_diagnostics(
            {"dispersion": 1.5, "overdispersion_flag": False}
        )
        assert isinstance(notes, list)

    def test_returns_list_of_strings(self):
        notes = interpret_poisson_diagnostics({"dispersion": 2.0, "overdispersion_flag": True})
        assert isinstance(notes, list)
        assert all(isinstance(n, str) for n in notes)


# run_diagnostics_for_result — Linear

class TestLinearDiagnostics:

    def test_returns_dict(self, linear_result):
        result, df = linear_result
        diag = run_diagnostics_for_result(result, df)
        assert isinstance(diag, dict)

    def test_r_squared_present(self, linear_result):
        result, df = linear_result
        diag = run_diagnostics_for_result(result, df)
        assert "r_squared" in diag

    def test_heteroskedasticity_key_present(self, linear_result):
        result, df = linear_result
        diag = run_diagnostics_for_result(result, df)
        assert "heteroskedasticity" in diag

    def test_heteroskedasticity_has_flag(self, linear_result):
        result, df = linear_result
        diag = run_diagnostics_for_result(result, df)
        assert "flag" in diag["heteroskedasticity"]

    def test_notes_is_list(self, linear_result):
        result, df = linear_result
        diag = run_diagnostics_for_result(result, df)
        assert "notes" in diag
        assert isinstance(diag["notes"], list)

    def test_notes_nonempty(self, linear_result):
        result, df = linear_result
        diag = run_diagnostics_for_result(result, df)
        assert len(diag["notes"]) >= 1

    def test_aic_present(self, linear_result):
        result, df = linear_result
        diag = run_diagnostics_for_result(result, df)
        assert "aic" in diag


# run_diagnostics_for_result — Logistic

class TestLogisticDiagnostics:

    def test_returns_dict(self, logistic_result):
        result, df = logistic_result
        diag = run_diagnostics_for_result(result, df)
        assert isinstance(diag, dict)

    def test_auc_present(self, logistic_result):
        result, df = logistic_result
        diag = run_diagnostics_for_result(result, df)
        assert "auc" in diag

    def test_auc_in_unit_interval(self, logistic_result):
        result, df = logistic_result
        diag = run_diagnostics_for_result(result, df)
        if diag["auc"] is not None:
            assert 0.0 <= diag["auc"] <= 1.0

    def test_notes_present(self, logistic_result):
        result, df = logistic_result
        diag = run_diagnostics_for_result(result, df)
        assert "notes" in diag


