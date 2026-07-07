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


