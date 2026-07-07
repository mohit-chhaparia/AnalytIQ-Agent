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


