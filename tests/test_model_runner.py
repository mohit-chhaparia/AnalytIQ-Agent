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


