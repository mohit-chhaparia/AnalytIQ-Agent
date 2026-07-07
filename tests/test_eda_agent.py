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


