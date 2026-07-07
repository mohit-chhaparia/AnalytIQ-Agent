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


