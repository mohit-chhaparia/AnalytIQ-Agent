"""
Integration tests for agents/controller_agent.py

These tests run the full deterministic pipeline (use_llm=False, use_dynamic=False)
on small synthetic datasets and assert that each key stage completes and
leaves expected artifacts in memory.

Implementation notes:
- The controller stores fitted models in memory["fitted_models"], NOT "model_results".
- Goal routing via infer_goal_type() is keyword-based:
    "predict"/"churn"/etc. -> "classification" -> logistic_regression
    No keywords + continuous outcome  -> "regression" -> linear_regression
    "count"/"poisson" keywords        -> "count_modeling" -> poisson_regression
- automl_pycaret is always absent in CI (not in requirements-ci.txt); it
  fails silently inside run_models, so fitted may have 1 result instead of 2.
"""

import numpy as np
import pandas as pd
import pytest

from agents.controller_agent import StatisticalAnalysisAgent


# Fixtures

@pytest.fixture
def linear_df():
    np.random.seed(42)
    x = np.linspace(1, 10, 30)
    y = 2.5 * x + 1.0 + np.random.normal(0, 1.5, 30)
    return pd.DataFrame({"sales": y, "price": x})


@pytest.fixture
def logistic_df():
    np.random.seed(42)
    x = np.linspace(-2, 2, 60)
    prob = 1 / (1 + np.exp(-3 * x))
    y = np.random.binomial(1, prob)
    return pd.DataFrame({"churn": y, "score": x})


@pytest.fixture
def poisson_df():
    np.random.seed(42)
    x = np.linspace(0, 2, 40)
    y = np.random.poisson(np.exp(0.3 + 0.5 * x))
    return pd.DataFrame({"claims": y, "age_group": x})


# Helper

def _run(df, goal, outcome):
    """Run the agent with LLM and dynamic analysis disabled."""
    agent = StatisticalAnalysisAgent(
        df=df,
        user_goal=goal,
        outcome=outcome,
        use_llm=False,
        use_dynamic=False,
    )
    return agent.run()


def _fitted(memory):
    """Return fitted models from the correct memory key."""
    return memory.get("fitted_models", [])


# Memory structure

class TestMemoryStructure:

    def test_run_returns_dict(self, linear_df):
        memory = _run(linear_df, "Analyze relationship between sales and price", "sales")
        assert isinstance(memory, dict)

    def test_profile_present_in_memory(self, linear_df):
        memory = _run(linear_df, "Analyze relationship between sales and price", "sales")
        assert "profile" in memory

    def test_profile_shape_correct(self, linear_df):
        memory = _run(linear_df, "Analyze relationship between sales and price", "sales")
        assert memory["profile"]["shape"]["rows"] == 30
        assert memory["profile"]["shape"]["columns"] == 2

    def test_logs_are_populated(self, linear_df):
        memory = _run(linear_df, "Analyze relationship between sales and price", "sales")
        assert "logs" in memory
        assert len(memory["logs"]) > 0

    def test_profile_step_logged(self, linear_df):
        memory = _run(linear_df, "Analyze relationship between sales and price", "sales")
        assert any("profile_data" in log for log in memory["logs"])

    def test_eda_step_logged(self, linear_df):
        memory = _run(linear_df, "Analyze relationship between sales and price", "sales")
        assert any("eda" in log for log in memory["logs"])


