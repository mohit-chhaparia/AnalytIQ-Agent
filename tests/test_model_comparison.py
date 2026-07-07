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

