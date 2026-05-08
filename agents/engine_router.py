"""Route dynamic or specialized analyses to Python or R."""

from __future__ import annotations

R_FRIENDLY = [
    "anova",
    "ancova",
    "mixed effects",
    "mixed model",
    "repeated measures",
    "emmeans",
    "nonparametric",
    "survival",
    "cox",
    "hazard",
]
PYTHON_FRIENDLY = [
    "machine learning",
    "classification",
    "regression",
    "automl",
    "clustering",
    "feature importance",
    "cross validation",
    "gradient boosting",
    "random forest",
    "time series",
    "forecast",
    "arima",
]


def choose_engine(
    analysis_type: str,
    user_preference: str | None = None,
