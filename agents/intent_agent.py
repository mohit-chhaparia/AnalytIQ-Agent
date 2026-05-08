"""Infer high-level analysis mode from goal text and data profile (orchestration / AI layer)."""

from __future__ import annotations


def infer_analysis_modes(goal: str, profile: dict, outcome: str) -> dict:
    """
    Returns suggested workflow modes. Multiple can be true (e.g. TS + ML).
    This is heuristic routing for the controller/UI, not a black-box model.
    """
    g = (goal or "").lower()
    columns = profile.get("columns", [])
    outcome_info = next((c for c in columns if c["name"] == outcome), None)
    outcome_type = (outcome_info or {}).get("inferred_type", "")

    has_datetime = any(c.get("inferred_type") == "date_or_datetime" for c in columns)
    ts_keywords = (
        "time series",
        "forecast",
        "arima",
        "seasonal",
        "temporal",
        "trend",
        "lag",
        "acf",
        "pacf",
    )
    ml_keywords = (
        "machine learning",
        "random forest",
        "xgboost",
        "gradient boosting",
        "feature importance",
        "cross-validation",
        "cross validation",
        "predictive",
        "classification accuracy",
        "auc",
        "tabular",
    )
    inference_keywords = (
        "coefficient",
        "confidence interval",
        "p-value",
        "anova",
        "interpretable",
        "glm",
        "logistic",
        "effect",
        "causal",
        "inference",
    )

    suggest_ts = has_datetime or any(k in g for k in ts_keywords)
    suggest_ml = any(k in g for k in ml_keywords) or (
        "predict" in g and "interpret" not in g and outcome_type in ("binary", "continuous_numeric")
    )
    suggest_inference = (
        any(k in g for k in inference_keywords)
        or outcome_type in ("binary", "continuous_numeric", "numeric_discrete_or_categorical")
    )

    primary = "inference_glm"
    if suggest_ts and has_datetime:
        primary = "time_series"
    elif suggest_ml and not suggest_inference:
        primary = "tabular_ml"
    elif suggest_ml and suggest_inference:
        primary = "mixed"

    return {
        "primary_mode": primary,
        "flags": {
            "time_series": suggest_ts,
            "tabular_ml": suggest_ml,
            "statistical_inference": suggest_inference,
        },
        "has_datetime_column": has_datetime,
        "rationale": "Heuristic keyword + schema routing for tool selection.",
    }
