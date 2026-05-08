"""High-level analysis plan from profile, outcome, and user goal."""

from __future__ import annotations

from agents.eda_agent import recommend_eda_plots
from agents.intent_agent import infer_analysis_modes
from agents.model_selector import recommend_models


def build_analysis_plan(profile: dict, outcome: str, goal: str) -> dict:
    model_recs = recommend_models(profile, outcome, goal)
    eda = recommend_eda_plots(profile, outcome)
    outcome_type = model_recs.get("outcome_type", "unknown")
    intent = infer_analysis_modes(goal, profile, outcome)
    steps = [
        "Profile dataset (types, missingness, duplicates, outliers).",
        "Review EDA visualizations for outcome–predictor relationships.",
    ]

    if intent["flags"].get("time_series"):
        steps.append(
            "If a time index exists: check stationarity (ADF), ACF/PACF, then fit ARIMA or smoothing models."
        )
    if intent["flags"].get("tabular_ml"):
        steps.append(
            "For prediction-focused goals: preprocess categoricals, cross-validate, and compare tree ensembles."
        )
    if intent["flags"].get("statistical_inference"):
        steps.append(
            "For inference: prefer GLM/OLS with explicit formulas, assumption checks, and coefficient interpretation."
        )

    if outcome_type == "binary":
        steps.append("For interpretability: start with logistic regression; calibrate thresholds for asymmetric costs.")
    elif outcome_type == "continuous_numeric":
        steps.append("For continuous outcomes: OLS or regularized regression; validate with residual diagnostics.")
    elif outcome_type == "numeric_discrete_or_categorical":
        steps.append("For counts: Poisson/negative binomial GLMs; check overdispersion.")
    else:
        steps.append("Clarify outcome encoding or add a time column for forecasting workflows.")

    steps.append("Compare candidate approaches with held-out or cross-validated metrics plus domain checks.")
    steps.append("Export templated report (HTML via Quarto when installed).")

    return {
        "outcome": outcome,
        "outcome_type": outcome_type,
        "user_goal": goal,
        "intent": intent,
        "recommended_models": model_recs.get("recommendations", []),
        "eda_plots": eda,
        "steps": steps,
    }
