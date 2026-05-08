"""Heuristic model recommendations from profile, outcome, and stated goal."""

from __future__ import annotations


def recommend_models(profile: dict, outcome: str, goal: str = "") -> dict:
    columns = profile.get("columns", [])
    outcome_info = next((c for c in columns if c["name"] == outcome), None)
    if outcome_info is None:
        return {"error": f"Outcome variable '{outcome}' not found."}

    outcome_type = outcome_info["inferred_type"]
    recommendations: list[dict] = []
    goal_l = (goal or "").lower()

    if outcome_type == "binary":
        recommendations.extend(
            [
                {
                    "model": "Logistic Regression",
                    "reason": "Interpretable baseline for a binary outcome.",
                    "python_engine": "statsmodels GLM Binomial",
                    "r_engine": "glm(..., family = binomial)",
                },
                {
                    "model": "Regularized / tree ensembles",
                    "reason": "Strong predictive performance and nonlinearities; use for ML baselines.",
                    "python_engine": "scikit-learn / histogram gradient boosting",
                    "r_engine": "ranger / xgboost",
                },
            ]
        )
    elif outcome_type == "continuous_numeric":
        recommendations.extend(
            [
                {
                    "model": "Linear Regression",
                    "reason": "Interpretable linear structure and classical inference.",
                    "python_engine": "statsmodels OLS",
                    "r_engine": "lm",
                },
                {
                    "model": "Gradient boosting / random forest",
                    "reason": "Flexible tabular regression when prediction quality dominates.",
                    "python_engine": "sklearn RandomForest / HistGradientBoosting",
                    "r_engine": "xgboost / ranger",
                },
            ]
        )
    elif outcome_type in ("numeric_discrete_or_categorical",):
        recommendations.extend(
            [
                {
                    "model": "Poisson Regression",
                    "reason": "Natural starting point for nonnegative counts.",
                    "python_engine": "statsmodels GLM Poisson",
                    "r_engine": "glm(..., family = poisson)",
                },
                {
                    "model": "Quasi-Poisson / Negative Binomial",
                    "reason": "Use if counts show overdispersion.",
                    "python_engine": "statsmodels NegativeBinomial",
                    "r_engine": "MASS::glm.nb",
                },
            ]
        )
    else:
        recommendations.append(
            {
                "model": "Manual review",
                "reason": "Outcome type is ambiguous; consider encoding, time index, or multivariate targets.",
                "python_engine": "—",
                "r_engine": "—",
            }
        )

    if any(k in goal_l for k in ("time series", "forecast", "arima", "seasonal", "temporal")):
        recommendations.append(
            {
                "model": "ARIMA / exponential smoothing",
                "reason": "Goal suggests temporal structure; model the series or residuals explicitly.",
                "python_engine": "statsmodels ARIMA / ETS",
                "r_engine": "forecast package",
            }
        )

    if "survival" in goal_l or "cox" in goal_l or "hazard" in goal_l:
        recommendations.append(
            {
                "model": "Survival (Cox / KM)",
                "reason": "Time-to-event modeling.",
                "python_engine": "lifelines (optional)",
                "r_engine": "survival::coxph",
            }
        )

    return {
        "outcome": outcome,
        "outcome_type": outcome_type,
        "recommendations": recommendations,
    }
