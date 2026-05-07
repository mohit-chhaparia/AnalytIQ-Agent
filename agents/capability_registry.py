from __future__ import annotations

CAPABILITY_REGISTRY: dict[str, dict] = {
    "linear_regression": {
        "supported": True,
        "engine": "python",
        "function": "run_linear_regression",
        "task_types": ["continuous_outcome", "regression", "inference"],
    },
    "logistic_regression": {
        "supported": True,
        "engine": "python",
        "function": "run_logistic_regression",
        "task_types": ["binary_outcome", "classification", "inference"],
    },
    "poisson_regression": {
        "supported": True,
        "engine": "python",
        "function": "run_poisson_regression",
        "task_types": ["count_outcome", "inference"],
    },
    "ols_anova": {
        "supported": True,
        "engine": "python",
        "function": "run_ols_anova_table",
        "task_types": ["linear_model", "anova_table", "inference"],
    },
    "tabular_ml": {
        "supported": True,
        "engine": "python",
        "function": "run_tabular_ml",
        "task_types": ["classification", "regression", "prediction"],
    },
    "time_series": {
        "supported": True,
        "engine": "python",
        "function": "run_arima_forecast",
        "task_types": ["forecasting", "arima", "seasonality"],
    },
    "anova_ancova": {
        "supported": True,
        "engine": "r",
        "function": "run_anova_ancova",
        "task_types": ["linear_model", "anova_table"],
    },
}


def find_supported_capability(requested_method: str) -> tuple[str | None, dict | None]:
    requested_method = (requested_method or "").lower().replace(" ", "_")
    for capability_name, details in CAPABILITY_REGISTRY.items():
        if requested_method in capability_name or capability_name in requested_method:
            return capability_name, details
    if "logistic" in requested_method or "logit" in requested_method:
        return "logistic_regression", CAPABILITY_REGISTRY["logistic_regression"]
    if "poisson" in requested_method:
        return "poisson_regression", CAPABILITY_REGISTRY["poisson_regression"]
    if "linear" in requested_method or "ols" in requested_method:
        return "linear_regression", CAPABILITY_REGISTRY["linear_regression"]
    if "random_forest" in requested_method or "tabular" in requested_method or "automl" in requested_method:
        return "tabular_ml", CAPABILITY_REGISTRY["tabular_ml"]
    if "arima" in requested_method or "forecast" in requested_method or "time_series" in requested_method:
        return "time_series", CAPABILITY_REGISTRY["time_series"]
    if "anova" in requested_method or "ancova" in requested_method:
        return "anova_ancova", CAPABILITY_REGISTRY["anova_ancova"]
    return None, None

