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
