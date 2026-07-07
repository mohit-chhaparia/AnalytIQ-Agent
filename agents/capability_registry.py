"""
Central registry of all deterministic analysis tools available to the agent.
When a user requests an analysis, the controller checks here first before falling back to dynamic code generation.
"""

CAPABILITY_REGISTRY = {
    "linear_regression": {
        "supported": True,
        "engine": "python",
        "function": "run_linear_regression",
        "module": "agents.model_runner",
        "task_types": ["continuous_outcome", "regression", "ols", "ordinary least squares"],
        "description": "Fit OLS linear regression with diagnostics (R², AIC, BIC, residual plots).",
    },
    "logistic_regression": {
        "supported": True,
        "engine": "python",
        "function": "run_logistic_regression",
        "module": "agents.model_runner",
        "task_types": ["binary_outcome", "classification", "churn", "binary", "logistic"],
        "description": "Fit binomial logistic regression; outputs AUC, confusion matrix, threshold tuning.",
    },
    "poisson_regression": {
        "supported": True,
        "engine": "python",
        "function": "run_poisson_regression",
        "module": "agents.model_runner",
        "task_types": ["count_outcome", "count data", "poisson", "count"],
        "description": "Fit Poisson GLM for count outcomes; flags overdispersion.",
    },
    "anova_ancova": {
        "supported": True,
        "engine": "r",
        "function": "run_anova_ancova",
        "module": "r_engine/run_anova_ancova.R",
        "task_types": ["doe", "factorial", "ancova", "anova", "treatment", "experimental design"],
        "description": "ANOVA / ANCOVA via R (car::Anova); supports CRD, RCBD, factorial, covariate adjustment.",
    },
    "time_series": {
        "supported": True,
        "engine": "python",
        "function": "run_time_series",
        "module": "agents.model_runner",
        "task_types": ["time series", "forecasting", "trend", "seasonal", "arima", "temporal"],
        "description": "Time series decomposition, trend analysis, and ARIMA forecasting.",
    },
    "automl_pycaret": {
        "supported": True,
        "engine": "python",
        "function": "run_automl",
        "module": "agents.model_runner",
        "task_types": ["automl", "compare models", "best model", "model comparison", "leaderboard"],
        "description": "PyCaret AutoML: compares multiple classifiers/regressors and returns a leaderboard.",
    },
}

# ── analyses NOT yet in registry (handled by dynamic agent) ─────────────────
KNOWN_UNSUPPORTED = [
    "survival analysis", "cox model", "time-to-event",
    "mixed effects", "lme4", "random effects", "multilevel",
    "bayesian", "mcmc",
    "pca", "principal component",
    "cluster", "kmeans", "hierarchical",
    "neural network", "deep learning",
]


def find_capability(user_request: str):
    """
    Match a free-text user request to a registry entry.
    Uses longest-match scoring so 'logistic regression' beats 'regression'.

    Returns (capability_name: str | None, details: dict | None)
    """
    text = user_request.lower()
    best_name    = None
    best_details = None
    best_score   = 0

    for name, details in CAPABILITY_REGISTRY.items():
        keywords = [name.replace("_", " ")] + details["task_types"]
        for kw in keywords:
            if kw in text:
                score = len(kw)          # longer match = more specific = wins
                if score > best_score:
                    best_score   = score
                    best_name    = name
                    best_details = details

    return best_name, best_details


def is_known_unsupported(user_request: str) -> bool:
    """Return True if request matches a known-unsupported but valid analysis type."""
    text = user_request.lower()
    return any(kw in text for kw in KNOWN_UNSUPPORTED)


def list_capabilities() -> list:
    """Return human-readable list of supported analyses."""
    return [
        f"• {name.replace('_', ' ').title()}: {d['description']}"
        for name, d in CAPABILITY_REGISTRY.items()
        if d["supported"]
    ]
