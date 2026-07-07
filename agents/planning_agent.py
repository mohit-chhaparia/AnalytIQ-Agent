"""
Creates a structured analysis plan given the dataset profile, user goal, and outcome variable.

V1: Rule-based plan generation (no LLM required).
V2: LLM-assisted plan generation via llm_rewriter.generate_analysis_plan().
"""

from agents.capability_registry import find_capability, is_known_unsupported, CAPABILITY_REGISTRY


def infer_goal_type(user_goal: str, profile: dict, outcome: str) -> str:
    """
    Classify the user's goal into a high-level task type.

    Returns one of:
      "classification", "regression", "count_modeling",
      "time_series", "doe_anova", "automl", "exploratory", "unknown"
    """
    text = user_goal.lower()
    outcome_info = next(
        (c for c in profile.get("columns", []) if c["name"] == outcome),
        None,
    )
    outcome_type = outcome_info["inferred_type"] if outcome_info else "unknown"

    if any(kw in text for kw in ["churn", "predict", "classify", "binary", "probability", "logistic"]):
        return "classification"
    if any(kw in text for kw in ["forecast", "time series", "trend", "temporal", "season"]):
        return "time_series"
    if any(kw in text for kw in ["count", "frequency", "number of", "poisson"]):
        return "count_modeling"
    if any(kw in text for kw in ["anova", "factorial", "treatment", "experiment", "doe", "covariate", "ancova"]):
        return "doe_anova"
    if any(kw in text for kw in ["compare models", "automl", "best model", "leaderboard"]):
        return "automl"
    if any(kw in text for kw in ["explore", "eda", "visualise", "visualize", "describe", "profile"]):
        return "exploratory"

    # Fall back on outcome type
    if outcome_type == "binary":
        return "classification"
    if outcome_type == "continuous_numeric":
        return "regression"
    if outcome_type == "numeric_discrete_or_categorical":
        return "count_modeling"
    return "unknown"


def create_analysis_plan(profile: dict, outcome: str, user_goal: str) -> dict:
    """
    Build a structured analysis plan.

    Returns
    -------
    dict with keys:
        goal_type          : str
        cleaning_steps     : list[str]
        eda_steps          : list[str]
        candidate_models   : list[dict]
        diagnostics_needed : list[str]
        requires_dynamic   : bool   — True if no deterministic tool found
        dynamic_hint       : str    — what the dynamic agent should attempt
    """
    goal_type = infer_goal_type(user_goal, profile, outcome)
    columns   = profile.get("columns", [])

    # cleaning steps
    cleaning_steps = []
    for col in columns:
        if col.get("missing_pct", 0) > 0:
            cleaning_steps.append(
                f"Handle {col['missing_pct']:.1f}% missing values in '{col['name']}'."
            )
        if col.get("inferred_type") == "text_or_high_cardinality_categorical":
            cleaning_steps.append(
                f"Review high-cardinality column '{col['name']}' — consider encoding or dropping."
            )
    dup = profile.get("duplicates", {}).get("duplicate_pct", 0)
    if dup > 0:
        cleaning_steps.append(f"Remove {dup:.1f}% duplicate rows before modeling.")

    # EDA steps
    eda_steps = [
        "Plot outcome variable distribution.",
        "Inspect missing-value heatmap.",
        "Compute correlation matrix for numeric predictors.",
    ]
    if goal_type == "classification":
        eda_steps.append("Plot outcome class balance (bar chart).")
        eda_steps.append("Boxplot numeric predictors by outcome group.")
    elif goal_type == "regression":
        eda_steps.append("Scatterplot key numeric predictors vs outcome.")
    elif goal_type == "time_series":
        eda_steps.append("Plot time series with trend and seasonal decomposition.")

    # candidate models
    GOAL_MODEL_MAP = {
        "classification": ["logistic_regression", "automl_pycaret"],
        "regression":     ["linear_regression", "automl_pycaret"],
        "count_modeling": ["poisson_regression"],
        "doe_anova":      ["anova_ancova"],
        "time_series":    ["time_series"],
        "automl":         ["automl_pycaret"],
        "exploratory":    [],
    }
    model_keys = GOAL_MODEL_MAP.get(goal_type, [])
    candidate_models = [
        {**CAPABILITY_REGISTRY[k], "name": k}
        for k in model_keys
        if k in CAPABILITY_REGISTRY
    ]

    # diagnostics
    GOAL_DIAGNOSTICS_MAP = {
        "classification": [
            "Confusion matrix", "ROC/AUC curve",
            "Sensitivity / Specificity", "Threshold tuning (Youden's J and F1)",
            "Multicollinearity (VIF)",
        ],
        "regression": [
            "Residual normality (Shapiro-Wilk)", "Homoscedasticity (Breusch-Pagan)",
            "Influential points (Cook's distance)", "Multicollinearity (VIF)",
            "Linearity check",
        ],
        "count_modeling": [
            "Overdispersion check (dispersion statistic)",
            "Zero-inflation check",
            "Influential points",
        ],
        "doe_anova": [
            "Levene's test for equal variances",
            "Normality of residuals",
            "Pairwise comparisons (emmeans)",
        ],
        "time_series": [
            "Stationarity (ADF test)", "ACF/PACF plots", "Ljung-Box test on residuals",
        ],
    }
    diagnostics_needed = GOAL_DIAGNOSTICS_MAP.get(goal_type, ["General model diagnostics"])

    # dynamic analysis flag
    cap_name, cap_details = find_capability(user_goal)
    requires_dynamic = cap_name is None and goal_type == "unknown"
    dynamic_hint = (
        f"No deterministic tool found for: '{user_goal}'. "
        "Dynamic code generation will be attempted."
        if requires_dynamic else ""
    )

    return {
        "goal_type":          goal_type,
        "outcome":            outcome,
        "cleaning_steps":     cleaning_steps,
        "eda_steps":          eda_steps,
        "candidate_models":   candidate_models,
        "diagnostics_needed": diagnostics_needed,
        "requires_dynamic":   requires_dynamic,
        "dynamic_hint":       dynamic_hint,
    }
