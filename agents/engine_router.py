"""
Decides which execution engine (Python / R) to use for a given analysis, based on analysis type and optional user preference.
"""

PYTHON_TASKS = [
    "machine learning", "classification", "regression",
    "clustering", "automl", "feature importance",
    "cross validation", "neural network", "deep learning",
    "time series", "arima", "forecasting",
    "logistic", "poisson", "linear regression",
    "gradient boosting", "random forest",
    "bagging", "boosting",
]

R_TASKS = [
    "anova", "ancova", "mixed effects", "lme4", "nlme",
    "survival analysis", "cox", "kaplan",
    "emmeans", "doe", "factorial design",
    "nonparametric", "wilcoxon", "kruskal",
    "repeated measures", "longitudinal",
    "mixed model", "hazard", "spatial statistics",
]

SAS_TASKS = [
    "proc logistic", "proc mixed", "proc glm", "proc freq",
    "proc lifetest", "proc phreg",
    "clinical", "biostatistics", "regulatory",
    "repeated measures anova", "mixed model anova",
]


def choose_engine(analysis_type: str, user_preference: str = None) -> str:
    """
    Return the best engine for a given analysis type.

    Parameters
    ----------
    analysis_type    : Free-text description of the analysis
    user_preference  : Optional explicit choice ("python", "r", "sas")

    Returns
    -------
    "python" | "r" | "sas"
    """
    if user_preference and user_preference.lower() in ("python", "r", "sas"):
        return user_preference.lower()

    text = analysis_type.lower()

    # SAS first: explicitly SAS-branded or regulatory
    if any(kw in text for kw in SAS_TASKS):
        return "sas"

    # R second: statistical modeling strengths
    if any(kw in text for kw in R_TASKS):
        return "r"

    # Python default
    if any(kw in text for kw in PYTHON_TASKS):
        return "python"

    return "python"  # safe default


def get_engine_description(engine: str) -> str:
    descriptions = {
        "python": "Python (statsmodels / scikit-learn / pandas)",
        "r":      "R (tidyverse / broom / car / emmeans / survival)",
        "sas":    "SAS via SASPy (PROC LOGISTIC / PROC MIXED / PROC GLM)",
    }
    return descriptions.get(engine, engine)
