"""Fit linear, logistic, and Poisson GLMs via statsmodels; ANOVA table for OLS."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def run_linear_regression(df: pd.DataFrame, formula: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula=formula, data=df).fit()
    return {
        "model_type": "Linear Regression",
        "formula": formula,
        "summary": model.summary().as_text(),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "residual_std": float(np.sqrt(model.mse_resid)) if hasattr(model, "mse_resid") else None,
        "_model": model,
    }


