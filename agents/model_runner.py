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


def run_logistic_regression(df: pd.DataFrame, formula: str, outcome: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
    pred_prob = model.predict(df)
    y_raw = df[outcome]
    y_true = pd.to_numeric(y_raw, errors="coerce")
    if y_true.notna().all() and set(y_true.dropna().unique()).issubset({0, 1}):
        pass
    else:
        uniq = y_raw.dropna().astype(str).unique().tolist()
        if len(uniq) == 2:
            low, high = sorted(uniq, key=str)
            y_true = (y_raw.astype(str) == high).astype(int)
        else:
            y_true = y_true.fillna(0).astype(int)

    pred_class = (pred_prob >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, pred_prob))
    except ValueError:
        auc = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred_class)),
        "sensitivity_recall": float(recall_score(y_true, pred_class, zero_division=0)),
        "precision": float(precision_score(y_true, pred_class, zero_division=0)),
        "auc": auc,
    }
    return {
        "model_type": "Logistic Regression",
        "formula": formula,
        "summary": model.summary().as_text(),
        "aic": float(model.aic),
        "bic": float(model.bic_llf),
        "metrics": metrics,
        "predicted_probabilities": pred_prob.tolist(),
        "_model": model,
        "_y_true": y_true,
        "_pred_prob": pred_prob,
    }


