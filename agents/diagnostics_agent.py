"""Model diagnostics: OLS, logistic GLM, Poisson GLM."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _breusch_pagan_safe(model) -> dict:
    try:
        bp = het_breuschpagan(model.resid, model.model.exog)
        labels = ("lm_stat", "lm_pvalue", "f_stat", "f_pvalue")
        return {k: float(v) for k, v in zip(labels, bp)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
