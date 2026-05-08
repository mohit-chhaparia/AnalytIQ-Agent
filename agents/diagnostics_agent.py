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


def _vif_table(df: pd.DataFrame, exog_cols: list[str]) -> dict:
    """VIF for numeric exog columns only."""
    numeric = [c for c in exog_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 2:
        return {"note": "VIF skipped: need at least two numeric predictors."}
    X = df[numeric].dropna()
    if len(X) < len(numeric) + 2:
        return {"note": "VIF skipped: insufficient complete rows."}
    X_const = sm.add_constant(X.astype(float), has_constant="add")
    vifs = []
    for i in range(1, X_const.shape[1]):
        try:
            vifs.append(
                {
                    "variable": X_const.columns[i],
                    "vif": float(variance_inflation_factor(X_const.values, i)),
                }
            )
        except Exception:
            continue
    return {"vif": vifs}


def interpret_poisson_diagnostics(model_results: dict) -> list[str]:
    notes: list[str] = []
    dispersion = model_results.get("dispersion")
    if dispersion is not None and not (isinstance(dispersion, float) and np.isnan(dispersion)):
        if dispersion > 1.5:
            notes.append(
                "The dispersion statistic is greater than 1.5, suggesting possible overdispersion. "
                "A quasi-Poisson or negative binomial model should be considered."
            )
        else:
            notes.append(
                "The dispersion statistic does not strongly suggest overdispersion for this fit."
            )
    if model_results.get("overdispersion_flag"):
        notes.append("Overdispersion flag is set; review residual deviance and Pearson chi-square.")
    return notes
