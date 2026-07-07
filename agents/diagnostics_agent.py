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


def run_diagnostics_for_result(result: dict, df=None) -> dict:
    """
    Run diagnostics on a fitted model result dict.
    Returns a flat dict of diagnostic stats (keys used by tests and the UI).
    """
    import statsmodels.stats.diagnostic as smd

    model_type = result.get("model_type", "").lower()
    diag = {}

    # Linear / OLS diagnostics
    if "linear" in model_type or "anova" in model_type:
        diag["r_squared"]     = result.get("r_squared")
        diag["adj_r_squared"] = result.get("adj_r_squared")
        diag["aic"]           = result.get("aic")

        residuals    = result.get("residuals", [])
        fitted_vals  = result.get("fitted_values", [])

        # Heteroskedasticity (Breusch-Pagan)
        if df is not None and residuals and fitted_vals:
            try:
                import numpy as np
                resid = np.array(residuals)
                fvals = np.array(fitted_vals)
                bp_stat, bp_p, _, _ = smd.het_breuschpagan(resid, np.column_stack([fvals]))
                diag["heteroskedasticity"] = {
                    "bp_stat": round(float(bp_stat), 4),
                    "p_value": round(float(bp_p), 4),
                    "flag":    bp_p < 0.05,
                }
            except Exception:
                diag["heteroskedasticity"] = {"flag": False, "note": "test skipped"}
        else:
            diag["heteroskedasticity"] = {"flag": False, "note": "no df provided"}

    # Logistic diagnostics
    elif "logistic" in model_type:
        diag["auc"]     = result.get("metrics", {}).get("auc")
        diag["metrics"] = result.get("metrics", {})

    # Poisson diagnostics
    elif "poisson" in model_type:
        diag["dispersion"]          = result.get("dispersion")
        diag["overdispersion_flag"] = result.get("overdispersion_flag")

    diag["notes"] = _build_notes(diag, model_type)
    return diag


def _build_notes(diag: dict, model_type: str) -> list:
    notes = []
    if "poisson" in model_type and diag.get("overdispersion_flag"):
        notes.append("Overdispersion detected. Consider quasi-Poisson or Negative Binomial.")
    if diag.get("heteroskedasticity", {}).get("flag"):
        notes.append("Heteroskedasticity detected (Breusch-Pagan p < 0.05). Consider robust SEs.")
    if diag.get("auc") and diag["auc"] < 0.6:
        notes.append("AUC below 0.6 — model discrimination is poor.")
    if not notes:
        notes.append("No major diagnostic flags detected.")
    return notes


def run_diagnostics(result: dict) -> dict:
    """Alias: compute diagnostic stats from a model result dict."""
    # Extract the df-independent stats from the result dict
    model_type = result.get("model_type", "").lower()
    diag = {}
    if "linear" in model_type:
        diag["r_squared"]          = result.get("r_squared")
        diag["adj_r_squared"]      = result.get("adj_r_squared")
        diag["aic"]                = result.get("aic")
        diag["heteroskedasticity"] = "check residuals"   # placeholder
    elif "logistic" in model_type:
        diag["auc"]     = result.get("metrics", {}).get("auc")
        diag["metrics"] = result.get("metrics", {})
    elif "poisson" in model_type:
        diag["dispersion"]          = result.get("dispersion")
        diag["overdispersion_flag"] = result.get("overdispersion_flag")
    return diag

def interpret_diagnostics(diag: dict, result: dict) -> list:
    """Alias: turn diagnostic stats into plain-English notes."""
    notes = []
    model_type = result.get("model_type", "").lower()
    if "poisson" in model_type:
        notes.extend(interpret_poisson_diagnostics(diag))
    if diag.get("overdispersion_flag"):
        notes.append("Overdispersion detected. Consider quasi-Poisson or Negative Binomial.")
    if diag.get("auc") and diag["auc"] < 0.6:
        notes.append("AUC below 0.6 — model discrimination is poor.")
    if not notes:
        notes.append("No major diagnostic flags detected.")
    return notes

