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


def run_diagnostics_for_result(
    model_result: dict,
    df: pd.DataFrame | None = None,
) -> dict:
    """Return structured diagnostics keyed by model_type."""
    mtype = model_result.get("model_type", "")
    diagnostics: dict = {"notes": []}

    if mtype == "Linear Regression" and "_model" in model_result:
        model = model_result["_model"]
        try:
            jb = model.jarque_bera()
            diagnostics["residual_normality"] = {
                "jarque_bera": dict(
                    zip(["jb_stat", "jb_pvalue", "skew", "kurtosis"], [float(x) for x in jb])
                ),
            }
        except Exception as exc:  # noqa: BLE001
            diagnostics["residual_normality"] = {"error": str(exc)}
        diagnostics["heteroskedasticity"] = {
            "breusch_pagan": _breusch_pagan_safe(model),
        }
        diagnostics["influence"] = {
            "max_cooks_d": float(np.nanmax(model.get_influence().cooks_distance[0])),
        }
        if df is not None and hasattr(model, "model") and hasattr(model.model, "exog_names"):
            exog_names = [x for x in model.model.exog_names if x != "Intercept"]
            diagnostics["multicollinearity"] = _vif_table(df, exog_names)
        diagnostics["notes"].append("Review residual plots for linearity and constant variance.")

    elif mtype == "Logistic Regression":
        if "metrics" in model_result:
            diagnostics["classification"] = model_result["metrics"]
        diagnostics["notes"].append(
            "For logistic GLM, check calibration, influential points, and multicollinearity among predictors."
        )
        if "_y_true" in model_result and "_pred_prob" in model_result:
            yt = model_result["_y_true"]
            pr = model_result["_pred_prob"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = (pr >= 0.5).astype(int)
            cm = confusion_matrix(yt, pred, labels=[0, 1])
            diagnostics["confusion_at_0.5"] = cm.tolist()

    elif mtype == "Poisson Regression":
        diagnostics["poisson"] = interpret_poisson_diagnostics(model_result)

    elif mtype.startswith("RandomForest"):
        diagnostics["notes"].append(
            "Tree ensembles capture nonlinearities; validate on held-out data and check for leakage or shift."
        )
        if "cv_scores" in model_result:
            diagnostics["cross_validation"] = {
                "metric": model_result.get("cv_metric"),
                "mean": model_result.get("cv_mean"),
                "std": model_result.get("cv_std"),
            }

    elif "Time series" in mtype or "ARIMA" in mtype:
        diagnostics["notes"].append(
            "For forecasting, compare multiple horizons, consider seasonality, and stress-test residual structure."
        )

    return diagnostics
