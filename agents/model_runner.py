"""
Unified model runner. Routes to the correct Python, R, or SAS implementation based on the model name and engine preference.

For models supported by the capability registry, this is a single call:
    result = run_model("logistic_regression", df, formula, outcome, engine="python")
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix,
)


# unified entry point

def run_model(model_name: str, df: pd.DataFrame, formula: str, outcome: str, engine: str = "python") -> dict:
    """Dispatch to the correct model function."""
    name = model_name.lower()
    if engine == "r":
        return _run_r_model(model_name, df, formula, outcome)
    if engine == "sas":
        return _run_sas_model(model_name, df, formula, outcome)

    # Python deterministic
    if "logistic" in name:
        return run_logistic_regression(df, formula, outcome)
    if "linear" in name:
        return run_linear_regression(df, formula)
    if "poisson" in name:
        return run_poisson_regression(df, formula)
    if "time_series" in name or "time series" in name:
        return run_time_series(df, outcome)
    if "automl" in name or "pycaret" in name:
        return run_automl(df, outcome)
    raise ValueError(f"Unknown model: {model_name}")


# Python models

def run_logistic_regression(df: pd.DataFrame, formula: str, outcome: str) -> dict:
    # Encode binary string outcomes
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[outcome]):
        vals = sorted(df[outcome].dropna().unique())
        df[outcome] = df[outcome].map({vals[0]: 0, vals[1]: 1}).astype("int64")

    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
    pred_prob = model.predict(df)
    pred_class = (pred_prob >= 0.5).astype(int)
    y_true = df[outcome]

    metrics = {
        "accuracy":            round(float(accuracy_score(y_true, pred_class)), 4),
        "sensitivity_recall":  round(float(recall_score(y_true, pred_class, zero_division=0)), 4),
        "precision":           round(float(precision_score(y_true, pred_class, zero_division=0)), 4),
        "auc":                 round(float(roc_auc_score(y_true, pred_prob)), 4),
    }
    return {
        "model_type": "Logistic Regression",
        "formula":    formula,
        "summary":    model.summary().as_text(),
        "aic":        model.aic,
        "bic":        model.bic_llf,
        "metrics":    metrics,
        "predicted_probabilities": pred_prob.tolist(),
    }


def run_linear_regression(df: pd.DataFrame, formula: str) -> dict:
    model = smf.ols(formula=formula, data=df).fit()
    return {
        "model_type":    "Linear Regression",
        "formula":       formula,
        "summary":       model.summary().as_text(),
        "aic":           model.aic,
        "bic":           model.bic,
        "r_squared":     model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "residuals":     model.resid.tolist(),
        "fitted_values": model.fittedvalues.tolist(),
    }


def run_poisson_regression(df: pd.DataFrame, formula: str) -> dict:
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
    dispersion = model.deviance / model.df_resid if model.df_resid > 0 else None
    return {
        "model_type":         "Poisson Regression",
        "formula":            formula,
        "summary":            model.summary().as_text(),
        "aic":                model.aic,
        "bic":                model.bic_llf,
        "dispersion":         round(float(dispersion), 4) if dispersion else None,
        "overdispersion_flag":bool(dispersion and dispersion > 1.5),
    }


def run_time_series(df: pd.DataFrame, outcome: str) -> dict:
    """Basic time-series decomposition + ARIMA via statsmodels."""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.arima.model import ARIMA

        series = df[outcome].dropna()
        # ADF stationarity test
        adf = adfuller(series)
        adf_result = {"stat": round(float(adf[0]), 4), "p_value": round(float(adf[1]), 4)}

        # ARIMA(1,1,1) baseline
        arima = ARIMA(series, order=(1, 1, 1)).fit()
        return {
            "model_type":  "Time Series (ARIMA 1,1,1)",
            "formula":     f"ARIMA(1,1,1) on {outcome}",
            "summary":     arima.summary().as_text(),
            "aic":         arima.aic,
            "bic":         arima.bic,
            "adf_test":    adf_result,
            "stationarity_note": (
                "Series appears stationary (p < 0.05)." if adf_result["p_value"] < 0.05
                else "Series may be non-stationary (p ≥ 0.05). Consider differencing."
            ),
        }
    except Exception as e:
        return {"model_type": "Time Series", "error": str(e)}


def run_automl(df: pd.DataFrame, outcome: str) -> dict:
    """PyCaret AutoML comparison."""
    try:
        from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull
        from pycaret.regression import setup as reg_setup, compare_models as reg_compare

        y = df[outcome]
        is_binary = y.nunique() == 2

        if is_binary:
            clf_setup(data=df, target=outcome, session_id=42, verbose=False, html=False)
            best = clf_compare(n_select=3, verbose=False)
        else:
            reg_setup(data=df, target=outcome, session_id=42, verbose=False, html=False)
            best = reg_compare(n_select=3, verbose=False)

        leaderboard = pull()
        top = leaderboard.head(1).to_dict(orient="records")[0] if not leaderboard.empty else {}
        return {
            "model_type":  "AutoML (PyCaret)",
            "formula":     f"AutoML on {outcome}",
            "summary":     leaderboard.to_string(),
            "metrics":     {k: round(float(v), 4) for k, v in top.items() if isinstance(v, (int, float))},
        }
    except ImportError:
        return {"model_type": "AutoML", "error": "pycaret not installed. Run: pip install pycaret[full]"}
    except Exception as e:
        return {"model_type": "AutoML", "error": str(e)}


# R engine bridge

def _run_r_model(model_name: str, df: pd.DataFrame, formula: str, outcome: str) -> dict:
    script_map = {
        "linear_regression":  "r_engine/run_linear_model.R",
        "logistic_regression":"r_engine/run_logistic_model.R",
        "poisson_regression": "r_engine/run_poisson_model.R",
        "anova_ancova":       "r_engine/run_anova_ancova.R",
    }
    script = script_map.get(model_name.lower())
    if not script or not Path(script).exists():
        return {"model_type": model_name, "error": f"R script not found: {script}"}

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        input_csv = f.name

    output_txt = input_csv.replace(".csv", "_output.txt")

    try:
        subprocess.run(
            ["Rscript", script, input_csv, formula, output_txt],
            check=True, capture_output=True, timeout=120,
        )
        output = Path(output_txt).read_text() if Path(output_txt).exists() else "No output"
    except subprocess.CalledProcessError as e:
        output = e.stderr.decode()
    except FileNotFoundError:
        output = "Rscript not found. Please install R."
    finally:
        os.unlink(input_csv)
        if Path(output_txt).exists():
            os.unlink(output_txt)

    return {
        "model_type": f"{model_name} (R)",
        "formula":    formula,
        "summary":    output[:5000],
        "engine":     "r",
    }


# SAS engine bridge

def _run_sas_model(model_name: str, df: pd.DataFrame, formula: str, outcome: str) -> dict:
    try:
        import saspy
        sas = saspy.SASsession()
        sas.df2sd(df, table="analysis_data")

        # Build a PROC LOGISTIC / PROC GLM script
        sas_script = _build_sas_script(model_name, formula, outcome)
        result = sas.submit(sas_script)
        return {
            "model_type": f"{model_name} (SAS)",
            "formula":    formula,
            "summary":    result.get("LST", "")[:5000],
            "log":        result.get("LOG", "")[:2000],
            "engine":     "sas",
        }
    except ImportError:
        return {"model_type": model_name, "error": "saspy not installed."}
    except Exception as e:
        return {"model_type": model_name, "error": str(e)}


def _build_sas_script(model_name: str, formula: str, outcome: str) -> str:
    # Parse a simple "Y ~ X1 + X2" formula for SAS
    parts = formula.split("~")
    response = parts[0].strip()
    predictors = parts[1].strip() if len(parts) > 1 else ""

    if "logistic" in model_name.lower():
        return f"""
proc logistic data=work.analysis_data;
    model {response}(event='1') = {predictors.replace('C(', '').replace(')', '')};
    ods output ParameterEstimates=pe;
run;
"""
    return f"""
proc glm data=work.analysis_data;
    model {response} = {predictors.replace('C(', '').replace(')', '')};
run;
"""

# ANOVA table for OLS models

def run_ols_anova_table(df: pd.DataFrame, formula: str) -> dict:
    """
    Fit an OLS model and return a Type II ANOVA table via statsmodels.
    Used for DOE-style factorial and ANCOVA analyses in Python.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.anova import anova_lm

        model = smf.ols(formula=formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        return {
            "model_type":    "OLS ANOVA (Type II)",
            "formula":       formula,
            "summary":       model.summary().as_text(),
            "anova_table":   anova_table.to_dict(),
            "aic":           model.aic,
            "bic":           model.bic,
            "r_squared":     model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "residuals":     model.resid.tolist(),
            "fitted_values": model.fittedvalues.tolist(),
        }
    except Exception as e:
        return {"model_type": "OLS ANOVA", "error": str(e)}


# Utility

def strip_internal_keys(result: dict) -> dict:
    """
    Remove large or internal-only keys from a model result dict before
    passing it to the UI or report agent.
    Keys stripped: 'summary', 'residuals', 'fitted_values', 'predicted_probabilities'
    """
    _INTERNAL = {"summary", "residuals", "fitted_values", "predicted_probabilities"}
    return {k: v for k, v in result.items() if k not in _INTERNAL}
