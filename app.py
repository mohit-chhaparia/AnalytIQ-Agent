"""Streamlit UI: profiling, statistical GLMs, tabular ML, time series, and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.controller_agent import StatisticalAnalysisAgent
from agents.data_profiler import profile_dataframe
from agents.eda_agent import recommend_eda_plots
from agents.formula_builder import build_formula
from agents.intent_agent import infer_analysis_modes
from agents.model_runner import strip_internal_keys
from agents.model_selector import recommend_models
from agents.quarto_export import dumps_compact, render_quarto_report
from agents.threshold_tuning import tune_thresholds
from agents.time_series_agent import recommend_time_series_columns

st.set_page_config(page_title="Analysis agent", layout="wide")
st.title("Statistical modeling, ML, and time series agent")
st.caption(
    "Reproducible tools (statsmodels, scikit-learn) plus an orchestration layer for routing and reporting. "
    "Optional R via Rscript."
)

examples_dir = Path(__file__).resolve().parent / "examples"

tab_upload, tab_quality, tab_model, tab_report = st.tabs(
    ["Upload & goal", "Data quality & EDA", "Modeling", "Report & export"]
)

with tab_upload:
    use_example = st.selectbox(
        "Load data",
        [
            "Upload CSV",
            "Example: linear regression",
            "Example: logistic (binary)",
            "Example: counts",
            "Example: time series",
        ],
    )
    df: pd.DataFrame | None = None
    if use_example == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
    else:
        path = {
            "Example: linear regression": examples_dir / "linear_regression_sample.csv",
            "Example: logistic (binary)": examples_dir / "telco_churn_sample.csv",
            "Example: counts": examples_dir / "count_data_sample.csv",
            "Example: time series": examples_dir / "time_series_sample.csv",
        }[use_example]
        if path.exists():
            df = pd.read_csv(path)
        else:
            st.error("Example file missing from examples/.")

    if df is not None:
        st.session_state["df"] = df
        st.subheader("Preview")
        st.dataframe(df.head(20))
        outcome = st.selectbox("Outcome / target column", df.columns.tolist(), key="outcome")
        st.session_state["outcome"] = outcome
        goal = st.text_area(
            "Analysis goal (used for routing hints)",
            placeholder="Example: Forecast monthly demand with ARIMA, or predict churn with cross-validated ML.",
            key="goal",
        )
        st.session_state["goal"] = goal

