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

with tab_quality:
    if "df" not in st.session_state:
        st.info("Upload or select example data in the first tab.")
    else:
        df = st.session_state["df"]
        outcome = st.session_state.get("outcome", df.columns[0])
        if st.button("Run profiler & EDA recommendations", key="prof_btn"):
            profile = profile_dataframe(df)
            st.session_state["profile"] = profile
            st.session_state["eda"] = recommend_eda_plots(profile, outcome)
            st.session_state["model_recs"] = recommend_models(
                profile, outcome, st.session_state.get("goal", "")
            )
            st.session_state["intent"] = infer_analysis_modes(
                st.session_state.get("goal", ""), profile, outcome
            )
        if "profile" in st.session_state:
            st.subheader("Suggested workflow (intent routing)")
            st.json(st.session_state.get("intent", {}))
            st.subheader("Data quality profile")
            st.json(st.session_state["profile"])
            st.subheader("Starter visualizations")
            for p in st.session_state.get("eda", []):
                st.write(p)
            st.subheader("Model & method recommendations")
            st.json(st.session_state.get("model_recs", {}))

with tab_model:
    if "df" not in st.session_state or "profile" not in st.session_state:
        st.info("Complete the Upload and Data quality tabs first.")
    else:
        df = st.session_state["df"]
        outcome = st.session_state["outcome"]
        profile = st.session_state["profile"]

        workflow = st.radio(
            "Modeling workflow",
            ["Statistical (GLM / OLS)", "Tabular ML (random forest)", "Time series"],
            horizontal=True,
        )

        if workflow == "Statistical (GLM / OLS)":
            predictors = st.multiselect(
                "Predictors",
                [c for c in df.columns if c != outcome],
                default=[c for c in df.columns if c != outcome][: min(5, len(df.columns) - 1)],
            )
            formula = build_formula(outcome, predictors, profile) if predictors else ""
            st.code(formula or "(select predictors)", language="text")

            model_kind = st.selectbox(
                "Model family",
                ["linear", "logistic", "poisson", "ols_anova"],
                format_func=lambda x: {
                    "linear": "OLS / linear regression",
                    "logistic": "Logistic (GLM binomial)",
                    "poisson": "Poisson GLM",
                    "ols_anova": "OLS with ANOVA table (Type II)",
                }[x],
            )

            if st.button("Fit & diagnose", key="fit_glm") and formula:
                agent = StatisticalAnalysisAgent(
                    df, st.session_state.get("goal", ""), outcome
                )
                agent.memory["profile"] = profile
                agent.build_plan()
                try:
                    agent.fit_primary_model(formula, model_kind)
                    agent.run_diagnostics()
                    agent.memory["plain_english"] = agent.generate_report_text()
                    st.session_state["agent_memory"] = agent.memory
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        elif workflow == "Tabular ML (random forest)":
            predictors = st.multiselect(
                "Features",
                [c for c in df.columns if c != outcome],
                default=[c for c in df.columns if c != outcome][: min(8, len(df.columns) - 1)],
            )
            task = st.selectbox("Task", ["classify", "regress"])
            if st.button("Run cross-validated forest", key="fit_ml"):
                agent = StatisticalAnalysisAgent(
                    df, st.session_state.get("goal", ""), outcome
                )
                agent.memory["profile"] = profile
                agent.build_plan()
                try:
                    agent.fit_ml(predictors, task)
                    agent.run_diagnostics()
                    agent.memory["plain_english"] = agent.generate_report_text()
                    st.session_state["agent_memory"] = agent.memory
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        else:
            ts_hint = recommend_time_series_columns(profile)
            st.caption("Pick a numeric series column; optionally parse a datetime column for ordering.")
            col_list = df.columns.tolist()
            default_ix = col_list.index(outcome) if outcome in col_list else 0
            series_col = st.selectbox(
                "Series column (y)",
                col_list,
                index=default_ix,
            )
            dt_cols = [None] + ts_hint.get("datetime_columns", [])
            time_col = st.selectbox("Time / index column (optional)", dt_cols)
            mode_ts = st.radio("Time series action", ["Characterize (ADF, ACF, PACF)", "ARIMA forecast"], horizontal=True)

            if time_col:
                dfp = df.copy()
                dfp["_ts_ix"] = pd.to_datetime(dfp[time_col], errors="coerce")
                dfp = dfp.sort_values("_ts_ix")
                series = dfp[series_col]
            else:
                series = df[series_col]

            if mode_ts.startswith("Characterize"):
                if st.button("Run characterization", key="ts_char"):
                    agent = StatisticalAnalysisAgent(
                        df, st.session_state.get("goal", ""), outcome
                    )
                    agent.memory["profile"] = profile
                    agent.build_plan()
                    try:
                        agent.fit_time_series_characterization(series)
                        agent.run_diagnostics()
                        agent.memory["plain_english"] = agent.generate_report_text()
                        st.session_state["agent_memory"] = agent.memory
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
            else:
                p = st.number_input("AR p", 0, 5, 1)
                d = st.number_input("I d", 0, 2, 0)
                q = st.number_input("MA q", 0, 5, 1)
                steps = st.number_input("Forecast steps", 1, 30, 8)
                if st.button("Fit ARIMA", key="ts_arima"):
                    agent = StatisticalAnalysisAgent(
                        df, st.session_state.get("goal", ""), outcome
                    )
                    agent.memory["profile"] = profile
                    agent.build_plan()
                    try:
                        agent.fit_time_series_arima(series, (int(p), int(d), int(q)), steps=int(steps))
                        agent.run_diagnostics()
                        agent.memory["plain_english"] = agent.generate_report_text()
                        st.session_state["agent_memory"] = agent.memory
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

        if st.session_state.get("agent_memory", {}).get("model_result"):
            mr = st.session_state["agent_memory"]["model_result"]
            st.subheader("Results")
            if mr.get("summary"):
                summary_text = mr.get("summary", "")
                st.text(summary_text[:8000] + ("..." if len(summary_text) > 8000 else ""))
            if mr.get("ols_summary"):
                st.text(mr.get("ols_summary", "")[:6000])
            if mr.get("anova_table"):
                st.text(mr.get("anova_table", "")[:4000])
            if mr.get("top_feature_importances"):
                st.dataframe(pd.DataFrame(mr["top_feature_importances"]))
            if mr.get("acf") is not None:
                st.line_chart(pd.DataFrame({"acf": mr["acf"][:50]}))
            if mr.get("forecast_mean"):
                fc = pd.DataFrame(
                    {
                        "mean": mr["forecast_mean"],
                        "lower": mr.get("forecast_ci_lower", []),
                        "upper": mr.get("forecast_ci_upper", []),
                    }
                )
                st.line_chart(fc)
            st.subheader("Diagnostics")
            st.json(
                {
                    k: v
                    for k, v in st.session_state["agent_memory"]
                    .get("diagnostics", {})
                    .items()
                }
            )
            if mr.get("model_type") == "Logistic Regression" and "_pred_prob" in mr:
                st.subheader("Threshold tuning (top rows)")
                tuned = tune_thresholds(mr["_y_true"], mr["_pred_prob"])
                st.dataframe(pd.DataFrame(tuned[:15]))

