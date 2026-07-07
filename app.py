"""
Streamlit multi-page application.

Pages
-----
1 · Upload & Profile          — upload CSV, run data profiler
2 · EDA                       — plot recommendations + auto charts
3 · Model Selection & Run     — formula builder, model runner, comparison
4 · Diagnostics               — diagnostic notes + threshold tuning
5 · Report                    — plain-English summary + HTML export
6 · Full Agentic Run (V2)     — rule-based controller end-to-end
7 · Claude ReAct Agent (V2)   — Claude tool-calling loop
8 · Visual Diagnostics (V2)   — Gemini interprets your plots
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from agents.data_profiler import profile_dataframe
from agents.eda_agent import recommend_eda_plots
from agents.model_selector import recommend_models
from agents.model_runner import run_model
from agents.diagnostics_agent import run_diagnostics, interpret_diagnostics
from agents.threshold_tuning import tune_thresholds, interpret_threshold_results
from agents.report_agent import generate_plain_english_summary
from agents.model_comparison_agent import compare_models
from agents.controller_agent import StatisticalAnalysisAgent
from agents.claude_tool_agent import ClaudeToolAgent
from agents.gemini_vision_agent import interpret_all_plots, interpret_plot
from agents.report_renderer import render_report
from agents.capability_registry import list_capabilities
from agents.llm_router import availability_report, AVAILABLE, any_llm_available
from agents.llm_rewriter import review_full_report_with_gemini

# page config
st.set_page_config(page_title="AnalytIQ Agent", page_icon="📊", layout="wide")

# sidebar
st.sidebar.title("📊 AnalytIQ Agent")
st.sidebar.caption("AI-Assisted Statistical Modeling")

page = st.sidebar.radio("Navigate", [
    "1 · Upload & Profile",
    "2 · EDA",
    "3 · Model Selection & Run",
    "4 · Diagnostics",
    "5 · Report",
    "6 · Full Agentic Run",
    "7 · Claude ReAct Agent",
    "8 · Visual Diagnostics — Gemini",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Settings**")

# use_llm auto-detects if any key is configured; user can override
_llm_default = any_llm_available()
use_llm = st.sidebar.toggle(
    "Enable LLM summaries",
    value=_llm_default,
    help="Uses the best available LLM (Claude / Gemini / Groq) for plain-English output.",
)
use_dynamic = st.sidebar.toggle(
    "Enable dynamic analysis",
    value=True,
    help="Generates code for analyses not in the built-in tool registry.",
)
engine_pref = st.sidebar.selectbox("Preferred engine", ["auto", "python", "r"])

st.sidebar.markdown("---")
st.sidebar.markdown("**LLM backends**")
for name, status in availability_report().items():
    st.sidebar.caption(f"{name}: {status}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built-in analyses**")
for cap in list_capabilities():
    st.sidebar.markdown(cap)

# session state
for _k, _v in {
    "df": None, "profile": None, "outcome": None, "goal": "",
    "formula": None, "model_results": [], "diagnostics": [], "agent_memory": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


#==============================================================================
# PAGE 1 — Upload & Profile
#==============================================================================
if page == "1 · Upload & Profile":
    st.title("📂 Upload & Data Profile")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("▶ Run Data Profiler"):
            with st.spinner("Profiling…"):
                profile = profile_dataframe(df)
                st.session_state["profile"] = profile

    profile = st.session_state.get("profile")
    if profile:
        p = profile
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",           p["shape"]["rows"])
        c2.metric("Columns",        p["shape"]["columns"])
        c3.metric("Duplicate rows", p["duplicates"]["duplicate_rows"])
        c4.metric("Missing cols",   sum(1 for c in p["columns"] if c.get("missing_pct", 0) > 0))

        st.subheader("Column Profile")
        st.dataframe(pd.DataFrame([{
            "Column":        c["name"],
            "Inferred Type": c["inferred_type"],
            "Missing %":     c.get("missing_pct", 0),
            "Unique":        c.get("unique_count", "—"),
            "Mean":          c.get("mean", "—"),
        } for c in p["columns"]]), use_container_width=True)

        missing = [(c["name"], c.get("missing_pct", 0))
                   for c in p["columns"] if c.get("missing_pct", 0) > 0]
        if missing:
            st.subheader("Missing Values")
            fig = px.bar(x=[m[0] for m in missing], y=[m[1] for m in missing],
                         labels={"x": "Column", "y": "Missing %"},
                         color=[m[1] for m in missing], color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full profile JSON"):
            st.json(p)


#==============================================================================
# PAGE 2 — EDA
#==============================================================================
elif page == "2 · EDA":
    st.title("🔍 EDA Recommendations")
    df      = st.session_state.get("df")
    profile = st.session_state.get("profile")

    if df is None or profile is None:
        st.info("Upload a dataset and run the profiler on Page 1 first.")
        st.stop()

    outcome = st.selectbox("Outcome variable", df.columns)
    st.session_state["outcome"] = outcome

    if st.button("▶ Get EDA Recommendations"):
        recs = recommend_eda_plots(profile, outcome)
        st.subheader("Recommended Plots")
        for r in recs:
            st.markdown(f"**{r['plot']}** — `{r.get('x','?')}` vs `{r.get('y','?')}`: {r.get('purpose','')}")

        st.subheader("Auto-generated Charts")
        for r in recs[:5]:
            x_col, y_col = r.get("x"), r.get("y")
            if not x_col or not y_col:
                continue
            if x_col not in df.columns or y_col not in df.columns:
                continue
            try:
                plot_type = r.get("plot", "").lower()
                if "bar" in plot_type or "proportion" in plot_type:
                    fig = px.histogram(df, x=x_col, color=y_col, barmode="group",
                                       title=f"{r['plot']}: {x_col} by {y_col}")
                elif "box" in plot_type:
                    fig = px.box(df, x=x_col, y=y_col,
                                 title=f"Boxplot: {y_col} by {x_col}")
                elif "scatter" in plot_type:
                    fig = px.scatter(df, x=x_col, y=y_col,
                                     title=f"Scatter: {x_col} vs {y_col}")
                else:
                    fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass


#==============================================================================
# PAGE 3 — Model Selection & Run
#==============================================================================
elif page == "3 · Model Selection & Run":
    st.title("🧮 Model Selection & Run")
    df      = st.session_state.get("df")
    profile = st.session_state.get("profile")

    if df is None or profile is None:
        st.info("Upload a dataset and run the profiler on Page 1 first.")
        st.stop()

    cols = list(df.columns)
    default_idx = cols.index(st.session_state["outcome"]) if st.session_state["outcome"] in cols else 0
    outcome = st.selectbox("Outcome variable", cols, index=default_idx)
    st.session_state["outcome"] = outcome

    goal = st.text_area("Analysis goal", value=st.session_state["goal"],
                        placeholder="E.g. Predict customer churn using contract and usage data.")
    st.session_state["goal"] = goal

    # Formula builder
    st.subheader("Formula Builder")
    predictors = st.multiselect(
        "Select predictor variables",
        [c for c in df.columns if c != outcome],
        default=[c for c in df.columns if c != outcome][:5],
    )
    if predictors:
        terms = []
        for p_col in predictors:
            info = next((c for c in profile["columns"] if c["name"] == p_col), {})
            if info.get("inferred_type") in ("categorical", "binary", "numeric_discrete_or_categorical"):
                terms.append(f"C({p_col})")
            else:
                terms.append(p_col)
        formula = f"{outcome} ~ " + " + ".join(terms)
        st.code(formula)
        st.session_state["formula"] = formula

    if goal and st.button("Get Model Recommendations"):
        recs = recommend_models(profile, outcome, goal)
        st.subheader("Recommended Models")
        for r in recs.get("recommendations", []):
            st.markdown(f"**{r['model']}** — {r.get('reason','')}")

    st.subheader("Run a Model")
    selected_model = st.selectbox("Model", [
        "logistic_regression", "linear_regression",
        "poisson_regression", "anova_ancova", "time_series", "automl_pycaret",
    ])
    run_engine = st.selectbox("Engine", ["python", "r"],
                              index=0 if engine_pref in ("auto", "python") else 1)

    if st.button("▶ Run Model"):
        formula = st.session_state.get("formula") or (
            f"{outcome} ~ " + " + ".join(c for c in df.columns if c != outcome)
        )
        with st.spinner(f"Running {selected_model}…"):
            try:
                result = run_model(selected_model, df, formula, outcome, run_engine)
                st.session_state["model_results"].append(result)
                st.success("Done!")
                c1, c2, c3 = st.columns(3)
                c1.metric("AIC", round(result["aic"], 2)   if result.get("aic")                    else "—")
                c2.metric("BIC", round(result["bic"], 2)   if result.get("bic")                    else "—")
                if result.get("r_squared"):
                    c3.metric("R²",  round(result["r_squared"], 4))
                elif result.get("metrics", {}).get("auc"):
                    c3.metric("AUC", round(result["metrics"]["auc"], 4))
                with st.expander("Model Summary"):
                    st.text(result.get("summary", "No summary available"))
            except Exception as e:
                st.error(f"Error: {e}")

    if len(st.session_state["model_results"]) > 1:
        st.subheader("Model Comparison")
        comp = compare_models(st.session_state["model_results"])
        st.markdown(f"**Best model:** {comp['best_model']}")
        st.markdown(comp["rationale"])
        st.dataframe(pd.DataFrame(comp["comparison_table"]), use_container_width=True)


#==============================================================================
# PAGE 4 — Diagnostics
#==============================================================================
elif page == "4 · Diagnostics":
    st.title("🩺 Diagnostics & Threshold Tuning")
    model_results = st.session_state.get("model_results", [])

    if not model_results:
        st.info("Run a model on Page 3 first.")
        st.stop()

    for i, result in enumerate(model_results):
        st.subheader(f"Model {i+1}: {result.get('model_type','Unknown')}")
        diag  = run_diagnostics(result)
        notes = interpret_diagnostics(diag, result)
        for note in notes:
            kw = note.lower()
            if any(w in kw for w in ["warn", "flag", "detect", "consider", "overdispers"]):
                st.warning(note)
            else:
                st.success(note)

        if "logistic" in result.get("model_type", "").lower():
            probs = result.get("predicted_probabilities")
            if probs:
                outcome = st.session_state.get("outcome")
                df      = st.session_state.get("df")
                y_col   = df[outcome]
                if y_col.dtype == object:
                    uniq   = sorted(y_col.unique())
                    y_true = y_col.map({uniq[0]: 0, uniq[1]: 1}).values
                else:
                    y_true = y_col.values

                tuning       = tune_thresholds(y_true, np.array(probs))
                thresh_notes = interpret_threshold_results(tuning)

                st.subheader("Threshold Tuning")
                for n in thresh_notes:
                    st.info(n)

                table = pd.DataFrame(tuning["threshold_table"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=table["threshold"], y=table["sensitivity"],
                                         name="Sensitivity", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=table["threshold"], y=table["specificity"],
                                         name="Specificity", line=dict(color="red")))
                fig.add_trace(go.Scatter(x=table["threshold"], y=table["f1"],
                                         name="F1", line=dict(color="green", dash="dash")))
                fig.update_layout(title="Sensitivity / Specificity / F1 vs Threshold",
                                  xaxis_title="Threshold", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

                by = tuning["best_youden"]
                bf = tuning["best_f1"]
                st.markdown(
                    f"**Best (Youden's J):** `{by['threshold']}` — "
                    f"Sensitivity={by['sensitivity']:.3f}, Specificity={by['specificity']:.3f}"
                )
                st.markdown(
                    f"**Best (F1):** `{bf['threshold']}` — F1={bf['f1']:.3f}"
                )
                with st.expander("Full threshold table"):
                    st.dataframe(table, use_container_width=True)


#==============================================================================
# PAGE 5 — Report
#==============================================================================
elif page == "5 · Report":
    st.title("📄 Report Export")
    model_results = st.session_state.get("model_results", [])
    agent_memory  = st.session_state.get("agent_memory")

    if not model_results and not agent_memory:
        st.info("Run a model (Page 3) or a full agent run (Pages 6–7) first.")
        st.stop()

    if agent_memory:
        memory = agent_memory
    else:
        best = model_results[-1] if model_results else {}
        diag_notes = []
        for r in model_results:
            diag_notes.extend(interpret_diagnostics(run_diagnostics(r), r))
        raw = generate_plain_english_summary(best, diag_notes)
        memory = {
            "profile":              st.session_state.get("profile", {}),
            "plan":                 {},
            "eda_recommendations":  [],
            "model_recommendations":{},
            "fitted_models":        model_results,
            "diagnostics":          [{"notes": diag_notes}],
            "best_model_result":    best,
            "model_comparison":     compare_models(model_results) if len(model_results) > 1 else {},
            "report": {
                "analysis_goal":     st.session_state.get("goal", ""),
                "outcome":           st.session_state.get("outcome", ""),
                "goal_type":         "classification",
                "plain_english":     raw,
                "diagnostics_notes": diag_notes,
                "revisions":         [],
                "dynamic_used":      False,
            },
        }

    st.subheader("Plain-English Summary")
    st.markdown(memory.get("report", {}).get("plain_english")
                or memory.get("final_narrative", "_No summary available._"))

    if st.button("▶ Generate & Download HTML Report"):
        with st.spinner("Rendering Quarto report…"):
            result = render_report(memory)
        if result["status"] == "success" and result.get("html_path"):
            st.download_button(
                "⬇ Download HTML Report",
                data      = open(result["html_path"], "rb").read(),
                file_name = "analytiq_report.html",
                mime      = "text/html",
            )
            st.success(result["message"])
        else:
            st.error(f"Render issue: {result.get('message')}")
            if result.get("qmd_path"):
                st.download_button(
                    "⬇ Download .qmd (install Quarto to render)",
                    data      = open(result["qmd_path"], "rb").read(),
                    file_name = "analytiq_report.qmd",
                    mime      = "text/plain",
                )

    # Gemini expert review
    if AVAILABLE.get("gemini") and memory.get("report"):
        st.markdown("---")
        if st.button("🔍 Gemini Expert Review (uses gemini-1.5-pro)"):
            with st.spinner("Gemini 1.5 Pro reviewing the full analysis…"):
                review = review_full_report_with_gemini(
                    report_text  = memory.get("report", {}).get("plain_english", ""),
                    profile_json = json.dumps(memory.get("profile", {}), default=str),
                )
            st.subheader("Gemini Expert Review")
            st.markdown(review or "_No review returned — check GEMINI_API_KEY._")


#==============================================================================
# PAGE 6 — Full Agentic Run (rule-based controller)
#==============================================================================
elif page == "6 · Full Agentic Run":
    st.title("🤖 Full Agentic Run")
    st.caption("Rule-based controller: deterministic pipeline end-to-end.")

    df = st.session_state.get("df")
    if df is None:
        st.info("Upload a dataset on Page 1 first.")
        st.stop()

    outcome = st.selectbox("Outcome variable", df.columns)
    st.session_state["outcome"] = outcome
    goal = st.text_area("Analysis goal",
                        placeholder="E.g. Predict customer churn and identify key drivers.")
    st.session_state["goal"] = goal
    engine_choice = None if engine_pref == "auto" else engine_pref

    if st.button("🚀 Run Agent"):
        if not goal:
            st.warning("Please describe your analysis goal.")
            st.stop()
        with st.spinner("Running…"):
            agent  = StatisticalAnalysisAgent(
                df=df, user_goal=goal, outcome=outcome,
                engine_pref=engine_choice, use_llm=use_llm, use_dynamic=use_dynamic,
            )
            memory = agent.run()
            st.session_state["agent_memory"] = memory

        st.success("Done!")
        with st.expander("Agent logs"):
            for log in memory.get("logs", []):
                st.text(log)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan")
            plan = memory.get("plan", {})
            st.markdown(f"**Goal type:** {plan.get('goal_type','—')}")
            for s in plan.get("cleaning_steps", []):
                st.warning(s)
        with col2:
            st.subheader("Best Model")
            best = memory.get("best_model_result", {})
            if best:
                st.markdown(f"**{best.get('model_type','—')}**")
                m = best.get("metrics", {})
                if m.get("auc"):       st.metric("AUC", round(m["auc"], 4))
                if best.get("r_squared"): st.metric("R²",  round(best["r_squared"], 4))
                if best.get("aic"):    st.metric("AIC", round(best["aic"], 2))

        for rev in memory.get("revisions", []):
            st.warning(rev)

        st.markdown("Go to **Page 5 · Report** to download the HTML report.")


#==============================================================================
# PAGE 7 — Claude ReAct Agent
#==============================================================================
elif page == "7 · Claude ReAct Agent":
    st.title("🧠 Claude ReAct Agent")
    st.markdown(
        "Claude uses **native tool calling** to orchestrate the analysis — "
        "deciding which tools to call, reviewing intermediate results, and "
        "revising its plan when diagnostics reveal problems."
    )

    if not AVAILABLE.get("claude"):
        st.error("ANTHROPIC_API_KEY not set in .env")
        st.code("ANTHROPIC_API_KEY=sk-ant-...")
        st.stop()

    df = st.session_state.get("df")
    if df is None:
        st.info("Upload a dataset on Page 1 first.")
        st.stop()

    outcome = st.selectbox("Outcome variable", df.columns)
    goal    = st.text_area("Analysis goal",
                           placeholder="E.g. Identify churn drivers and fit the best model.")

    # Model selector — reads CLAUDE_MODEL_AGENT env var by default
    claude_model = st.selectbox(
        "Claude model",
        ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        help="Opus is most capable for complex multi-step reasoning. Sonnet is faster.",
    )
    os.environ["CLAUDE_MODEL_AGENT"] = claude_model   # ← fixed: was CLAUDE_MODEL

    max_rounds = st.slider("Max tool-calling rounds", 4, 20, 12)

    if st.button("🚀 Run Claude ReAct Agent"):
        if not goal:
            st.warning("Please describe your analysis goal.")
            st.stop()
        with st.spinner(f"Claude is reasoning and calling tools… (up to {max_rounds} rounds)"):
            agent  = ClaudeToolAgent(df=df, user_goal=goal, outcome=outcome, max_rounds=max_rounds)
            result = agent.run()
            st.session_state["agent_memory"] = result

        if result.get("status") == "success":
            st.success(f"Done — {result['rounds']} tool calls in {result['rounds']} rounds.")

            st.subheader("Tool Call Log")
            for entry in result.get("tool_log", []):
                with st.expander(f"Round {entry['round']}: `{entry['tool']}`"):
                    st.json(entry.get("result", {}))

            st.subheader("Claude's Analysis Report")
            st.markdown(result.get("final_narrative", "_No narrative generated._"))

            if result.get("model_results"):
                st.subheader("Models Fitted")
                for r in result["model_results"]:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Model", r.get("model_type", "—"))
                    c2.metric("AIC",   round(r["aic"], 2)                      if r.get("aic")                        else "—")
                    c3.metric("AUC",   round(r["metrics"]["auc"], 4)           if r.get("metrics", {}).get("auc")     else "—")
                    c4.metric("R²",    round(r["r_squared"], 4)                if r.get("r_squared")                  else "—")

            # Gemini expert review of Claude's output
            if AVAILABLE.get("gemini"):
                st.markdown("---")
                if st.button("🔍 Get Gemini Expert Review of Claude's Analysis"):
                    with st.spinner("Gemini 1.5 Pro reviewing…"):
                        review = review_full_report_with_gemini(
                            report_text  = result.get("final_narrative", ""),
                            profile_json = json.dumps(result.get("profile", {}), default=str),
                        )
                    st.subheader("Gemini Expert Review")
                    st.markdown(review or "_No review returned._")

            st.markdown("---")
            st.caption("Go to **Page 5 · Report** to download the full HTML report.")
        else:
            st.error(f"Agent failed: {result.get('reason', result.get('status'))}")


#==============================================================================
# PAGE 8 — Visual Diagnostics (Gemini)
#==============================================================================
elif page == "8 · Visual Diagnostics — Gemini":
    st.title("👁 Visual Diagnostics with Gemini")
    st.markdown(
        "Gemini looks at your diagnostic **images** and flags issues "
        "— heteroscedasticity in residual plots, poor AUC curves, class imbalance — "
        "that rule-based checks can't catch."
    )

    if not AVAILABLE.get("gemini"):
        st.error("GEMINI_API_KEY not set in .env — get a free key at aistudio.google.com/apikey")
        st.stop()

    model_results = st.session_state.get("model_results", [])

    if model_results:
        if st.button("▶ Generate Plots & Interpret with Gemini"):
            from agents.model_runner import save_diagnostic_plots
            last   = model_results[-1]
            with st.spinner("Saving plots and sending to Gemini…"):
                saved  = save_diagnostic_plots(last, "sandbox/output/plots")
                interps = interpret_all_plots(
                    "sandbox/output/plots",
                    model_type = last.get("model_type", ""),
                    context    = f"Outcome: {st.session_state.get('outcome')}. "
                                 f"Goal: {st.session_state.get('goal', '')}.",
                )
            if interps:
                for interp in interps:
                    st.subheader(f"📊 {interp.get('plot_file','Plot')}")
                    if interp["status"] == "success":
                        st.markdown(interp.get("interpretation", ""))
                        if interp.get("flags"):
                            st.warning("**Issues detected:** " + "  |  ".join(interp["flags"]))
                        for rec in interp.get("recommendations", []):
                            st.info(f"💡 {rec}")
                    else:
                        st.error(f"Gemini error: {interp.get('reason')}")
            else:
                st.info("No plots generated. Run a linear regression on Page 3 first (it saves a residual plot).")

    st.markdown("---")
    st.subheader("Upload your own plot")
    uploaded_plot = st.file_uploader("Upload a diagnostic plot (PNG/JPEG)", type=["png","jpg","jpeg"])
    plot_type     = st.selectbox("Plot type", [
        "residual_plot", "qq_plot", "roc_curve", "histogram",
        "correlation_heatmap", "calibration_curve", "class_balance",
    ])
    model_ctx = st.text_input("Model type (optional)", placeholder="e.g. Logistic Regression")

    if uploaded_plot and st.button("▶ Interpret with Gemini"):
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(uploaded_plot.read())
            tmp_path = f.name
        with st.spinner("Gemini is looking at your plot…"):
            result = interpret_plot(tmp_path, plot_type, model_ctx)
        pathlib.Path(tmp_path).unlink(missing_ok=True)

        if result["status"] == "success":
            st.subheader("Gemini's Interpretation")
            st.markdown(result.get("interpretation", ""))
            if result.get("flags"):
                st.warning("**Issues flagged:** " + "  |  ".join(result["flags"]))
            for rec in result.get("recommendations", []):
                st.info(f"💡 {rec}")
        else:
            st.error(f"Error: {result.get('reason')}")
