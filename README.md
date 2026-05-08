# AnalytIQ Agent: AI-Powered Profiling, Modeling, and Recommendation System with Deterministic Engines and Local LLM Support

This repository is an **AI-assisted analysis assistant** aimed at **statistical modeling**, **tabular machine learning**, **time series**, and reproducible reporting.

- **Deterministic engines**: `statsmodels` (GLM/OLS, ARIMA), `scikit-learn` (pipelines, cross-validation, random forests).
- **Orchestration / “AI” layer**: intent routing (`intent_agent`), planning (`planning_agent`), controller memory (`StatisticalAnalysisAgent`), capability registry, optional local LLM hooks for **wording only** (never for inventing numbers).
- **Interfaces**: Streamlit app (`app.py`), Quarto HTML export when Quarto is installed.
- **Optional R** scripts under `r_engine/` for users who want `lm`/`glm` workflows via `Rscript`.

Dynamic routing uses **Python** or **R** only.

## Capabilities

| Area | What is implemented |
|------|---------------------|
| Statistical inference | OLS, GLM binomial (logistic), GLM Poisson, Type-II ANOVA table, residual/VIF diagnostics, logistic threshold sweep |
| Tabular ML | Preprocessing + RandomForest with stratified CV (AUC) or R², hold-out metrics, feature importances |
| Time series | ADF + ACF/PACF summary; ARIMA forecast with confidence intervals |
| AI-style routing | Keyword + schema heuristics to suggest TS vs ML vs inference paths |

## Layout

```
app.py
agents/
  data_profiler.py, eda_agent.py, model_selector.py, model_runner.py
  ml_agent.py, time_series_agent.py, intent_agent.py
  diagnostics_agent.py, threshold_tuning.py, report_agent.py
  formula_builder.py, planning_agent.py, controller_agent.py
  capability_registry.py, engine_router.py, dynamic_analysis_agent.py
  code_validator.py, llm_report_writer.py, quarto_export.py, r_runner.py
examples/          # includes time_series_sample.csv
tests/
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
python3 -m pytest tests/ -q
```
