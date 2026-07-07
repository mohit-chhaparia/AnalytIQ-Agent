"""Public re-exports for `from agents import ...`."""

from agents.controller_agent import StatisticalAnalysisAgent
from agents.data_profiler import profile_dataframe
from agents.diagnostics_agent import (
    run_diagnostics_for_result,
    run_diagnostics,
    interpret_diagnostics,
)
from agents.eda_agent import recommend_eda_plots
from agents.intent_agent import infer_analysis_modes
from agents.ml_agent import run_tabular_ml
from agents.model_runner import (
    run_linear_regression,
    run_logistic_regression,
    run_poisson_regression,
    run_ols_anova_table,
    strip_internal_keys,
)
from agents.model_selector import recommend_models
from agents.report_agent import generate_plain_english_summary
from agents.time_series_agent import (
    recommend_time_series_columns,
    run_arima_forecast,
    run_time_series_summary,
)

__all__ = [
    "profile_dataframe",
    "recommend_models",
    "recommend_eda_plots",
    "infer_analysis_modes",
    "run_tabular_ml",
    "run_time_series_summary",
    "run_arima_forecast",
    "recommend_time_series_columns",
    "run_linear_regression",
    "run_logistic_regression",
    "run_poisson_regression",
    "run_ols_anova_table",
    "strip_internal_keys",
    "run_diagnostics_for_result",
    "generate_plain_english_summary",
    "StatisticalAnalysisAgent",
]
