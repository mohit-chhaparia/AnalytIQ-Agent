"""Orchestrates profiling, planning, GLM/OLS, ML, time-series, diagnostics, and reporting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from agents.data_profiler import profile_dataframe
from agents.diagnostics_agent import run_diagnostics_for_result
from agents.ml_agent import run_tabular_ml
from agents.model_comparison_agent import compare_models
from agents.model_runner import (
    run_linear_regression,
    run_logistic_regression,
    run_poisson_regression,
    run_ols_anova_table,
    strip_internal_keys,
)
from agents.planning_agent import build_analysis_plan
from agents.report_agent import generate_plain_english_summary
from agents.time_series_agent import run_arima_forecast, run_time_series_summary
