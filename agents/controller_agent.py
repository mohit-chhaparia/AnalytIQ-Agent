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


class StatisticalAnalysisAgent:
    """Controller: statistical models + tabular ML + time series + memory for AI-style orchestration."""

    def __init__(self, df: pd.DataFrame, user_goal: str, outcome: str):
        self.df = df
        self.user_goal = user_goal
        self.outcome = outcome
        self.memory: dict[str, Any] = {}

    def profile_data(self) -> None:
        self.memory["profile"] = profile_dataframe(self.df)

    def build_plan(self) -> None:
        self.memory["plan"] = build_analysis_plan(
            self.memory["profile"], self.outcome, self.user_goal
        )

    def fit_primary_model(self, formula: str, model_kind: str) -> None:
        """model_kind: linear | logistic | poisson | ols_anova"""
        df = self.df
        self.memory["workflow"] = "glm_ols"
        if model_kind == "linear":
            self.memory["model_result"] = run_linear_regression(df, formula)
        elif model_kind == "logistic":
            self.memory["model_result"] = run_logistic_regression(df, formula, self.outcome)
        elif model_kind == "poisson":
            self.memory["model_result"] = run_poisson_regression(df, formula)
        elif model_kind == "ols_anova":
            self.memory["model_result"] = run_ols_anova_table(df, formula)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

    def fit_ml(self, predictors: list[str], task: str) -> None:
        """task: classify | regress"""
        self.memory["workflow"] = "tabular_ml"
        self.memory["model_result"] = run_tabular_ml(self.df, self.outcome, predictors, task)

    def fit_time_series_characterization(self, series: pd.Series) -> None:
        self.memory["workflow"] = "time_series"
        self.memory["model_result"] = run_time_series_summary(series)

    def fit_time_series_arima(
        self,
        series: pd.Series,
        order: tuple[int, int, int],
        steps: int = 8,
    ) -> None:
        self.memory["workflow"] = "time_series"
        self.memory["model_result"] = run_arima_forecast(series, order, steps=steps)

    def run_diagnostics(self) -> None:
        mr = self.memory.get("model_result")
        if not mr:
            return
        self.memory["diagnostics"] = run_diagnostics_for_result(mr, self.df)

    def compare_if_multiple(self, other_results: list[dict] | None = None) -> None:
        base = []
        if self.memory.get("model_result"):
            base.append(self.memory["model_result"])
        if other_results:
            base.extend(other_results)
        if len(base) >= 1:
            self.memory["comparison"] = compare_models(base)

    def generate_report_text(self) -> str:
        mr = self.memory.get("model_result", {})
        diag = self.memory.get("diagnostics", {})
        return generate_plain_english_summary(mr, diag)

    def serializable_payload(self) -> dict:
        """UI / export: strip non-serializable fitted models."""
        out = {}
        for k, v in self.memory.items():
            if k == "model_result" and isinstance(v, dict):
                out[k] = strip_internal_keys(v)
            else:
                out[k] = v
        return out

    def run(
        self,
        formula: str | None = None,
        model_kind: str | None = None,
    ) -> dict:
        self.profile_data()
        self.build_plan()
        if formula and model_kind:
            self.fit_primary_model(formula, model_kind)
            self.run_diagnostics()
            self.memory["plain_english"] = self.generate_report_text()
        return self.memory
