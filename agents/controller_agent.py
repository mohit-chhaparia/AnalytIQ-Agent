"""
Rule-based orchestration pipeline for AnalytIQ Agent.

For a fully agentic loop where Claude decides the workflow dynamically, use ClaudeToolAgent from agents/claude_tool_agent.py instead.

This controller runs a deterministic sequence — useful when you want predictable, fast analysis without API calls on every step.
"""

import json
import traceback
from typing import Optional

import numpy as np

from agents.data_profiler import profile_dataframe
from agents.eda_agent import recommend_eda_plots
from agents.model_selector import recommend_models
from agents.model_runner import run_model
from agents.diagnostics_agent import run_diagnostics, interpret_diagnostics
from agents.threshold_tuning import tune_thresholds, interpret_threshold_results
from agents.model_comparison_agent import compare_models
from agents.report_agent import generate_plain_english_summary
from agents.planning_agent import create_analysis_plan
from agents.capability_registry import find_capability, list_capabilities
from agents.dynamic_analysis_agent import DynamicAnalysisAgent, load_plugin
from agents.llm_rewriter import (
    rewrite_summary,
    interpret_diagnostics_with_llm,
)
from agents.llm_router import any_llm_available


class StatisticalAnalysisAgent:
    """
    Deterministic statistical analysis pipeline.

    Parameters
    ----------
    df           : pandas DataFrame
    user_goal    : natural-language analysis goal
    outcome      : name of the outcome/target variable
    formula      : optional patsy formula (auto-built if omitted)
    engine_pref  : preferred execution engine ("python" | "r")
    use_llm      : if True and any LLM backend is configured, use LLM for
                   plain-English summaries and diagnostic interpretation.
                   Defaults to auto-detect (True when a key is available).
    use_dynamic  : whether to attempt dynamic code generation for unknown analyses
    """

    def __init__(
        self,
        df,
        user_goal:   str,
        outcome:     str,
        formula:     Optional[str] = None,
        engine_pref: Optional[str] = None,
        use_llm:     Optional[bool] = None,   # None = auto-detect
        use_dynamic: bool = True,
    ):
        self.df          = df
        self.user_goal   = user_goal
        self.outcome     = outcome
        self.formula     = formula
        self.engine_pref = engine_pref
        self.use_llm     = any_llm_available() if use_llm is None else use_llm
        self.use_dynamic = use_dynamic
        self.memory      = {}
        self.logs        = []

    def run(self) -> dict:
        """Execute the full pipeline. Returns self.memory."""
        steps = [
            "profile_data", "create_plan", "eda",
            "select_models", "run_models", "run_diagnostics",
            "compare_models", "generate_report",
        ]
        for step in steps:
            try:
                self.logs.append(f"[STEP] {step}")
                getattr(self, f"_run_{step}")()
            except Exception as e:
                self.logs.append(f"[ERROR in {step}] {e}")
                self.memory["error"] = str(e)
                break

        self.memory["logs"] = self.logs
        return self.memory

    # pipeline steps

    def _run_profile_data(self):
        self.memory["profile"] = profile_dataframe(self.df)
        shape = self.memory["profile"]["shape"]
        self.logs.append(f"  Profiled {shape['rows']} rows × {shape['columns']} columns.")

    def _run_create_plan(self):
        plan = create_analysis_plan(self.memory["profile"], self.outcome, self.user_goal)
        self.memory["plan"]      = plan
        self.memory["goal_type"] = plan["goal_type"]
        self.logs.append(f"  Goal type: {plan['goal_type']}")

    def _run_eda(self):
        self.memory["eda_recommendations"] = recommend_eda_plots(
            self.memory["profile"], self.outcome
        )

    def _run_select_models(self):
        recs = recommend_models(self.memory["profile"], self.outcome, self.user_goal)
        self.memory["model_recommendations"] = recs
        names = [r["model"] for r in recs.get("recommendations", [])]
        self.logs.append(f"  Recommended: {names}")

    def _run_run_models(self):
        fitted = []
        for model_def in self.memory["plan"].get("candidate_models", []):
            model_name = model_def.get("name", "")
            _, cap = find_capability(model_name)
            if cap and cap.get("supported"):
                self.logs.append(f"  Running: {model_name}")
                try:
                    result = run_model(
                        model_name = model_name,
                        df         = self.df,
                        formula    = self.formula or _auto_formula(
                            self.outcome, self.df, self.memory["profile"]
                        ),
                        outcome    = self.outcome,
                        engine     = cap.get("engine", "python"),
                    )
                    fitted.append(result)
                except Exception as e:
                    self.logs.append(f"  [WARN] {model_name} failed: {e}")

        # Dynamic fallback
        if not fitted and self.use_dynamic:
            goal_type = self.memory.get("goal_type", "unknown")
            plugin = load_plugin(goal_type)
            if plugin:
                self.logs.append(f"  Loading plugin: {plugin['name']}")
                fitted.append({"model_type": plugin["name"], "source": "plugin"})
            else:
                self.logs.append("  Entering dynamic analysis mode.")
                dyn = DynamicAnalysisAgent(
                    df               = self.df,
                    user_goal        = self.user_goal,
                    outcome          = self.outcome,
                    outcome_type     = goal_type,
                    available_tools  = list_capabilities(),
                    user_engine_pref = self.engine_pref,
                ).run()
                self.memory["dynamic_result"] = dyn
                if dyn.get("status") == "success":
                    fitted.append({
                        "model_type": dyn.get("analysis_plan", {}).get("analysis_type", "Dynamic Analysis"),
                        "source":     "dynamic",
                        **dyn.get("output", {}),
                    })

        self.memory["fitted_models"] = fitted

    def _run_run_diagnostics(self):
        all_diagnostics = []
        for result in self.memory.get("fitted_models", []):
            if result.get("source") in ("dynamic", "plugin"):
                continue

            diag  = run_diagnostics(result)
            notes = interpret_diagnostics(diag, result)

            # LLM-enhanced interpretation when available
            llm_notes = ""
            if self.use_llm and notes:
                llm_notes = interpret_diagnostics_with_llm(
                    notes,
                    result.get("model_type", ""),
                    context=f"Outcome: {self.outcome}. Goal: {self.user_goal}.",
                )

            entry = {
                "model":       result.get("model_type"),
                "diagnostics": diag,
                "notes":       notes,
                "llm_notes":   llm_notes,
            }

            # Threshold tuning for logistic models
            if "logistic" in result.get("model_type", "").lower():
                probs = result.get("predicted_probabilities")
                if probs:
                    y_col = self.df[self.outcome]
                    if y_col.dtype == object:
                        uniq = sorted(y_col.unique())
                        y_true = y_col.map({uniq[0]: 0, uniq[1]: 1}).values
                    else:
                        y_true = y_col.values
                    tuning = tune_thresholds(y_true, np.array(probs))
                    entry["threshold_tuning"] = tuning
                    entry["notes"] += interpret_threshold_results(tuning)

            all_diagnostics.append(entry)

            # Diagnostic feedback loop
            for note in notes:
                nl = note.lower()
                if "overdispersion" in nl:
                    self.memory.setdefault("revisions", []).append(
                        "Overdispersion detected in Poisson model. Consider quasi-Poisson or Negative Binomial."
                    )
                if "multicollinearity" in nl or "vif" in nl:
                    self.memory.setdefault("revisions", []).append(
                        "High VIF detected. Consider removing correlated predictors or using regularisation."
                    )

        self.memory["diagnostics"] = all_diagnostics

    def _run_compare_models(self):
        fitted = [
            m for m in self.memory.get("fitted_models", [])
            if m.get("source") not in ("dynamic", "plugin")
        ]
        if len(fitted) > 1:
            comp = compare_models(fitted)
            self.memory["model_comparison"]  = comp
            self.memory["best_model_result"] = comp.get("best_result")
        elif fitted:
            self.memory["best_model_result"] = fitted[0]

    def _run_generate_report(self):
        best = self.memory.get("best_model_result") or {}
        diag_notes = []
        for d in self.memory.get("diagnostics", []):
            diag_notes.extend(d.get("notes", []))

        raw = generate_plain_english_summary(best, diag_notes)

        enhanced = raw
        if self.use_llm and best:
            context  = f"User goal: {self.user_goal}. Outcome: {self.outcome}."
            enhanced = rewrite_summary(raw, context=context)

        self.memory["report"] = {
            "analysis_goal":     self.user_goal,
            "outcome":           self.outcome,
            "goal_type":         self.memory.get("goal_type", "unknown"),
            "plain_english":     enhanced,
            "raw_summary":       raw,
            "diagnostics_notes": diag_notes,
            "revisions":         self.memory.get("revisions", []),
            "dynamic_used":      "dynamic_result" in self.memory,
        }


# helpers

def _auto_formula(outcome: str, df, profile: dict) -> str:
    terms = []
    for col in profile.get("columns", []):
        if col["name"] == outcome:
            continue
        if col["inferred_type"] in ("categorical", "binary", "numeric_discrete_or_categorical"):
            terms.append(f"C({col['name']})")
        elif col["inferred_type"] == "continuous_numeric":
            terms.append(col["name"])
    return f"{outcome} ~ " + " + ".join(terms) if terms else f"{outcome} ~ 1"
