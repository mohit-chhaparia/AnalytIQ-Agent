"""
Replaces the hard-coded pipeline in controller_agent.py with a genuine ReAct (Reason + Act) loop powered by Claude's native tool use.

How it works:
  1. Claude is given the user goal, dataset profile, and a set of tools
     (the deterministic statistical functions as JSON schema definitions).
  2. Claude reasons about what to do next, picks a tool, and calls it.
  3. The result is fed back to Claude as a tool result.
  4. Claude continues until it decides the analysis is complete.
  5. Claude writes the final plain-English report.

This makes the controller genuinely agentic:
  - Claude decides the analysis order (not a fixed script).
  - Claude revises its plan if diagnostics reveal problems.
  - Claude can call tools multiple times (e.g. run logistic, check AUC,
    then tune threshold, then re-run with a better formula).
  - Claude explains every decision in the final report.
"""

import os
import json
import traceback
import pandas as pd
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL_AGENT", os.environ.get("CLAUDE_MODEL", "claude-opus-4-6"))
MAX_TOOL_ROUNDS   = int(os.environ.get("CLAUDE_MAX_ROUNDS", "12"))


# Tool definitions (Claude's JSON schema)

TOOL_DEFINITIONS = [
    {
        "name": "profile_dataset",
        "description": (
            "Profile a dataset: detect column types, missing values, duplicates, "
            "outliers, and suspicious categories. Always call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "recommend_eda_plots",
        "description": "Recommend starter visualisations based on the dataset profile and outcome variable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "description": "Name of the outcome/target variable."},
            },
            "required": ["outcome"],
        },
    },
    {
        "name": "recommend_models",
        "description": "Recommend candidate statistical models based on the outcome type and user goal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string"},
                "goal":    {"type": "string", "description": "User's analysis goal in plain English."},
            },
            "required": ["outcome", "goal"],
        },
    },
    {
        "name": "run_logistic_regression",
        "description": (
            "Fit a logistic regression model. Use for binary outcomes. "
            "Returns AIC, BIC, AUC, accuracy, sensitivity, and predicted probabilities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "Patsy formula, e.g. 'Churn ~ C(Contract) + tenure + MonthlyCharges'",
                },
                "outcome": {"type": "string"},
            },
            "required": ["formula", "outcome"],
        },
    },
    {
        "name": "run_linear_regression",
        "description": "Fit OLS linear regression. Use for continuous outcomes. Returns R², AIC, BIC.",
        "input_schema": {
            "type": "object",
            "properties": {
                "formula": {"type": "string"},
            },
            "required": ["formula"],
        },
    },
    {
        "name": "run_poisson_regression",
        "description": "Fit Poisson regression. Use for count outcomes. Returns AIC, BIC, dispersion statistic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "formula": {"type": "string"},
            },
            "required": ["formula"],
        },
    },
    {
        "name": "run_diagnostics",
        "description": (
            "Run model diagnostics on a fitted model result. "
            "Returns normality checks, VIF (multicollinearity), "
            "overdispersion flags, and influential point detection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "e.g. 'logistic', 'linear', 'poisson'",
                },
            },
            "required": ["model_type"],
        },
    },
    {
        "name": "tune_classification_threshold",
        "description": (
            "For logistic regression: sweep thresholds 0.1–0.9 and return "
            "sensitivity, specificity, F1, and Youden's J at each threshold. "
            "Call this after run_logistic_regression when threshold optimisation matters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compare_models",
        "description": "Compare all fitted models by AIC, BIC, AUC, and R². Returns ranked table and best model.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Generate the final plain-English report. Call this last, "
            "after all modelling and diagnostics are complete. "
            "Returns the complete analysis summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key_findings": {
                    "type": "string",
                    "description": "Your summary of the key findings from this analysis.",
                },
                "recommended_model": {
                    "type": "string",
                    "description": "The model you recommend as the final answer.",
                },
            },
            "required": ["key_findings", "recommended_model"],
        },
    },
]


# Tool executor

class ToolExecutor:
    """Executes tool calls using the actual deterministic agent modules."""

    def __init__(self, df: pd.DataFrame, outcome: str, user_goal: str):
        self.df          = df
        self.outcome     = outcome
        self.user_goal   = user_goal
        self.profile     = None
        self.model_results = []
        self.last_logistic_result = None

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call and return the result as a JSON string."""
        try:
            result = self._dispatch(tool_name, tool_input)
            return json.dumps(result, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()[-500:]})

    def _dispatch(self, name: str, inp: dict) -> dict:
        from agents.data_profiler    import profile_dataframe
        from agents.eda_agent        import recommend_eda_plots
        from agents.model_selector   import recommend_models
        from agents.model_runner     import (
            run_logistic_regression, run_linear_regression, run_poisson_regression
        )
        from agents.diagnostics_agent import run_diagnostics, interpret_diagnostics
        from agents.threshold_tuning  import tune_thresholds, interpret_threshold_results
        from agents.model_comparison_agent import compare_models

        if name == "profile_dataset":
            self.profile = profile_dataframe(self.df)
            # Return a compact summary so it fits in Claude's context
            return {
                "shape":       self.profile["shape"],
                "duplicates":  self.profile["duplicates"],
                "columns":     [
                    {k: v for k, v in c.items() if k != "sample_values"}
                    for c in self.profile["columns"]
                ],
            }

        if name == "recommend_eda_plots":
            if not self.profile:
                return {"error": "Call profile_dataset first."}
            return {"plots": recommend_eda_plots(self.profile, inp["outcome"])}

        if name == "recommend_models":
            if not self.profile:
                return {"error": "Call profile_dataset first."}
            return recommend_models(self.profile, inp["outcome"], inp.get("goal", self.user_goal))

        if name == "run_logistic_regression":
            result = run_logistic_regression(self.df, inp["formula"], inp["outcome"])
            self.model_results.append(result)
            self.last_logistic_result = result
            # Return summary without the full model text (too long for context)
            return {k: v for k, v in result.items() if k not in ("summary", "predicted_probabilities")}

        if name == "run_linear_regression":
            result = run_linear_regression(self.df, inp["formula"])
            self.model_results.append(result)
            return {k: v for k, v in result.items() if k not in ("summary", "residuals", "fitted_values")}

        if name == "run_poisson_regression":
            result = run_poisson_regression(self.df, inp["formula"])
            self.model_results.append(result)
            return {k: v for k, v in result.items() if k != "summary"}

        if name == "run_diagnostics":
            if not self.model_results:
                return {"error": "No model results yet. Run a model first."}
            last = self.model_results[-1]
            diag  = run_diagnostics(last)
            notes = interpret_diagnostics(diag, last)
            return {"diagnostics": diag, "notes": notes}

        if name == "tune_classification_threshold":
            if not self.last_logistic_result:
                return {"error": "Run run_logistic_regression first."}
            probs  = np.array(self.last_logistic_result["predicted_probabilities"])
            y_col  = self.df[self.outcome]
            y_true = (y_col.map({sorted(y_col.unique())[0]: 0, sorted(y_col.unique())[1]: 1})
                      if y_col.dtype == object else y_col.values)
            tuning = tune_thresholds(y_true, probs)
            notes  = interpret_threshold_results(tuning)
            return {
                "best_youden": tuning["best_youden"],
                "best_f1":     tuning["best_f1"],
                "auc":         tuning["auc"],
                "notes":       notes,
            }

        if name == "compare_models":
            if len(self.model_results) < 2:
                return {"message": "Only one model fitted — no comparison needed.", "best": self.model_results[0] if self.model_results else {}}
            comp = compare_models(self.model_results)
            return {"best_model": comp["best_model"], "rationale": comp["rationale"],
                    "table": comp["comparison_table"]}

        if name == "generate_report":
            return {
                "status":              "report_ready",
                "key_findings":        inp.get("key_findings", ""),
                "recommended_model":   inp.get("recommended_model", ""),
                "models_fitted":       [r.get("model_type") for r in self.model_results],
                "message":             "Report generation complete.",
            }

        return {"error": f"Unknown tool: {name}"}


# Main Claude ReAct agent

class ClaudeToolAgent:
    """
    Uses Claude's native tool use to orchestrate the full statistical analysis.

    This is a genuine ReAct loop:
      Claude reasons → calls a tool → reviews result → reasons again → ...
      until Claude decides the analysis is complete.

    Advantages over the rule-based controller:
      - Claude adapts the analysis sequence to what it finds in the data.
      - Claude revises its plan when diagnostics reveal problems.
      - Claude explains its reasoning at every step.
      - Claude can call the same tool multiple times with different inputs.
    """

    def __init__(
        self,
        df:            pd.DataFrame,
        user_goal:     str,
        outcome:       str,
        max_rounds:    int = MAX_TOOL_ROUNDS,
    ):
        self.df         = df
        self.user_goal  = user_goal
        self.outcome    = outcome
        self.max_rounds = max_rounds
        self.executor   = ToolExecutor(df, outcome, user_goal)
        self.messages   = []
        self.tool_log   = []

    def run(self) -> dict:
        """Execute the full ReAct loop. Returns the final memory dict."""
        if not ANTHROPIC_API_KEY:
            return {
                "status": "skipped",
                "reason": "ANTHROPIC_API_KEY not set in .env",
                "fallback": "Use StatisticalAnalysisAgent from controller_agent.py instead.",
            }

        try:
            import anthropic
        except ImportError:
            return {"status": "error", "reason": "Run: pip install anthropic"}

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # System prompt
        system = f"""You are an expert statistical analyst and data scientist.
You have access to a suite of deterministic statistical tools.

Your job is to conduct a complete statistical analysis by calling these tools
in whatever order makes sense, reviewing results, and revising your approach
when needed.

Dataset: {len(self.df)} rows × {len(self.df.columns)} columns.
Columns available: {list(self.df.columns)}.
Outcome variable: {self.outcome}.
User goal: {self.user_goal}

Rules:
- Always start by calling profile_dataset.
- Choose models based on what the profile tells you about the outcome type.
- If diagnostics reveal problems (overdispersion, multicollinearity, poor AUC),
  adjust your approach — try a different model, different formula, or threshold tuning.
- Call generate_report last to finalise the analysis.
- Be precise and statistical in your reasoning between tool calls.
- Do not invent numbers — only report what the tools return."""

        # Initial user message
        self.messages = [
            {"role": "user", "content": f"Please conduct a complete statistical analysis. Goal: {self.user_goal}"}
        ]

        # ReAct loop
        final_text = ""
        for round_num in range(self.max_rounds):
            response = client.messages.create(
                model     = CLAUDE_MODEL,
                max_tokens= 4096,
                system    = system,
                tools     = TOOL_DEFINITIONS,
                messages  = self.messages,
            )

            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": response.content})

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Claude is done — extract final text
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                break

            if response.stop_reason != "tool_use":
                break

            # Execute all tool calls in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_result = self.executor.execute(block.name, block.input)
                    self.tool_log.append({
                        "round":  round_num + 1,
                        "tool":   block.name,
                        "input":  block.input,
                        "result": json.loads(tool_result) if tool_result else {},
                    })
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     tool_result,
                    })

            # Feed tool results back to Claude
            self.messages.append({"role": "user", "content": tool_results})

        # Package results
        return {
            "status":           "success",
            "model_used":       CLAUDE_MODEL,
            "rounds":           len(self.tool_log),
            "tool_log":         self.tool_log,
            "final_narrative":  final_text,
            "profile":          self.executor.profile,
            "model_results":    self.executor.model_results,
            "best_model_result":self.executor.model_results[-1] if self.executor.model_results else {},
            "report": {
                "plain_english": final_text,
                "analysis_goal": self.user_goal,
                "outcome":       self.outcome,
                "dynamic_used":  False,
            },
        }
