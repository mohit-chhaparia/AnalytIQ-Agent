"""Dynamic analysis scaffold: plan + code template + validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from agents.code_validator import validate_generated_code
from agents.engine_router import choose_engine


@dataclass
class DynamicAnalysisAgent:
    df: pd.DataFrame
    user_goal: str
    outcome: str
    llm_generate: Callable[[str], str] | None = None

    def build_prompt(self, engine: str) -> str:
        cols = list(self.df.columns)
        return f"""You are generating code for a local statistical analysis agent.
Task: {self.user_goal}
Columns: {cols}
Outcome: {self.outcome}
Engine: {engine}

Requirements:
1. Read data from a path provided by the caller (do not hardcode secrets).
2. Save numerical results to JSON or text for downstream reporting.
3. Do not use subprocess, os.remove, shutil.rmtree, sockets, or external APIs.
4. Use fixed random seeds where applicable.
Return only code."""

    def run(self) -> dict[str, Any]:
        engine = choose_engine(self.user_goal)
        plan = {
            "engine": engine,
            "note": "SAS is not used in this project; prefer statsmodels or R for regulated-style GLMs.",
        }
        if self.llm_generate is None:
            return {
                "analysis_plan": plan,
                "generated_code": "",
                "validation": {"is_safe": False, "problems": ["No LLM client configured."]},
                "summary": "Configure an optional local LLM (LM Studio or Ollama) to generate code dynamically.",
            }
        code = self.llm_generate(self.build_prompt(engine))
        validation = validate_generated_code(code)
        return {
            "analysis_plan": plan,
            "generated_code": code,
            "validation": validation,
            "summary": "Generated code is ready for manual review or sandbox execution when validation passes.",
        }
