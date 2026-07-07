"""
Handles analysis requests that are NOT in the deterministic capability registry.

Flow:
  1. Understand the requested analysis (LLM)
  2. Choose an execution engine (engine_router)
  3. Generate code (LLM)
  4. Validate code (code_validator)
  5. Execute code locally (sandbox)
  6. Parse and validate output
  7. Summarise results (LLM rewriter)
  8. Optionally save as a reusable plugin
"""

import os
import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional

from agents.engine_router import choose_engine, get_engine_description
from agents.code_validator import validate_generated_code, strip_markdown_fences
from agents.llm_rewriter import generate_analysis_plan, generate_analysis_code, rewrite_summary

SANDBOX_DIR = Path(os.environ.get("SANDBOX_DIR", "sandbox"))
PLUGINS_DIR = Path(os.environ.get("PLUGINS_DIR", "plugins"))
USE_DOCKER   = os.environ.get("USE_DOCKER", "false").lower() == "true"
DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "stat-agent-python")


class DynamicAnalysisAgent:
    """
    Executes analyses not available in the deterministic tool registry.
    """

    def __init__(
        self,
        df,
        user_goal: str,
        outcome: str,
        outcome_type: str,
        available_tools: list,
        user_engine_pref: Optional[str] = None,
    ):
        self.df             = df
        self.user_goal      = user_goal
        self.outcome        = outcome
        self.outcome_type   = outcome_type
        self.available_tools = available_tools
        self.user_engine_pref = user_engine_pref
        self.memory         = {}

    def run(self) -> dict:
        """Execute the full dynamic analysis pipeline."""

        # Step 1: generate analysis plan via LLM
        profile_summary = (
            f"Outcome: {self.outcome} ({self.outcome_type}). "
            f"Columns: {list(self.df.columns[:10])}. "
            f"Rows: {len(self.df)}."
        )
        plan = generate_analysis_plan(
            user_goal       = self.user_goal,
            profile_summary = profile_summary,
            outcome_type    = self.outcome_type,
            available_tools = self.available_tools,
        )
        self.memory["analysis_plan"] = plan

        # Step 2: choose engine
        engine = choose_engine(
            plan.get("analysis_type", self.user_goal),
            self.user_engine_pref,
        )
        self.memory["engine"] = engine
        self.memory["engine_description"] = get_engine_description(engine)

        # Step 3: generate code
        code_prompt = plan.get("code_prompt") or (
            f"Perform {plan.get('analysis_type', 'the requested analysis')} "
            f"on the dataset. Outcome variable: {self.outcome}. "
            f"Goal: {self.user_goal}."
        )
        raw_code = generate_analysis_code(code_prompt, engine)
        code = strip_markdown_fences(raw_code)
        self.memory["generated_code"] = code

        # Step 4: validate code
        validation = validate_generated_code(code)
        self.memory["code_validation"] = validation

        if not validation["is_safe"]:
            return {
                **self.memory,
                "status":  "blocked",
                "message": "Generated code failed safety validation.",
                "problems": validation["problems"],
            }

        # Step 5: prepare sandbox & execute
        sandbox = self._prepare_sandbox()
        try:
            result = self._execute(code, engine, sandbox)
            self.memory["execution_result"] = result
        except Exception as e:
            return {**self.memory, "status": "execution_error", "message": str(e)}
        finally:
            self._cleanup_sandbox(sandbox)

        # Step 6: parse output
        output = self._parse_output(sandbox)
        self.memory["output"] = output

        # Step 7: LLM summary
        raw_summary = output.get("model_summary", json.dumps(output.get("metrics", {}), indent=2))
        plain_summary = rewrite_summary(raw_summary, context=f"Analysis: {plan.get('analysis_type')}")
        self.memory["plain_english_summary"] = plain_summary

        # Step 8: save as plugin
        if output.get("status") == "success":
            self._save_plugin(plan, code, engine)

        return {**self.memory, "status": "success"}

    # private helpers

    def _prepare_sandbox(self) -> Path:
        sandbox = SANDBOX_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        (sandbox / "input").mkdir(parents=True, exist_ok=True)
        (sandbox / "generated").mkdir(parents=True, exist_ok=True)
        (sandbox / "output" / "plots").mkdir(parents=True, exist_ok=True)

        # Write dataset
        self.df.to_csv(sandbox / "input" / "dataset.csv", index=False)
        return sandbox

    def _execute(self, code: str, engine: str, sandbox: Path) -> dict:
        # Patch paths in the code to point at sandbox directories
        code = code.replace("/input/", str(sandbox / "input") + "/")
        code = code.replace("/output/", str(sandbox / "output") + "/")

        script_path = sandbox / "generated" / f"analysis.{_ext(engine)}"
        script_path.write_text(code)

        if USE_DOCKER and engine == "python":
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{sandbox}/input:/input",
                "-v", f"{sandbox}/generated:/generated",
                "-v", f"{sandbox}/output:/output",
                DOCKER_IMAGE, "python", "/generated/analysis.py",
            ]
        elif engine == "python":
            cmd = ["python", str(script_path)]
        elif engine == "r":
            cmd = ["Rscript", str(script_path)]
        elif engine == "sas":
            return self._execute_sas(code)
        else:
            cmd = ["python", str(script_path)]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "returncode": proc.returncode,
            "stdout":     proc.stdout[-3000:],
            "stderr":     proc.stderr[-2000:],
        }

    def _execute_sas(self, code: str) -> dict:
        try:
            import saspy
            sas = saspy.SASsession()
            sas.df2sd(self.df, table="analysis_data")
            result = sas.submit(code)
            return {"returncode": 0, "stdout": result.get("LST", ""), "stderr": result.get("LOG", "")}
        except ImportError:
            return {"returncode": 1, "stdout": "", "stderr": "saspy not installed"}

    def _parse_output(self, sandbox: Path) -> dict:
        results_path = sandbox / "output" / "results.json"
        if results_path.exists():
            try:
                return json.loads(results_path.read_text())
            except Exception:
                pass
        # Fallback: return raw stdout
        return {
            "status": "partial",
            "model_summary": self.memory.get("execution_result", {}).get("stdout", ""),
        }

    def _cleanup_sandbox(self, sandbox: Path):
        try:
            shutil.rmtree(sandbox)
        except Exception:
            pass

    def _save_plugin(self, plan: dict, code: str, engine: str):
        PLUGINS_DIR.mkdir(exist_ok=True)
        analysis_type = plan.get("analysis_type", "custom_analysis")
        safe_name = analysis_type.lower().replace(" ", "_")
        ext = _ext(engine)

        plugin_file = PLUGINS_DIR / f"{safe_name}.{ext}"
        plugin_file.write_text(code)

        meta = {
            "name":             safe_name,
            "language":         engine,
            "analysis_type":    analysis_type,
            "outcome_type":     self.outcome_type,
            "created":          datetime.now().isoformat(),
            "status":           "validated",
            "plan":             plan,
        }
        (PLUGINS_DIR / f"{safe_name}.json").write_text(json.dumps(meta, indent=2))


def _ext(engine: str) -> str:
    return {"python": "py", "r": "R", "sas": "sas"}.get(engine, "py")


# Plugin loader

def load_plugin(analysis_type: str) -> Optional[dict]:
    """
    Check the plugin library for a previously generated analysis.
    Returns the plugin metadata+code if found, else None.
    """
    safe_name = analysis_type.lower().replace(" ", "_")
    meta_path = PLUGINS_DIR / f"{safe_name}.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    engine = meta.get("language", "python")
    code_path = PLUGINS_DIR / f"{safe_name}.{_ext(engine)}"
    if code_path.exists():
        meta["code"] = code_path.read_text()
    return meta
