"""
Fills the Quarto report template and renders to HTML.
Handles memory dicts from both StatisticalAnalysisAgent and ClaudeToolAgent.

Install Quarto: https://quarto.org/docs/get-started/
"""

import os
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from jinja2 import Template

TEMPLATE_PATH = Path("reports/report_template.qmd")
OUTPUT_DIR    = Path("reports/outputs")


def render_report(memory: dict, output_name: str = None) -> dict:
    """
    Fill the Quarto template and render to HTML.

    Accepts memory from either:
      - StatisticalAnalysisAgent.run()  (has 'plan', 'fitted_models', etc.)
      - ClaudeToolAgent.run()           (has 'tool_log', 'final_narrative', etc.)

    Returns
    -------
    {"html_path", "qmd_path", "data_path", "status", "message"}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        output_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ctx = _build_context(memory)

    # Save data JSON for reference
    data_path = OUTPUT_DIR / f"{output_name}_data.json"
    data_path.write_text(json.dumps(_make_serialisable(memory), indent=2, default=str))
    ctx["data_path"] = str(data_path)

    # Fill template
    if not TEMPLATE_PATH.exists():
        return {"status": "error", "message": f"Template not found: {TEMPLATE_PATH}",
                "html_path": None, "qmd_path": None, "data_path": str(data_path)}

    filled = Template(TEMPLATE_PATH.read_text()).render(**ctx)
    qmd_path = OUTPUT_DIR / f"{output_name}.qmd"
    qmd_path.write_text(filled)

    # Render with Quarto
    html_path = OUTPUT_DIR / f"{output_name}.html"
    quarto = shutil.which("quarto")
    if quarto:
        try:
            subprocess.run(
                [quarto, "render", str(qmd_path), "--to", "html"],
                check=True, capture_output=True, timeout=120
            )
            # Quarto sometimes writes next to the .qmd file
            if not html_path.exists():
                candidate = qmd_path.with_suffix(".html")
                if candidate.exists():
                    shutil.move(str(candidate), str(html_path))
        except subprocess.CalledProcessError as e:
            return {
                "status":    "error",
                "message":   f"Quarto render failed: {e.stderr.decode()[:500]}",
                "html_path": None,
                "qmd_path":  str(qmd_path),
                "data_path": str(data_path),
            }
    else:
        # Quarto not installed — write a plain HTML fallback
        html_path = OUTPUT_DIR / f"{output_name}_plain.html"
        html_path.write_text(
            "<html><body><pre style='font-family:sans-serif;padding:2em'>"
            + filled.replace("<", "&lt;").replace(">", "&gt;")
            + "</pre></body></html>"
        )

    return {
        "status":    "success",
        "message":   "Report rendered." if quarto else "Quarto not found — plain HTML saved.",
        "html_path": str(html_path) if html_path.exists() else None,
        "qmd_path":  str(qmd_path),
        "data_path": str(data_path),
    }


