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


# context builder (handles both agent memory formats)

def _build_context(memory: dict) -> dict:
    report  = memory.get("report", {})
    profile = memory.get("profile", {})
    diag    = memory.get("diagnostics", [])

    # Best model — works for both agent types
    best = (
        memory.get("best_model_result")
        or (memory.get("model_results") or [{}])[-1]
    )

    # Plan — StatisticalAnalysisAgent has it; ClaudeToolAgent doesn't
    plan = memory.get("plan", {})

    # Diagnostics
    all_diag_notes = []
    for d in diag:
        all_diag_notes.extend(d.get("notes", []))
    diag_md = "\n".join(f"- {n}" for n in all_diag_notes) or "_No diagnostic issues flagged._"

    # Threshold tuning
    threshold_section = False
    youden = {}
    f1_thresh = {}
    for d in diag:
        tt = d.get("threshold_tuning")
        if tt:
            threshold_section = True
            youden    = tt.get("best_youden", {})
            f1_thresh = tt.get("best_f1", {})
            break

    # Model comparison
    comp        = memory.get("model_comparison", {})
    comp_md     = _table_to_md(comp.get("comparison_table", []))
    comp_notes  = comp.get("rationale", "")

    # EDA
    eda_recs = memory.get("eda_recommendations", [])
    eda_md   = "\n".join(
        f"- **{r.get('plot','Plot')}**: {r.get('purpose','')}" for r in eda_recs[:8]
    ) or "_Run EDA on Page 2 to see recommendations._"

    # Model recommendations
    rec_list = memory.get("model_recommendations", {}).get("recommendations", [])
    recs_md  = "\n".join(
        f"- **{r['model']}**: {r.get('reason','')}" for r in rec_list
    ) or "_See Page 3 for model recommendations._"

    # Engine
    engine = "python"
    if plan.get("candidate_models"):
        engine = plan["candidate_models"][0].get("engine", "python")

    # For ClaudeToolAgent: final_narrative is the report
    plain_english = (
        report.get("plain_english")
        or memory.get("final_narrative")
        or "_No summary generated._"
    )

    n_missing = sum(1 for c in profile.get("columns", []) if c.get("missing_pct", 0) > 0)
    dup_rows  = profile.get("duplicates", {}).get("duplicate_rows", 0)
    n_hc      = sum(
        1 for c in profile.get("columns", [])
        if c.get("inferred_type") == "text_or_high_cardinality_categorical"
    )

    metrics    = best.get("metrics", {})
    metrics_md = _dict_to_md(metrics) if metrics else "_No classification metrics._"

    return {
        "analysis_goal":         report.get("analysis_goal", memory.get("user_goal", "")),
        "outcome":               report.get("outcome", ""),
        "goal_type":             report.get("goal_type", plan.get("goal_type", "")),
        "report_date":           datetime.now().strftime("%B %d, %Y"),
        "n_rows":                profile.get("shape", {}).get("rows", "—"),
        "n_cols":                profile.get("shape", {}).get("columns", "—"),
        "missing_cols":          n_missing,
        "duplicate_rows":        dup_rows,
        "high_cardinality_cols": n_hc,
        "outcome_type":          report.get("goal_type", ""),
        "best_model_type":       best.get("model_type", "N/A"),
        "formula":               best.get("formula", "N/A"),
        "engine":                engine,
        "model_summary_text":    (best.get("summary", "") or "")[:3000],
        "metrics_table":         metrics_md,
        "diagnostics_summary":   diag_md,
        "threshold_section":     threshold_section,
        "threshold_summary":     "",
        "youden_threshold":      youden.get("threshold", "—"),
        "youden_sens":           youden.get("sensitivity", "—"),
        "youden_spec":           youden.get("specificity", "—"),
        "f1_threshold":          f1_thresh.get("threshold", "—"),
        "f1_recall":             f1_thresh.get("sensitivity", "—"),
        "f1_score":              f1_thresh.get("f1", "—"),
        "comparison_table":      bool(comp_md),
        "comparison_notes":      comp_notes,
        "comparison_table_md":   comp_md,
        "plain_english_summary": plain_english,
        "revisions":             report.get("revisions", []),
        "dynamic_used":          report.get("dynamic_used", bool(memory.get("dynamic_result"))),
        "agent_version":         "1.0.0",
        "eda_summary":           eda_md,
        "model_recommendations": recs_md,
        "data_quality_summary":  _profile_to_md(profile),
        "data_path":             "",  # filled after save
    }


