"""Write a Quarto report from structured analysis outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def render_quarto_report(
    output_dir: Path,
    analysis_goal: str,
    data_quality_json: str,
    model_recommendations_json: str,
    fitted_summary_text: str,
    diagnostics_text: str,
    plain_english: str,
    to: str = "html",
) -> tuple[Path | None, str]:
    """
    Creates `generated_report.qmd` under output_dir and runs `quarto render` if available.
    Returns (path to output artifact or None, status message).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qmd = output_dir / "generated_report.qmd"
    body = f"""---
title: "Automated Statistical Modeling Report"
format:
  html: default
---

## 1. Analysis Goal

{analysis_goal}

## 2. Data Quality Summary

```json
{data_quality_json}
```

## 3. Recommended Models

```json
{model_recommendations_json}
```

## 4. Fitted Model

```
{fitted_summary_text}
```

## 5. Diagnostics

```
{diagnostics_text}
```

## 6. Plain-English Interpretation

{plain_english}
"""
    qmd.write_text(body, encoding="utf-8")
    try:
        subprocess.run(
            ["quarto", "render", str(qmd), "--to", to],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, "Quarto CLI not found; .qmd written but not rendered."
    except subprocess.CalledProcessError as exc:
        return qmd, f"Quarto render failed: {exc.stderr or exc.stdout}"

    ext = "html" if to == "html" else "pdf" if to == "pdf" else to
    out = output_dir / f"generated_report.{ext}"
    return out if out.exists() else qmd.with_suffix(f".{ext}"), "ok"


def dumps_compact(obj) -> str:
    return json.dumps(obj, indent=2, default=str)[:50000]
