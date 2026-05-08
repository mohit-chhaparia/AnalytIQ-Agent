"""Invoke R_engine scripts via Rscript."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_r_script(
    script_name: str,
    input_csv: str | Path,
    formula: str,
    output_file: str | Path,
    extra_args: list[str] | None = None,
) -> str:
    root = Path(__file__).resolve().parents[1]
    script = root / "r_engine" / script_name
    cmd = ["Rscript", str(script), str(input_csv), formula, str(output_file)]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "Rscript failed")
    return Path(output_file).read_text(encoding="utf-8", errors="replace")
