"""Validate generated code before any sandbox execution (defense in depth)."""

from __future__ import annotations

FORBIDDEN_PATTERNS = [
    "os.remove",
    "shutil.rmtree",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "open(",
    "eval(",
    "exec(",
]


def validate_generated_code(code: str) -> dict:
    problems: list[str] = []
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            problems.append(f"Forbidden pattern found: {pattern}")
    return {"is_safe": len(problems) == 0, "problems": problems}
