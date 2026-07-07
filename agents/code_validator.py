"""
Safety validator for dynamically generated analysis code.
Blocks dangerous patterns before any code is executed.
"""

import re

FORBIDDEN_PATTERNS = [
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "shutil.rmtree",
    "shutil.move",
    "subprocess",
    "socket",
    "requests.",
    "urllib.",
    "__import__",
    "eval(",
    "exec(",
    "open(",          # raw file access outside /input and /output is blocked
    "import sys",
    "sys.exit",
]

# Allow reading from /input and writing to /output — these safe patterns
# override the 'open(' block above when inspected in context
ALLOWED_FILE_PATTERNS = [
    r'open\(["\']\/input\/',
    r'open\(["\']\/output\/',
    r'pd\.read_csv\(["\']\/input\/',
    r'\.to_csv\(["\']\/output\/',
    r'savefig\(["\']\/output\/',
]


def validate_generated_code(code: str) -> dict:
    """
    Check generated code for forbidden patterns.

    Returns
    -------
    {
        "is_safe": bool,
        "problems": list[str],
        "warnings": list[str],
    }
    """
    problems = []
    warnings = []

    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            # Check if it's an allowed safe file pattern
            if pattern == "open(":
                safe = any(re.search(ap, code) for ap in ALLOWED_FILE_PATTERNS)
                raw_opens = re.findall(r'open\([^)]+\)', code)
                unsafe_opens = [
                    o for o in raw_opens
                    if not any(safe_kw in o for safe_kw in ["/input/", "/output/"])
                ]
                if unsafe_opens:
                    problems.append(f"Unsafe file access: {unsafe_opens}")
            else:
                problems.append(f"Forbidden pattern found: '{pattern}'")

    # Warn on heavy imports that might time out
    heavy_imports = ["tensorflow", "torch", "keras", "h2o", "autogluon"]
    for lib in heavy_imports:
        if f"import {lib}" in code or f"from {lib}" in code:
            warnings.append(f"Heavy library detected: '{lib}' — may take longer to execute.")

    # Check for output contract compliance
    if "/output/results.json" not in code:
        warnings.append("Code does not appear to write /output/results.json — output contract may not be met.")

    return {
        "is_safe": len(problems) == 0,
        "problems": problems,
        "warnings": warnings,
    }


def strip_markdown_fences(code: str) -> str:
    """Remove ```python / ``` wrappers that LLMs often add."""
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code.strip()
