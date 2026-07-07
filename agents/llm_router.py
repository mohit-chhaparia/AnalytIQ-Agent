"""
Smart task router. Assigns each LLM task to the best available model based on API keys configured in .env.
"""

import os
from typing import Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

# Per-task model env vars
_CLAUDE_AGENT   = os.environ.get("CLAUDE_MODEL_AGENT",   "claude-opus-4-6")
_CLAUDE_CODE    = os.environ.get("CLAUDE_MODEL_CODE",    "claude-sonnet-4-6")
_CLAUDE_REWRITE = os.environ.get("CLAUDE_MODEL_REWRITE", "claude-haiku-4-5-20251001")

_GEMINI_VISION  = os.environ.get("GEMINI_MODEL_VISION",  "gemini-2.0-pro-exp")
_GEMINI_CONTEXT = os.environ.get("GEMINI_MODEL_CONTEXT", "gemini-1.5-pro")
_GEMINI_FAST    = os.environ.get("GEMINI_MODEL_FAST",    "gemini-2.0-flash")

_GROQ_MODEL     = os.environ.get("GROQ_MODEL",           "llama3-8b-8192")
_OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",          "llama3.2")
_LMS_MODEL      = os.environ.get("LM_STUDIO_MODEL",       "local-model")


def _has(key: str) -> bool:
    val = os.environ.get(key, "").strip()
    return bool(val and "your_" not in val and val != "")


AVAILABLE = {
    "claude":   _has("ANTHROPIC_API_KEY"),
    "gemini":   _has("GEMINI_API_KEY"),
    "groq":     _has("GROQ_API_KEY"),
    "ollama":   os.environ.get("LLM_BACKEND", "").lower() == "ollama",
    "lmstudio": os.environ.get("LLM_BACKEND", "").lower() == "lmstudio",
}


def any_llm_available() -> bool:
    return any(AVAILABLE.values())


# Task routing: (backend, model) in preference order
#
#   Task                   Best model          Why
#   ─────────────────────  ─────────────────── ────────────────────────────────
#   tool_orchestration     claude-opus         Best multi-step reasoning
#   analysis_planning      claude-opus         Nuanced statistical thinking
#   code_generation        claude-sonnet       Excellent code, faster + cheaper
#   diagnostic_notes       claude-sonnet       Careful interpretation
#   chart_interpretation   gemini-2.0-pro-exp  Only model with vision
#   long_context_review    gemini-1.5-pro      1M token window
#   plain_english_rewrite  groq / haiku        Simple task; save cost

TASK_ROUTES = {
    "tool_orchestration":    [("claude", _CLAUDE_AGENT),   ],
    "analysis_planning":     [("claude", _CLAUDE_AGENT),   ("gemini", _GEMINI_CONTEXT), ("groq", _GROQ_MODEL)],
    "code_generation":       [("claude", _CLAUDE_CODE),    ("groq", _GROQ_MODEL),       ("ollama", _OLLAMA_MODEL)],
    "diagnostic_notes":      [("claude", _CLAUDE_CODE),    ("gemini", _GEMINI_FAST),    ("groq", _GROQ_MODEL)],
    "chart_interpretation":  [("gemini", _GEMINI_VISION),  ],
    "long_context_review":   [("gemini", _GEMINI_CONTEXT), ("claude", _CLAUDE_AGENT)],
    "plain_english_rewrite": [("groq",   _GROQ_MODEL),     ("claude", _CLAUDE_REWRITE), ("gemini", _GEMINI_FAST),
                               ("ollama", _OLLAMA_MODEL),  ("lmstudio", _LMS_MODEL)],
}


def best_for(task: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (backend, model_string) for the best available model for a task.
    Returns (None, None) if nothing is configured.
    """
    for backend, model in TASK_ROUTES.get(task, [("groq", _GROQ_MODEL)]):
        if AVAILABLE.get(backend):
            return backend, model
    return None, None


