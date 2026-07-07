import os
import pytest


def test_best_for_returns_tuple():
    from agents.llm_router import best_for
    result = best_for("plain_english_rewrite")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_best_for_unknown_task_returns_none_when_no_keys(monkeypatch):
    # Patch all keys to empty
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY",    "")
    monkeypatch.setenv("GROQ_API_KEY",      "")
    monkeypatch.setenv("LLM_BACKEND",       "none")

    # Re-import to pick up patched env
    import importlib
    import agents.llm_router as router
    importlib.reload(router)

    backend, model = router.best_for("plain_english_rewrite")
    assert backend is None
    assert model   is None


def test_availability_report_has_all_backends():
    from agents.llm_router import availability_report
    report = availability_report()
    for key in ("claude", "gemini", "groq", "ollama", "lmstudio"):
        assert key in report


def test_best_model_for_backward_compat():
    from agents.llm_router import best_model_for
    result = best_model_for("plain_english_rewrite")
    # Returns either a backend name string or None
    assert result is None or isinstance(result, str)


def test_chart_interpretation_only_gemini():
    from agents.llm_router import TASK_ROUTES
    backends = [b for b, _ in TASK_ROUTES.get("chart_interpretation", [])]
    assert backends == ["gemini"], "Chart interpretation must only route to Gemini"


def test_tool_orchestration_only_claude():
    from agents.llm_router import TASK_ROUTES
    backends = [b for b, _ in TASK_ROUTES.get("tool_orchestration", [])]
    assert backends == ["claude"], "Tool orchestration must only route to Claude"
