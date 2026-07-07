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


