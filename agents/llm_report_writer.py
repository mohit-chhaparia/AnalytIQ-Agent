"""Optional local LLM rewriting via OpenAI-compatible API (LM Studio) or Ollama."""

from __future__ import annotations

import os
from typing import Any


def rewrite_summary_with_openai_compatible(
    raw_summary: str,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str = "local-model",
) -> str:
    """LM Studio default: http://localhost:1234/v1"""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai package for LM Studio compatibility.") from exc

    client = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "lm-studio"),
    )
    prompt = f"""Rewrite the following statistical summary in clear plain English for a business audience.
Do not invent numbers or contradict the text. Keep all numeric claims exactly as stated.

Summary:
{raw_summary}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


