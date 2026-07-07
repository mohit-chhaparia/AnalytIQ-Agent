"""
Multi-backend LLM dispatcher. Uses llm_router to pick the right model + backend for each task type.
"""

import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
OLLAMA_URL        = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "llama3.2")
LM_STUDIO_URL     = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL   = os.environ.get("LM_STUDIO_MODEL", "local-model")


# backend callers

def _call_claude(prompt: str, system: str, model: str, max_tokens: int = 1024) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        kwargs = {"model": model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text.strip()
    except ImportError:
        return "[anthropic not installed — pip install anthropic]"
    except Exception as e:
        return f"[Claude error: {e}]"


def _call_gemini(prompt: str, system: str, model: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        full = f"{system}\n\n{prompt}" if system else prompt
        resp = genai.GenerativeModel(model).generate_content(full)
        return resp.text.strip()
    except ImportError:
        return "[google-generativeai not installed — pip install google-generativeai]"
    except Exception as e:
        return f"[Gemini error: {e}]"


def _call_groq(prompt: str, system: str, model: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=model, messages=msgs,
                                               temperature=0.3, max_tokens=1024)
        return resp.choices[0].message.content.strip()
    except ImportError:
        return "[openai not installed — pip install openai]"
    except Exception as e:
        return f"[Groq error: {e}]"


def _call_ollama(prompt: str, system: str, model: str) -> str:
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": prompt})
        return client.chat(model=model, messages=msgs)["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama error: {e}]"


def _call_lmstudio(prompt: str, system: str, model: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=model, messages=msgs,
                                               temperature=0.3, max_tokens=1024)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LM Studio error: {e}]"


def _dispatch(task: str, prompt: str, system: str = "") -> str:
    from agents.llm_router import best_for
    backend, model = best_for(task)
    if not backend:
        return ""
    if backend == "claude":
        return _call_claude(prompt, system, model)
    if backend == "gemini":
        return _call_gemini(prompt, system, model)
    if backend == "groq":
        return _call_groq(prompt, system, model)
    if backend == "ollama":
        return _call_ollama(prompt, system, model)
    if backend == "lmstudio":
        return _call_lmstudio(prompt, system, model)
    return ""


# public API

def rewrite_summary(raw_summary: str, context: str = "") -> str:
    if not raw_summary:
        return raw_summary
    ctx = f"\nContext: {context}" if context else ""
    prompt = (f"Rewrite this statistical summary in clear plain English. "
              f"Keep all numbers exact. Do not invent results. 3-6 sentences.{ctx}\n\n"
              f"Summary: {raw_summary}\n\nPlain-English version:")
    system = ("You are a statistical communication assistant. "
              "Rephrase statistical summaries accurately. Never invent numbers.")
    result = _dispatch("plain_english_rewrite", prompt, system)
    return result if result and not result.startswith("[") else raw_summary


def generate_analysis_plan(user_goal, profile_summary, outcome_type, available_tools) -> dict:
    prompt = (f"User goal: {user_goal}\nOutcome type: {outcome_type}\n"
              f"Profile: {profile_summary}\nAvailable tools: {', '.join(available_tools)}\n\n"
              f"The requested analysis is NOT in the tool registry. Plan it.\n"
              f'Respond ONLY with valid JSON: {{"engine":"python"|"r","analysis_type":"...",'
              f'"steps":[...],"rationale":"...","code_prompt":"..."}}')
    system = "You are a statistical analysis planner. Return only valid JSON."
    raw = _dispatch("analysis_planning", prompt, system)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"engine": "python", "analysis_type": "unknown", "steps": [],
            "rationale": raw, "code_prompt": ""}


def generate_analysis_code(code_prompt: str, engine: str) -> str:
    system = (
        "Generate only complete, executable code. "
        "Read from /input/dataset.csv. Write results to /output/results.json. "
        "Write plots to /output/plots/. No subprocess, os.remove, eval, or internet. "
        "Random seed 42. Return ONLY the code."
    )
    return _dispatch("code_generation", code_prompt, system)


def interpret_diagnostics_with_llm(diagnostic_notes: list, model_type: str, context: str = "") -> str:
    if not diagnostic_notes:
        return ""
    notes_text = "\n".join(f"- {n}" for n in diagnostic_notes)
    prompt = (f"A {model_type} model has these diagnostic findings:\n{notes_text}\n"
              f"{f'Context: {context}' if context else ''}\n\n"
              f"Give a concise expert interpretation: what each issue means practically "
              f"and what to do next. 3-5 sentences.")
    system = "You are an expert statistician providing diagnostic interpretation."
    return _dispatch("diagnostic_notes", prompt, system)


def review_full_report_with_gemini(report_text: str, profile_json: str) -> str:
    """
    Use Gemini 1.5 Pro's 1M context window to review the complete analysis
    report alongside the full dataset profile. Returns an expert critique
    and suggestions for improvement.

    This is the one task where Gemini's long context is genuinely better
    than any other model in the stack.
    """
    from agents.llm_router import best_for, AVAILABLE
    if not AVAILABLE.get("gemini"):
        return ""

    # Route explicitly to long-context Gemini
    backend, model = best_for("long_context_review")
    if backend != "gemini":
        return ""

    prompt = (
        f"You are reviewing a complete statistical analysis report.\n\n"
        f"DATASET PROFILE:\n{profile_json[:50000]}\n\n"
        f"ANALYSIS REPORT:\n{report_text[:50000]}\n\n"
        f"Review this analysis for:\n"
        f"1. Statistical soundness — are the right models used for the data types?\n"
        f"2. Missing analyses — what important analyses were not performed?\n"
        f"3. Diagnostic gaps — what assumption checks are missing?\n"
        f"4. Interpretation accuracy — are conclusions supported by the results?\n"
        f"5. Suggested next steps for the analyst.\n\n"
        f"Be specific and constructive. Reference actual column names and metrics where relevant."
    )
    return _call_gemini(prompt, "You are an expert statistical reviewer.", model)
