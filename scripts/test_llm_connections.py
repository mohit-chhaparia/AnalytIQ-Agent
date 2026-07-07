#!/usr/bin/env python3
"""
scripts/test_llm_connections.py

Verifies that all three cloud LLM backends (Claude, Gemini, Groq) are
reachable and responding correctly with your current API keys.

Usage:
    # Set keys in .env or export them, then run:
    python scripts/test_llm_connections.py

    # Or pass keys directly:
    ANTHROPIC_API_KEY=sk-ant-... GEMINI_API_KEY=AIza... GROQ_API_KEY=gsk_... \
        python scripts/test_llm_connections.py

Exit codes:
    0  All configured backends passed
    1  One or more backends failed
"""

import os
import sys
import time

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")

PING_PROMPT = (
    "You are a health-check endpoint. "
    "Reply with exactly one word: OK"
)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _ok(label: str, detail: str, elapsed: float):
    print(f"  {GREEN}✓{RESET} {BOLD}{label}{RESET}  →  {detail}  ({elapsed:.2f}s)")


def _fail(label: str, error: str):
    print(f"  {RED}✗{RESET} {BOLD}{label}{RESET}  →  {RED}{error}{RESET}")


def _skip(label: str, reason: str):
    print(f"  {YELLOW}–{RESET} {BOLD}{label}{RESET}  →  {YELLOW}skipped: {reason}{RESET}")


# Claude

def test_claude() -> bool:
    if not ANTHROPIC_API_KEY:
        _skip("Claude (Anthropic)", "ANTHROPIC_API_KEY not set")
        return True  # Not a failure — key simply not configured

    try:
        import anthropic
    except ImportError:
        _fail("Claude (Anthropic)", "anthropic package not installed — pip install anthropic")
        return False

    try:
        model = os.environ.get("CLAUDE_MODEL_AGENT", "claude-haiku-4-5-20251001")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        t0 = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": PING_PROMPT}],
        )
        elapsed = time.time() - t0

        text = response.content[0].text.strip()
        tokens_in  = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        _ok(
            "Claude (Anthropic)",
            f'"{text}" via {model}  '
            f"[in={tokens_in} out={tokens_out}]",
            elapsed,
        )
        return True

    except anthropic.AuthenticationError:
        _fail("Claude (Anthropic)", "Invalid API key — check ANTHROPIC_API_KEY")
        return False
    except anthropic.RateLimitError:
        _fail("Claude (Anthropic)", "Rate limit hit — try again shortly")
        return False
    except Exception as e:
        _fail("Claude (Anthropic)", str(e))
        return False


# Gemini

def test_gemini() -> bool:
    if not GEMINI_API_KEY:
        _skip("Gemini (Google)", "GEMINI_API_KEY not set")
        return True

    try:
        import google.generativeai as genai
    except ImportError:
        _fail("Gemini (Google)", "google-generativeai not installed — pip install google-generativeai")
        return False

    try:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)

        t0 = time.time()
        response = model.generate_content(PING_PROMPT)
        elapsed = time.time() - t0

        text = response.text.strip()
        _ok("Gemini (Google)", f'"{text}" via {model_name}', elapsed)
        return True

    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "PERMISSION_DENIED" in err:
            _fail("Gemini (Google)", "Invalid API key — check GEMINI_API_KEY")
        elif "quota" in err.lower() or "429" in err:
            _fail("Gemini (Google)", "Quota exceeded — check billing / rate limits")
        else:
            _fail("Gemini (Google)", err)
        return False


# Groq

def test_groq() -> bool:
    if not GROQ_API_KEY:
        _skip("Groq", "GROQ_API_KEY not set")
        return True

    try:
        from openai import OpenAI
    except ImportError:
        _fail("Groq", "openai package not installed — pip install openai")
        return False

    try:
        model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

        t0 = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PING_PROMPT}],
            max_tokens=10,
            temperature=0.0,
        )
        elapsed = time.time() - t0

        text = response.choices[0].message.content.strip()
        tokens_in  = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        _ok(
            "Groq",
            f'"{text}" via {model_name}  '
            f"[in={tokens_in} out={tokens_out}]",
            elapsed,
        )
        return True

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            _fail("Groq", "Invalid API key — check GROQ_API_KEY")
        elif "429" in err or "rate_limit" in err.lower():
            _fail("Groq", "Rate limit hit — try again shortly")
        else:
            _fail("Groq", err)
        return False


# LLM Router smoke test

def test_llm_router() -> bool:
    """Verify the router's availability_report() returns without errors."""
    try:
        from agents.llm_router import availability_report, best_for
        report = availability_report()
        print(f"\n  {BOLD}LLM Router availability:{RESET}")
        for backend, status in report.items():
            icon = GREEN + "✓" + RESET if "available" in status.lower() else YELLOW + "–" + RESET
            print(f"    {icon}  {backend}: {status}")

        # Verify routing assignments match expectations
        EXPECTED_ROUTES = {
            "tool_orchestration":   "claude",
            "plain_english":        "groq",
            "visual_interpretation":"gemini",
        }
        routing_ok = True
        print(f"\n  {BOLD}Routing assignments:{RESET}")
        for task, expected in EXPECTED_ROUTES.items():
            backend, model = best_for(task)
            if backend is None:
                print(f"    {YELLOW}–{RESET}  {task}: no backend available (key missing)")
            elif backend == expected:
                print(f"    {GREEN}✓{RESET}  {task} → {backend} ({model})")
            else:
                print(f"    {YELLOW}!{RESET}  {task} → {backend} ({model})  "
                      f"[expected {expected}]")
                routing_ok = False
        return routing_ok

    except Exception as e:
        _fail("LLM Router", str(e))
        return False


# Main

def main():
    print(f"\n{BOLD}AnalytIQ Agent — LLM Connection Test{RESET}")
    print("=" * 50)

    # Show which keys are present
    print(f"\n{BOLD}API key status:{RESET}")
    for name, val in [
        ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        ("GEMINI_API_KEY",    GEMINI_API_KEY),
        ("GROQ_API_KEY",      GROQ_API_KEY),
    ]:
        masked = (val[:8] + "..." + val[-4:]) if len(val) > 12 else ("(not set)" if not val else "(set)")
        icon   = GREEN + "✓" + RESET if val else YELLOW + "–" + RESET
        print(f"  {icon}  {name}: {masked}")

    print(f"\n{BOLD}Running connection tests:{RESET}")
    results = {
        "Claude":     test_claude(),
        "Gemini":     test_gemini(),
        "Groq":       test_groq(),
    }

    test_llm_router()

    # Summary
    passed  = [k for k, v in results.items() if v]
    failed  = [k for k, v in results.items() if not v]

    print(f"\n{'=' * 50}")
    if failed:
        print(f"{RED}{BOLD}FAILED:{RESET} {', '.join(failed)}")
        print(f"{GREEN}Passed:{RESET} {', '.join(passed) or 'none'}")
        sys.exit(1)
    else:
        configured = [k for k, v in [
            ("Claude", ANTHROPIC_API_KEY),
            ("Gemini", GEMINI_API_KEY),
            ("Groq",   GROQ_API_KEY),
        ] if v]
        if configured:
            print(f"{GREEN}{BOLD}All configured backends passed ✓{RESET}")
            print(f"Tested: {', '.join(configured)}")
        else:
            print(f"{YELLOW}No API keys set — set at least one of ANTHROPIC_API_KEY, "
                  f"GEMINI_API_KEY, GROQ_API_KEY{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
