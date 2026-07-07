"""
Uses Gemini's multimodal vision to interpret diagnostic plots.

After the model runner saves plots (residuals, ROC, Q-Q, etc.) as PNG files, send them here and Gemini returns structured findings: what it sees, what issues it flags, and what the analyst should do next.

Setup:
  GEMINI_API_KEY → https://aistudio.google.com/apikey (free, no card)
  GEMINI_MODEL_VISION → gemini-2.0-pro-exp recommended with Gemini Advanced
"""

import os
import json
import re
from pathlib import Path
from typing import Union, Optional, List

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL_VISION",
                  os.environ.get("GEMINI_MODEL", "gemini-2.0-pro-exp"))

PLOT_PROMPTS = {
    "residual_plot": (
        "This is a residual plot from a {model_type} model. "
        "Examine it for: (1) non-random patterns suggesting non-linearity, "
        "(2) funnel shapes indicating heteroscedasticity, "
        "(3) extreme outliers. Be specific about what you observe."
    ),
    "qq_plot": (
        "This is a Q-Q plot of residuals from a {model_type} model. "
        "Assess whether points follow the diagonal. "
        "Note heavy tails, light tails, or S-curves indicating non-normality."
    ),
    "roc_curve": (
        "This is a ROC curve for a {model_type} classifier. "
        "Estimate the AUC from the curve shape and comment on discriminative ability."
    ),
    "histogram": (
        "This is a distribution histogram. "
        "Describe the shape: normal, skewed, bimodal, or heavy-tailed? "
        "Note any gaps or unexpected spikes."
    ),
    "correlation_heatmap": (
        "This is a correlation heatmap. "
        "Identify variable pairs with correlation above 0.8 or below -0.8 "
        "that could indicate multicollinearity. List the specific pairs."
    ),
    "calibration_curve": (
        "This is a calibration curve for a {model_type} classifier. "
        "Assess how well predicted probabilities match observed proportions."
    ),
    "class_balance": (
        "This is a bar chart showing class distribution. "
        "Assess class imbalance. Flag if one class is below 20%."
    ),
}

RESPONSE_SCHEMA = (
    'Respond ONLY with valid JSON (no markdown fences):\n'
    '{"interpretation":"2-4 sentence description","flags":["issue1","issue2"],'
    '"recommendations":["action1","action2"]}'
)


def interpret_plot(
    image_path: Union[str, Path],
    plot_type: str,
    model_type: str = "",
    context: str = "",
) -> dict:
    """
    Send a diagnostic plot to Gemini for visual interpretation.

    Parameters
    ----------
    image_path : path to a PNG/JPEG file
    plot_type  : one of the PLOT_PROMPTS keys, or any string
    model_type : e.g. "Logistic Regression"
    context    : e.g. outcome variable, formula

    Returns
    -------
    dict with keys: interpretation, flags, recommendations, model, status
    """
    if not GEMINI_API_KEY:
        return {
            "interpretation": "", "flags": [], "recommendations": [],
            "model": GEMINI_MODEL, "status": "skipped",
            "reason": "GEMINI_API_KEY not set in .env",
        }

    image_path = Path(image_path)
    if not image_path.exists():
        return {"status": "error", "reason": f"Image not found: {image_path}",
                "interpretation": "", "flags": [], "recommendations": []}

    base = PLOT_PROMPTS.get(
        plot_type,
        "This is a statistical diagnostic plot from a {model_type} model. "
        "Describe what you see and flag any statistical concerns."
    ).format(model_type=model_type or "statistical")

    ctx_block = f"\nAdditional context: {context}" if context else ""
    prompt = f"{base}{ctx_block}\n\n{RESPONSE_SCHEMA}"

    try:
        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=GEMINI_API_KEY)
        model  = genai.GenerativeModel(GEMINI_MODEL)
        img    = Image.open(image_path)
        resp   = model.generate_content([prompt, img])
        raw    = resp.text.strip()

        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$",     "", raw)
        parsed = json.loads(raw)

        return {**parsed, "model": GEMINI_MODEL, "status": "success"}

    except ImportError as e:
        return {"status": "error", "reason": f"Missing package: {e}. Run: pip install google-generativeai pillow",
                "interpretation": "", "flags": [], "recommendations": []}
    except json.JSONDecodeError:
        return {"status": "success", "interpretation": raw, "flags": [], "recommendations": [],
                "model": GEMINI_MODEL}
    except Exception as e:
        return {"status": "error", "reason": str(e),
                "interpretation": "", "flags": [], "recommendations": []}


def interpret_all_plots(
    plot_dir: Union[str, Path],
    model_type: str = "",
    context: str = "",
) -> List[dict]:
    """
    Interpret every PNG in a directory. Returns list of interpretation dicts.
    Called after model_runner.save_diagnostic_plots().
    """
    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        return []

    FILENAME_TO_TYPE = {
        "residual": "residual_plot",
        "qq":       "qq_plot",
        "roc":      "roc_curve",
        "hist":     "histogram",
        "corr":     "correlation_heatmap",
        "calibr":   "calibration_curve",
        "balance":  "class_balance",
        "class":    "class_balance",
    }

    results = []
    for img_path in sorted(plot_dir.glob("*.png")):
        stem  = img_path.stem.lower()
        ptype = next((v for k, v in FILENAME_TO_TYPE.items() if k in stem), "residual_plot")
        result = interpret_plot(img_path, ptype, model_type, context)
        result["plot_file"] = img_path.name
        results.append(result)

    return results
