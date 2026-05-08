"""Templated plain-English summaries from model output and diagnostic notes."""

from __future__ import annotations


def generate_plain_english_summary(model_result: dict, diagnostics: dict | list | None) -> str:
    model_type = model_result.get("model_type", "model")
    formula = model_result.get("formula", "")

    if model_type.startswith("RandomForest"):
        return _summary_ml(model_result, diagnostics)
    if "ARIMA" in model_type or model_type.startswith("Time series"):
        return _summary_ts(model_result, diagnostics)

    text = f"""
The selected model was a {model_type}. The model was fit using the formula:
{formula or '(n/a)'}

The analysis evaluates the relationship between the selected predictors and the outcome.
Model diagnostics were reviewed to assess whether the fitted model was appropriate for the data.
""".strip()

    if "metrics" in model_result:
        metrics = model_result["metrics"]
        text += f"""

For classification performance at a 0.5 probability threshold, the model achieved an accuracy of {metrics.get("accuracy", 0):.3f},
sensitivity/recall of {metrics.get("sensitivity_recall", 0):.3f}, precision of {metrics.get("precision", 0):.3f},
and AUC of {metrics.get("auc", float("nan")):.3f}.
"""

    if "r_squared" in model_result:
        text += f"\nThe model R-squared is {model_result.get('r_squared', 0):.4f} (adjusted {model_result.get('adj_r_squared', 0):.4f}).\n"

    if "dispersion" in model_result and model_result.get("dispersion") is not None:
        text += f"\nPoisson dispersion (deviance / df residual) is approximately {model_result['dispersion']:.4f}.\n"

    text += _diagnostic_notes_paragraph(diagnostics)
    return text.strip()


def _summary_ml(model_result: dict, diagnostics: dict | list | None) -> str:
    task = model_result.get("task", "")
    m = model_result.get("metrics", {})
    lines = [
        f"The pipeline fits a **{model_result.get('model_type')}** for **{task}** prediction.",
        f"Cross-validated {model_result.get('cv_metric')} mean: {model_result.get('cv_mean', 0):.4f} (sd {model_result.get('cv_std', 0):.4f}).",
    ]
    if task == "classify":
        lines.append(
            f"Hold-out accuracy {m.get('accuracy', 0):.3f}; ROC-AUC {m.get('roc_auc_holdout', float('nan')):.3f}."
        )
    else:
        lines.append(
            f"Hold-out RMSE {m.get('rmse_holdout', 0):.4f}; R² {m.get('r2_holdout', 0):.4f}."
        )
    imps = model_result.get("top_feature_importances") or []
    if imps:
        top = ", ".join(f"{d['feature']}" for d in imps[:5])
        lines.append(f"Largest tree-based importances (post encoding) include: {top}.")
    body = "\n".join(lines)
    return (body + "\n" + _diagnostic_notes_paragraph(diagnostics)).strip()


def _summary_ts(model_result: dict, diagnostics: dict | list | None) -> str:
    if model_result.get("model_type") == "Time series characterization":
        lines = [
            "Univariate series summary:",
            f"ADF test p-value ≈ {model_result.get('adf_pvalue', 0):.4f} ({model_result.get('stationarity_hint', '')}).",
            "Use ACF/PACF for seasonal structure and candidate ARIMA orders.",
        ]
    else:
        lines = [
            f"Fitted **{model_result.get('model_type')}** with AIC {model_result.get('aic', 0):.2f}.",
            f"Short-horizon forecast means: {model_result.get('forecast_mean', [])}.",
        ]
    body = "\n".join(lines)
    return (body + "\n" + _diagnostic_notes_paragraph(diagnostics)).strip()


def _diagnostic_notes_paragraph(diagnostics: dict | list | None) -> str:
    notes: list[str] = []
    if isinstance(diagnostics, list):
        notes.extend(str(n) for n in diagnostics)
    elif isinstance(diagnostics, dict):
        notes.extend(str(n) for n in diagnostics.get("notes", []))
        if "poisson" in diagnostics:
            notes.extend(str(n) for n in diagnostics["poisson"])
    if not notes:
        return ""
    out = "\nKey diagnostic findings:\n"
    for note in notes:
        out += f"- {note}\n"
    return out
