"""
Compares multiple fitted model results and recommends the best one.
Works with results from model_runner.py and dynamic_analysis_agent.py.
"""

from typing import List, Optional


def compare_models(model_results: List[dict]) -> dict:
    """
    Rank fitted models by a composite score.

    Supports comparison of:
      - Linear models     : AIC, BIC, R², Adj R²
      - Logistic models   : AIC, BIC, AUC, accuracy
      - Poisson models    : AIC, BIC, deviance/df
      - Dynamic results   : whatever metrics are available

    Parameters
    ----------
    model_results : list of result dicts from model_runner / dynamic_analysis_agent

    Returns
    -------
    {
        "ranked":        list of dicts, best first
        "best_model":    str — name/type of best model
        "rationale":     str — plain-English explanation of winner selection
        "comparison_table": list of dicts for display in Streamlit
    }
    """
    if not model_results:
        return {"ranked": [], "best_model": "None", "rationale": "No models to compare.", "comparison_table": []}

    scored = []
    for result in model_results:
        score = _score_model(result)
        scored.append({**result, "_score": score})

    ranked = sorted(scored, key=lambda x: x["_score"], reverse=True)
    best = ranked[0]

    rationale = _build_rationale(best, ranked)

    table = [
        {
            "Model":       r.get("model_type", "Unknown"),
            "AIC":         r.get("aic", "—"),
            "BIC":         r.get("bic", "—"),
            "R²":          r.get("r_squared", "—"),
            "AUC":         r.get("metrics", {}).get("auc", "—") if "metrics" in r else "—",
            "Accuracy":    r.get("metrics", {}).get("accuracy", "—") if "metrics" in r else "—",
            "Dispersion":  r.get("dispersion", "—"),
            "Score":       round(r["_score"], 4),
        }
        for r in ranked
    ]

    return {
        "ranked":           ranked,
        "best_model":       best.get("model_type", "Unknown"),
        "best_result":      best,
        "rationale":        rationale,
        "comparison_table": table,
    }


def _score_model(result: dict) -> float:
    """
    Compute a composite rank score (higher = better).
    Normalised so all model types are roughly comparable.
    """
    score = 0.0
    model_type = result.get("model_type", "").lower()

    # AIC / BIC: lower is better — use negative, normalise to 0-1
    aic = result.get("aic")
    bic = result.get("bic")
    if aic is not None:
        score -= aic * 0.0001    # small weight; absolute values vary hugely
    if bic is not None:
        score -= bic * 0.0001

    # Classification metrics
    metrics = result.get("metrics", {})
    auc = metrics.get("auc")
    if auc is not None:
        score += auc * 10        # AUC 0–1, weight heavily

    acc = metrics.get("accuracy")
    if acc is not None:
        score += acc * 3

    # Regression metrics
    r2 = result.get("r_squared")
    if r2 is not None:
        score += r2 * 5

    adj_r2 = result.get("adj_r_squared")
    if adj_r2 is not None:
        score += adj_r2 * 2

    # Diagnostics penalty
    # Overdispersion (Poisson)
    disp = result.get("dispersion")
    if disp is not None and disp > 1.5:
        score -= 2.0

    # Flagged issues
    if result.get("overdispersion_flag"):
        score -= 1.5

    return score


def _build_rationale(best: dict, ranked: list) -> str:
    model_type = best.get("model_type", "Selected model")
    parts = [f"{model_type} was selected as the best-fitting model."]

    aic = best.get("aic")
    if aic:
        parts.append(f"It achieved the lowest AIC ({aic:.2f}).")

    metrics = best.get("metrics", {})
    auc = metrics.get("auc")
    if auc:
        parts.append(f"AUC = {auc:.3f}.")

    r2 = best.get("r_squared")
    if r2:
        parts.append(f"R² = {r2:.3f}.")

    if len(ranked) > 1:
        runner_up = ranked[1].get("model_type", "runner-up")
        parts.append(f"{runner_up} was the next best candidate.")

    return " ".join(parts)


def interpret_comparison(comparison: dict) -> list:
    """Return plain-English diagnostic notes for the model comparison."""
    notes = []
    best = comparison.get("best_model", "Unknown")
    notes.append(f"Best model selected: {best}.")
    notes.append(comparison.get("rationale", ""))
    table = comparison.get("comparison_table", [])
    if len(table) > 1:
        notes.append(
            f"{len(table)} models were compared. "
            "Review the comparison table for full rankings."
        )
    return [n for n in notes if n]
