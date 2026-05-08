"""Compare fitted models using information criteria and/or CV metrics."""

from __future__ import annotations


def compare_models(results: list[dict]) -> dict:
    """Supports GLM/OLS (AIC), ML (CV mean), and ARIMA (AIC)."""
    rows = []
    for r in results:
        row = {
            "model_type": r.get("model_type"),
            "formula": r.get("formula"),
            "aic": r.get("aic"),
            "bic": r.get("bic"),
            "cv_metric": r.get("cv_metric"),
            "cv_mean": r.get("cv_mean"),
        }
        if "metrics" in r:
            row["accuracy"] = r["metrics"].get("accuracy")
            row["auc"] = r["metrics"].get("auc")
            row["roc_auc_holdout"] = r["metrics"].get("roc_auc_holdout")
            row["rmse_holdout"] = r["metrics"].get("rmse_holdout")
            row["r2_holdout"] = r["metrics"].get("r2_holdout")
        if "r_squared" in r:
            row["r_squared"] = r.get("r_squared")
        rows.append(row)

    valid_aic = [x for x in rows if x.get("aic") is not None and str(x["aic"]) != "nan"]
    best_aic = min(valid_aic, key=lambda x: x["aic"]) if valid_aic else None

    valid_cv = [x for x in rows if x.get("cv_mean") is not None and str(x["cv_mean"]) != "nan"]
    best_cv = max(valid_cv, key=lambda x: x["cv_mean"]) if valid_cv else None

    return {
        "comparison_table": rows,
        "best_by_aic": best_aic,
        "best_by_cv_mean": best_cv,
    }
