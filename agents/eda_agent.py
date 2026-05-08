"""Rule-based EDA plot recommendations from profile and outcome."""

from __future__ import annotations


def recommend_eda_plots(profile: dict, outcome: str) -> list[dict]:
    columns = profile.get("columns", [])
    outcome_info = next((c for c in columns if c["name"] == outcome), None)
    if outcome_info is None:
        return []
    plots: list[dict] = []
    outcome_type = outcome_info["inferred_type"]

    for col in columns:
        if col["name"] == outcome:
            continue
        predictor_type = col["inferred_type"]

        if outcome_type == "binary" and predictor_type in (
            "categorical",
            "binary",
            "numeric_discrete_or_categorical",
        ):
            plots.append(
                {
                    "plot": "Grouped bar chart",
                    "x": col["name"],
                    "y": outcome,
                    "purpose": "Compare outcome proportions across categories.",
                }
            )
        elif outcome_type == "binary" and predictor_type == "continuous_numeric":
            plots.append(
                {
                    "plot": "Boxplot or density plot",
                    "x": outcome,
                    "y": col["name"],
                    "purpose": "Compare numeric predictor distribution by outcome groups.",
                }
            )
        elif outcome_type == "continuous_numeric" and predictor_type == "continuous_numeric":
            plots.append(
                {
                    "plot": "Scatterplot",
                    "x": col["name"],
                    "y": outcome,
                    "purpose": "Check linear relationship and potential outliers.",
                }
            )
        elif outcome_type == "continuous_numeric" and predictor_type in (
            "categorical",
            "binary",
            "numeric_discrete_or_categorical",
        ):
            plots.append(
                {
                    "plot": "Boxplot",
                    "x": col["name"],
                    "y": outcome,
                    "purpose": "Compare response distribution across groups.",
                }
            )
        elif outcome_type == "numeric_discrete_or_categorical" and predictor_type in (
            "categorical",
            "binary",
        ):
            plots.append(
                {
                    "plot": "Bar plot / mean count",
                    "x": col["name"],
                    "y": outcome,
                    "purpose": "Summarize count or discrete outcome by category.",
                }
            )

    return plots
