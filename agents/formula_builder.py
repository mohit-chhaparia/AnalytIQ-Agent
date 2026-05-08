"""Build statsmodels patsy formulas with C() for categorical columns."""

from __future__ import annotations


def _term(name: str, profile: dict) -> str:
    col_map = {c["name"]: c for c in profile.get("columns", [])}
    inf = col_map.get(name, {}).get("inferred_type", "")
    if inf in ("categorical", "binary", "numeric_discrete_or_categorical"):
        return f"C({name})"
    return name


def build_formula(outcome: str, predictors: list[str], profile: dict) -> str:
    """Standard additive linear / GLM formula."""
    formula_terms = [_term(p, profile) for p in predictors]
    if not formula_terms:
        return outcome + " ~ 1"
    return outcome + " ~ " + " + ".join(formula_terms)



