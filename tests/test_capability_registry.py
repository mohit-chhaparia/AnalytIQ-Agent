from agents.capability_registry import (
    find_capability, is_known_unsupported, list_capabilities, CAPABILITY_REGISTRY
)


def test_find_logistic():
    name, details = find_capability("predict churn using logistic regression")
    assert name == "logistic_regression"
    assert details["supported"] is True


def test_find_linear():
    name, details = find_capability("run OLS regression on the outcome")
    assert name == "linear_regression"


def test_find_poisson():
    name, details = find_capability("count outcome, use poisson model")
    assert name == "poisson_regression"


def test_find_anova():
    name, details = find_capability("factorial ANOVA with covariates")
    assert name == "anova_ancova"


def test_returns_none_for_unknown():
    name, details = find_capability("train a transformer model on embeddings")
    assert name is None
    assert details is None


def test_known_unsupported_survival():
    assert is_known_unsupported("survival analysis for time to churn") is True


