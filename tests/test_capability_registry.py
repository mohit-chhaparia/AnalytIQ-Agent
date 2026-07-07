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


def test_known_unsupported_mixed():
    assert is_known_unsupported("fit a mixed effects model with lme4") is True


def test_list_capabilities_non_empty():
    caps = list_capabilities()
    assert len(caps) >= 4
    assert all(isinstance(c, str) for c in caps)


def test_all_registry_entries_have_required_keys():
    for name, details in CAPABILITY_REGISTRY.items():
        assert "supported" in details, f"{name} missing 'supported'"
        assert "engine"    in details, f"{name} missing 'engine'"
        assert "task_types" in details, f"{name} missing 'task_types'"
