from agents.capability_registry import (
    find_capability, is_known_unsupported, list_capabilities, CAPABILITY_REGISTRY
)


def test_find_logistic():
    name, details = find_capability("predict churn using logistic regression")
    assert name == "logistic_regression"
    assert details["supported"] is True


