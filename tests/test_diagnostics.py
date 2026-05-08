import pandas as pd

from agents.diagnostics_agent import interpret_poisson_diagnostics, run_diagnostics_for_result
from agents.model_runner import run_linear_regression, run_poisson_regression


def test_poisson_interpretation():
    notes = interpret_poisson_diagnostics({"dispersion": 2.0, "overdispersion_flag": True})
    assert any("overdispersion" in n.lower() for n in notes)



