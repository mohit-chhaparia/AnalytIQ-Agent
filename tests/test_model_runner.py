"""
Tests for agents/model_runner.py

Covers the three deterministic Python engines:
  run_linear_regression, run_logistic_regression, run_poisson_regression
"""

import numpy as np
import pandas as pd
import pytest

from agents.model_runner import (
    run_linear_regression,
    run_logistic_regression,
    run_poisson_regression,
)


