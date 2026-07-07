"""
Tests for agents/model_selector.py

Verifies recommendations for binary, continuous, count, and
time-series outcomes, plus error handling.
"""

import pandas as pd
import pytest

from agents.data_profiler import profile_dataframe
from agents.model_selector import recommend_models


