"""
Tests for agents/eda_agent.py

Verifies that recommend_eda_plots returns well-structured, useful
recommendations for different outcome types.
"""

import pandas as pd
import numpy as np
import pytest

from agents.data_profiler import profile_dataframe
from agents.eda_agent import recommend_eda_plots


