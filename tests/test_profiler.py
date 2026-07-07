"""
Tests for agents/data_profiler.py

Covers: infer_variable_type, numeric_summary, profile_dataframe
(missing values, duplicates, outlier detection, type inference).
"""

import pandas as pd
import numpy as np
import pytest

from agents.data_profiler import infer_variable_type, numeric_summary, profile_dataframe


