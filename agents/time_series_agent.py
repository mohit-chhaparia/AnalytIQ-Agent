"""Time series: univariate characterization and simple ARIMA-style forecast (statsmodels)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
except ImportError:  # pragma: no cover
    ARIMA = None  # type: ignore[misc, assignment]


