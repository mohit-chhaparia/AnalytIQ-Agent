"""
Smart task router. Assigns each LLM task to the best available model based on API keys configured in .env.
"""

import os
from typing import Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

