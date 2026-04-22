"""
Chinese Fortune Telling Bench - A benchmark for evaluating LLMs on Chinese fortune telling tasks.
"""

__version__ = "1.0.0"

from .benchmark import FortuneTellingBenchmark
from .models.base import ModelClient

__all__ = ["FortuneTellingBenchmark", "ModelClient"]