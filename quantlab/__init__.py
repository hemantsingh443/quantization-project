"""
QuantLab - LLM Quantization Experimentation Platform

A modular platform for rapid experimentation with model quantization strategies.
"""

__version__ = "0.1.0"

from quantlab.core import QuantLab
from quantlab.config import QuantLabConfig, ExperimentConfig

__all__ = [
    "QuantLab",
    "QuantLabConfig", 
    "ExperimentConfig",
]
