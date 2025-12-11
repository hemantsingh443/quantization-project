"""
Quantization strategies and utilities.
"""

# New clean abstraction (recommended)
from quantlab.quantization.strategies import (
    QuantizationStrategy,
    QuantizationResult,
    NoQuantizationStrategy,
    FP16Strategy,
    BF16Strategy,
    INT8Strategy,
    INT4Strategy,
    NF4Strategy,
    GPTQStrategy,
    AWQStrategy,
    get_strategy,
    register_strategy,
    list_strategies,
)

# Legacy exports (for backward compatibility)
from quantlab.quantization.base import BaseQuantizer
from quantlab.quantization.precision import FP16Quantizer, BF16Quantizer
from quantlab.quantization.integer import INT8Quantizer, INT4Quantizer
from quantlab.quantization.advanced import NF4Quantizer

# Legacy registry (for backward compatibility)
_QUANTIZER_REGISTRY = {
    "none": None,
    "fp32": None,
    "fp16": FP16Quantizer,
    "bf16": BF16Quantizer,
    "int8": INT8Quantizer,
    "int4": INT4Quantizer,
    "nf4": NF4Quantizer,
}


def get_quantizer(method: str):
    """Get a legacy quantizer by method name (deprecated, use get_strategy instead)."""
    if method not in _QUANTIZER_REGISTRY:
        raise ValueError(f"Unknown quantization method: {method}. Available: {list(_QUANTIZER_REGISTRY.keys())}")
    
    quantizer_class = _QUANTIZER_REGISTRY[method]
    if quantizer_class is None:
        return None
    
    return quantizer_class()


def register_quantizer(name: str, quantizer_class: type):
    """Register a custom quantizer (deprecated, use register_strategy instead)."""
    _QUANTIZER_REGISTRY[name] = quantizer_class


def list_quantizers() -> list:
    """List available quantization methods."""
    return list(_QUANTIZER_REGISTRY.keys())


__all__ = [
    # New API
    "QuantizationStrategy",
    "QuantizationResult", 
    "NoQuantizationStrategy",
    "FP16Strategy",
    "BF16Strategy",
    "INT8Strategy",
    "INT4Strategy",
    "NF4Strategy",
    "GPTQStrategy",
    "AWQStrategy",
    "get_strategy",
    "register_strategy",
    "list_strategies",
    # Legacy API
    "BaseQuantizer",
    "FP16Quantizer",
    "BF16Quantizer",
    "INT8Quantizer",
    "INT4Quantizer",
    "NF4Quantizer",
    "get_quantizer",
    "register_quantizer",
    "list_quantizers",
]
