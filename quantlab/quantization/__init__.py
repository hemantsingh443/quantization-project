"""
Quantization strategies and utilities.
"""

from quantlab.quantization.base import BaseQuantizer, QuantizationResult
from quantlab.quantization.precision import FP16Quantizer, BF16Quantizer
from quantlab.quantization.integer import INT8Quantizer, INT4Quantizer
from quantlab.quantization.advanced import NF4Quantizer

# Registry of available quantizers
_QUANTIZER_REGISTRY = {
    "none": None,
    "fp32": None,
    "fp16": FP16Quantizer,
    "bf16": BF16Quantizer,
    "int8": INT8Quantizer,
    "int4": INT4Quantizer,
    "nf4": NF4Quantizer,
}


def get_quantizer(method: str) -> BaseQuantizer:
    """Get a quantizer by method name."""
    if method not in _QUANTIZER_REGISTRY:
        raise ValueError(f"Unknown quantization method: {method}. Available: {list(_QUANTIZER_REGISTRY.keys())}")
    
    quantizer_class = _QUANTIZER_REGISTRY[method]
    if quantizer_class is None:
        return None
    
    return quantizer_class()


def register_quantizer(name: str, quantizer_class: type):
    """Register a custom quantizer."""
    _QUANTIZER_REGISTRY[name] = quantizer_class


def list_quantizers() -> list:
    """List available quantization methods."""
    return list(_QUANTIZER_REGISTRY.keys())


__all__ = [
    "BaseQuantizer",
    "QuantizationResult",
    "FP16Quantizer",
    "BF16Quantizer",
    "INT8Quantizer",
    "INT4Quantizer",
    "NF4Quantizer",
    "get_quantizer",
    "register_quantizer",
    "list_quantizers",
]
