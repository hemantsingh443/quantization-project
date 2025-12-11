"""
Base quantizer interface and utilities.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from torch import nn


logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Result of a quantization operation."""
    success: bool
    method: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    layer_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    @property
    def size_reduction_pct(self) -> float:
        """Percentage of size reduction."""
        if self.original_size_mb == 0:
            return 0.0
        return (1 - self.quantized_size_mb / self.original_size_mb) * 100


class BaseQuantizer(ABC):
    """
    Base class for all quantizers.
    
    Provides a common interface for quantizing models with different strategies.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply quantization to a model.
        
        Args:
            model: The model to quantize
            config: Quantization configuration
            
        Returns:
            Quantized model
        """
        pass
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: Any,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Calibrate quantization parameters using representative data.
        
        Override in subclasses that require calibration.
        
        Args:
            model: The model to calibrate
            calibration_data: Representative data for calibration
            config: Calibration configuration
            
        Returns:
            Calibrated model
        """
        logger.info(f"{self.name} does not require calibration")
        return model
    
    def get_layer_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get statistics about quantized layers."""
        stats = {
            "total_params": 0,
            "quantized_params": 0,
            "layers": {}
        }
        
        for name, module in model.named_modules():
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            if param_count > 0:
                stats["total_params"] += param_count
                dtype = next(module.parameters()).dtype if list(module.parameters()) else None
                stats["layers"][name] = {
                    "params": param_count,
                    "dtype": str(dtype),
                }
        
        return stats
    
    def estimate_memory(self, model: nn.Module) -> float:
        """Estimate model memory in MB."""
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes / (1024 * 1024)
    
    def validate(self, model: nn.Module) -> List[str]:
        """
        Validate a quantized model for common issues.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for NaN/Inf in parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                errors.append(f"NaN detected in {name}")
            if torch.isinf(param).any():
                errors.append(f"Inf detected in {name}")
        
        return errors


class IdentityQuantizer(BaseQuantizer):
    """No-op quantizer that returns the model unchanged."""
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """Return model unchanged."""
        logger.info("Identity quantizer: no quantization applied")
        return model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 * 1024)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters by dtype."""
    counts = {}
    for param in model.parameters():
        dtype = str(param.dtype)
        if dtype not in counts:
            counts[dtype] = 0
        counts[dtype] += param.numel()
    return counts


def get_quantizable_layers(model: nn.Module) -> List[str]:
    """Get list of layer names that can be quantized."""
    quantizable = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
            quantizable.append(name)
    return quantizable
