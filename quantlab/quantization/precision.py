"""
Precision quantizers (FP16, BF16).
"""

import logging
from typing import Dict, Any

import torch
from torch import nn

from quantlab.quantization.base import BaseQuantizer


logger = logging.getLogger(__name__)


class FP16Quantizer(BaseQuantizer):
    """
    Convert model to FP16 (half precision).
    
    Simple but effective - reduces memory by ~50% with minimal accuracy loss
    for most models. Well-supported on modern GPUs.
    """
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Convert model to FP16.
        
        Args:
            model: Model to convert
            config: Configuration (unused for FP16)
            
        Returns:
            FP16 model
        """
        logger.info("Converting to FP16...")
        
        original_dtype = next(model.parameters()).dtype
        model = model.half()
        
        logger.info(f"Converted from {original_dtype} to torch.float16")
        return model


class BF16Quantizer(BaseQuantizer):
    """
    Convert model to BF16 (bfloat16).
    
    Better numerical stability than FP16 for training-like operations.
    Requires hardware support (Ampere+ NVIDIA GPUs, recent CPUs).
    """
    
    def __init__(self):
        super().__init__()
        # Check hardware support
        self.cuda_supported = (
            torch.cuda.is_available() and 
            torch.cuda.is_bf16_supported()
        )
        self.cpu_supported = hasattr(torch, 'bfloat16')
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Convert model to BF16.
        
        Args:
            model: Model to convert
            config: Configuration (unused for BF16)
            
        Returns:
            BF16 model
        """
        if not (self.cuda_supported or self.cpu_supported):
            logger.warning("BF16 not supported on this hardware, falling back to FP16")
            return model.half()
        
        logger.info("Converting to BF16...")
        
        original_dtype = next(model.parameters()).dtype
        model = model.to(torch.bfloat16)
        
        logger.info(f"Converted from {original_dtype} to torch.bfloat16")
        return model


class MixedPrecisionQuantizer(BaseQuantizer):
    """
    Apply mixed precision - keep certain layers in higher precision.
    
    Useful for models where specific layers (like embeddings or output)
    benefit from higher precision while compute layers can use lower.
    """
    
    def __init__(
        self,
        high_precision_layers: list = None,
        low_precision: torch.dtype = torch.float16,
        high_precision: torch.dtype = torch.float32,
    ):
        """
        Args:
            high_precision_layers: Layer names to keep in high precision
            low_precision: Dtype for most layers
            high_precision: Dtype for specified layers
        """
        super().__init__()
        self.high_precision_layers = high_precision_layers or [
            "embed", "embedding", "lm_head", "output"
        ]
        self.low_precision = low_precision
        self.high_precision = high_precision
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """Apply mixed precision quantization."""
        logger.info("Applying mixed precision...")
        
        # First convert everything to low precision
        model = model.to(self.low_precision)
        
        # Then convert specified layers back to high precision
        converted_layers = []
        for name, module in model.named_modules():
            should_convert = any(
                hp_name in name.lower() 
                for hp_name in self.high_precision_layers
            )
            if should_convert and hasattr(module, 'weight'):
                module.to(self.high_precision)
                converted_layers.append(name)
        
        logger.info(f"Kept {len(converted_layers)} layers in {self.high_precision}")
        return model
