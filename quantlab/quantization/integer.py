"""
Integer quantizers (INT8, INT4).
"""

import logging
from typing import Dict, Any, Optional, List

import torch
from torch import nn

from quantlab.quantization.base import BaseQuantizer


logger = logging.getLogger(__name__)


class INT8Quantizer(BaseQuantizer):
    """
    8-bit integer quantization.
    
    Uses PyTorch's dynamic quantization or BitsAndBytes for LLM-specific
    INT8 with mixed-precision decomposition.
    """
    
    def __init__(self, use_bitsandbytes: bool = True):
        """
        Args:
            use_bitsandbytes: Use bitsandbytes library (recommended for LLMs)
        """
        super().__init__()
        self.use_bitsandbytes = use_bitsandbytes
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for required dependencies."""
        if self.use_bitsandbytes:
            try:
                import bitsandbytes
                self.bnb_available = True
            except ImportError:
                logger.warning("bitsandbytes not available, using PyTorch quantization")
                self.bnb_available = False
                self.use_bitsandbytes = False
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply INT8 quantization.
        
        Note: For HuggingFace models, INT8 is typically applied at load time
        via BitsAndBytesConfig. This method is for post-hoc quantization.
        
        Args:
            model: Model to quantize
            config: Configuration with:
                - threshold: INT8 threshold for mixed precision
                - skip_modules: Modules to skip
                
        Returns:
            Quantized model
        """
        threshold = config.get("threshold", 6.0)
        skip_modules = config.get("skip_modules", [])
        
        if self.use_bitsandbytes and self.bnb_available:
            return self._quantize_bnb(model, threshold, skip_modules)
        else:
            return self._quantize_pytorch(model, skip_modules)
    
    def _quantize_bnb(
        self,
        model: nn.Module,
        threshold: float,
        skip_modules: List[str],
    ) -> nn.Module:
        """Quantize using bitsandbytes."""
        import bitsandbytes as bnb
        
        logger.info(f"Applying BitsAndBytes INT8 (threshold={threshold})")
        
        # Replace Linear layers with Int8 variants
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(skip in name for skip in skip_modules):
                    continue
                
                # Get parent and attribute name
                *parent_path, attr = name.split(".")
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                
                # Create Int8 linear
                new_module = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    threshold=threshold,
                )
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                
                setattr(parent, attr, new_module)
        
        return model
    
    def _quantize_pytorch(
        self,
        model: nn.Module,
        skip_modules: List[str],
    ) -> nn.Module:
        """Quantize using PyTorch dynamic quantization."""
        logger.info("Applying PyTorch dynamic INT8 quantization")
        
        # Dynamic quantization for Linear layers
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        return model


class INT4Quantizer(BaseQuantizer):
    """
    4-bit integer quantization.
    
    Aggressive compression with good quality retention for LLMs.
    Uses bitsandbytes NF4/FP4 formats.
    """
    
    def __init__(self, quant_type: str = "fp4"):
        """
        Args:
            quant_type: Quantization format ('fp4' or 'nf4')
        """
        super().__init__()
        self.quant_type = quant_type
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for bitsandbytes."""
        try:
            import bitsandbytes
            self.bnb_available = True
        except ImportError:
            self.bnb_available = False
            raise ImportError(
                "bitsandbytes is required for INT4 quantization. "
                "Install with: pip install bitsandbytes"
            )
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply INT4 quantization.
        
        Note: For HuggingFace models, 4-bit is typically applied at load time.
        This method demonstrates the concept for educational purposes.
        
        Args:
            model: Model to quantize
            config: Configuration with:
                - group_size: Quantization group size
                - double_quant: Use double quantization
                
        Returns:
            Quantized model
        """
        logger.info(f"INT4 quantization (type={self.quant_type})")
        logger.warning(
            "For best results with 4-bit quantization, load the model with "
            "BitsAndBytesConfig(load_in_4bit=True) at initialization."
        )
        
        # 4-bit is typically done at load time with transformers
        # This is a placeholder showing the configuration
        config_summary = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": self.quant_type,
            "bnb_4bit_use_double_quant": config.get("double_quant", True),
            "bnb_4bit_compute_dtype": "float16",
        }
        
        logger.info(f"Recommended BitsAndBytesConfig: {config_summary}")
        
        return model


class StaticQuantizer(BaseQuantizer):
    """
    Static quantization with calibration.
    
    Requires a calibration dataset to determine optimal quantization ranges.
    Provides better accuracy than dynamic quantization at the cost of
    requiring calibration data.
    """
    
    def __init__(self):
        super().__init__()
        self.calibration_data = None
        self.observer_stats = {}
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: Any,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Collect statistics for static quantization.
        
        Args:
            model: Model to calibrate
            calibration_data: DataLoader or list of input tensors
            config: Calibration configuration
            
        Returns:
            Calibrated model ready for quantization
        """
        logger.info("Calibrating for static quantization...")
        
        # Prepare model for calibration
        model.eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        
        # Prepare with observers
        prepared_model = torch.ao.quantization.prepare(model)
        
        # Run calibration
        num_samples = config.get("num_samples", 100)
        samples_processed = 0
        
        with torch.no_grad():
            for batch in calibration_data:
                if samples_processed >= num_samples:
                    break
                    
                if isinstance(batch, dict):
                    prepared_model(**batch)
                else:
                    prepared_model(batch)
                    
                samples_processed += 1
        
        logger.info(f"Calibration complete. Processed {samples_processed} samples.")
        
        return prepared_model
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply static quantization.
        
        Model should be calibrated first using calibrate().
        
        Args:
            model: Calibrated model
            config: Quantization configuration
            
        Returns:
            Quantized model
        """
        logger.info("Converting to static INT8...")
        
        # Convert calibrated model to quantized
        quantized_model = torch.ao.quantization.convert(model)
        
        return quantized_model
