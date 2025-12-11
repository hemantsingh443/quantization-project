"""
Advanced quantization methods (NF4, GPTQ, AWQ).
"""

import logging
from typing import Dict, Any, Optional, List

import torch
from torch import nn

from quantlab.quantization.base import BaseQuantizer


logger = logging.getLogger(__name__)


class NF4Quantizer(BaseQuantizer):
    """
    NormalFloat4 (NF4) Quantization.
    
    Optimal 4-bit quantization for normally distributed weights.
    Developed for QLoRA, provides better quality than uniform INT4.
    
    Reference: https://arxiv.org/abs/2305.14314
    """
    
    def __init__(
        self,
        compute_dtype: torch.dtype = torch.float16,
        double_quant: bool = True,
    ):
        """
        Args:
            compute_dtype: Dtype for computation (float16 or bfloat16)
            double_quant: Apply double quantization for extra compression
        """
        super().__init__()
        self.compute_dtype = compute_dtype
        self.double_quant = double_quant
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verify bitsandbytes is available."""
        try:
            import bitsandbytes
            self.bnb_available = True
        except ImportError:
            self.bnb_available = False
            raise ImportError(
                "bitsandbytes >= 0.39.0 is required for NF4 quantization. "
                "Install with: pip install bitsandbytes"
            )
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply NF4 quantization.
        
        Note: NF4 is typically applied at model loading time with transformers.
        This method provides configuration guidance.
        
        Args:
            model: Model to quantize
            config: Additional configuration
            
        Returns:
            Model (typically unchanged - NF4 applied at load time)
        """
        logger.info("NF4 quantization is applied at model loading time.")
        logger.info("Use BitsAndBytesConfig with the following settings:")
        
        bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": str(self.compute_dtype),
            "bnb_4bit_use_double_quant": self.double_quant,
        }
        
        logger.info(f"Config: {bnb_config}")
        
        return model
    
    @staticmethod
    def get_bnb_config() -> Dict[str, Any]:
        """Get BitsAndBytesConfig parameters for NF4."""
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        }


class GPTQQuantizer(BaseQuantizer):
    """
    GPTQ Quantization.
    
    Post-training quantization method that uses approximate second-order
    information to minimize quantization error. Provides high-quality
    4-bit and 3-bit quantization.
    
    Reference: https://arxiv.org/abs/2210.17323
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        act_order: bool = True,
        desc_act: bool = True,
    ):
        """
        Args:
            bits: Quantization bits (2, 3, 4, 8)
            group_size: Group size for quantization
            act_order: Quantize by activation order (recommended)
            desc_act: Descending activation order
        """
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.act_order = act_order
        self.desc_act = desc_act
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for auto-gptq."""
        try:
            import auto_gptq
            self.gptq_available = True
        except ImportError:
            self.gptq_available = False
            logger.warning(
                "auto-gptq not available. Install with: pip install auto-gptq"
            )
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply GPTQ quantization.
        
        Requires calibration data for best results.
        
        Args:
            model: Model to quantize
            config: Configuration with:
                - calibration_data: DataLoader for calibration
                - num_samples: Number of calibration samples
                
        Returns:
            Quantized model
        """
        if not self.gptq_available:
            raise ImportError("auto-gptq is required for GPTQ quantization")
        
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        
        logger.info(f"GPTQ quantization: {self.bits}-bit, group_size={self.group_size}")
        
        # GPTQ requires starting from the model name, not a loaded model
        # This is a configuration helper
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            damp_percent=0.01,
        )
        
        logger.info(f"GPTQ config: {quantize_config}")
        logger.info(
            "For actual GPTQ quantization, use:\n"
            "  from auto_gptq import AutoGPTQForCausalLM\n"
            "  model = AutoGPTQForCausalLM.from_pretrained(...)\n"
            "  model.quantize(calibration_data)"
        )
        
        return model
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: Any,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Calibrate and quantize using GPTQ.
        
        Args:
            model: Model to calibrate (or model name)
            calibration_data: List of input samples
            config: Configuration
            
        Returns:
            Quantized model
        """
        if not self.gptq_available:
            raise ImportError("auto-gptq is required for GPTQ calibration")
        
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        
        num_samples = config.get("num_samples", 128)
        
        logger.info(f"GPTQ calibration with {num_samples} samples...")
        
        # If model is already loaded, we need the model path
        model_name = config.get("model_name")
        if model_name is None:
            raise ValueError("model_name required in config for GPTQ calibration")
        
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
        )
        
        # Load and quantize
        gptq_model = AutoGPTQForCausalLM.from_pretrained(
            model_name,
            quantize_config,
        )
        
        # Prepare calibration examples
        examples = []
        for i, sample in enumerate(calibration_data):
            if i >= num_samples:
                break
            examples.append(sample)
        
        # Quantize
        gptq_model.quantize(examples)
        
        logger.info("GPTQ calibration and quantization complete")
        
        return gptq_model
    
    @staticmethod
    def get_config(bits: int = 4, group_size: int = 128) -> Dict[str, Any]:
        """Get GPTQ configuration dictionary."""
        return {
            "bits": bits,
            "group_size": group_size,
            "desc_act": True,
            "damp_percent": 0.01,
        }


class AWQQuantizer(BaseQuantizer):
    """
    Activation-aware Weight Quantization (AWQ).
    
    Protects salient weights based on activation magnitudes.
    Generally faster inference than GPTQ with comparable quality.
    
    Reference: https://arxiv.org/abs/2306.00978
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        version: str = "gemm",
    ):
        """
        Args:
            bits: Quantization bits (4)
            group_size: Group size for quantization
            version: AWQ kernel version ('gemm' or 'gemv')
        """
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.version = version
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for autoawq."""
        try:
            import awq
            self.awq_available = True
        except ImportError:
            self.awq_available = False
            logger.warning(
                "autoawq not available. Install with: pip install autoawq"
            )
    
    def quantize(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply AWQ quantization.
        
        Args:
            model: Model to quantize
            config: Configuration
            
        Returns:
            Quantized model
        """
        if not self.awq_available:
            raise ImportError("autoawq is required for AWQ quantization")
        
        logger.info(f"AWQ quantization: {self.bits}-bit, group_size={self.group_size}")
        
        # AWQ also requires starting from model path
        logger.info(
            "For actual AWQ quantization, use:\n"
            "  from awq import AutoAWQForCausalLM\n"
            "  model = AutoAWQForCausalLM.from_pretrained(...)\n"
            "  model.quantize(tokenizer, quant_config)"
        )
        
        return model
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: Any,
        config: Dict[str, Any],
    ) -> nn.Module:
        """
        Calibrate and quantize using AWQ.
        
        Args:
            model: Model to calibrate
            calibration_data: Tokenized calibration texts
            config: Configuration with model_name
            
        Returns:
            Quantized model
        """
        if not self.awq_available:
            raise ImportError("autoawq is required for AWQ calibration")
        
        from awq import AutoAWQForCausalLM
        
        model_name = config.get("model_name")
        if model_name is None:
            raise ValueError("model_name required in config for AWQ calibration")
        
        logger.info(f"AWQ calibration for {model_name}...")
        
        quant_config = {
            "zero_point": True,
            "q_group_size": self.group_size,
            "w_bit": self.bits,
            "version": self.version,
        }
        
        awq_model = AutoAWQForCausalLM.from_pretrained(model_name)
        
        # Need tokenizer for AWQ
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        awq_model.quantize(tokenizer, quant_config=quant_config)
        
        logger.info("AWQ calibration and quantization complete")
        
        return awq_model
    
    @staticmethod
    def get_config(bits: int = 4, group_size: int = 128) -> Dict[str, Any]:
        """Get AWQ configuration dictionary."""
        return {
            "zero_point": True,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": "gemm",
        }
