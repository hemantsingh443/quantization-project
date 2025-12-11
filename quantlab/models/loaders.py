"""
Model loaders for different sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from quantlab.config import QuantizationConfig, HardwareConfig


logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Base class for model loaders."""
    
    @abstractmethod
    def load(
        self,
        model_path: str,
        quantization_config: QuantizationConfig,
        hardware_config: HardwareConfig,
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        pass
    
    @abstractmethod
    def supports(self, model_path: str) -> bool:
        """Check if this loader can handle the given model path."""
        pass


class HFModelLoader(BaseLoader):
    """Loader for HuggingFace Hub models."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
    
    def supports(self, model_path: str) -> bool:
        """Check if path looks like a HuggingFace model ID."""
        # HF model IDs are in format "org/model" or just "model"
        if Path(model_path).exists():
            return False
        return "/" in model_path or not any(c in model_path for c in ["\\", ":"])
    
    def load(
        self,
        model_path: str,
        quantization_config: QuantizationConfig,
        hardware_config: HardwareConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model from HuggingFace Hub."""
        logger.info(f"Loading from HuggingFace: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        kwargs = self._build_kwargs(quantization_config, hardware_config)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs
        )
        
        model.eval()
        return model, tokenizer
    
    def _build_kwargs(
        self,
        quant_config: QuantizationConfig,
        hw_config: HardwareConfig
    ) -> Dict[str, Any]:
        """Build model loading kwargs."""
        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        
        # Device mapping
        if hw_config.cpu_offload:
            kwargs["device_map"] = "auto"
        
        # Memory limits
        if hw_config.gpu_memory_limit:
            kwargs["max_memory"] = {
                0: f"{hw_config.gpu_memory_limit}GB",
                "cpu": "16GB"
            }
        
        # Dtype
        if quant_config.method == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif quant_config.method == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        
        return kwargs


class LocalModelLoader(BaseLoader):
    """Loader for local model checkpoints."""
    
    def supports(self, model_path: str) -> bool:
        """Check if path is a local directory with model files."""
        path = Path(model_path)
        if not path.exists():
            return False
        
        # Check for common model files
        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        return any((path / f).exists() for f in model_files)
    
    def load(
        self,
        model_path: str,
        quantization_config: QuantizationConfig,
        hardware_config: HardwareConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model from local path."""
        logger.info(f"Loading from local: {model_path}")
        
        path = Path(model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=True,
            local_files_only=True
        )
        
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        
        if hardware_config.cpu_offload:
            kwargs["device_map"] = "auto"
        
        if quantization_config.method == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif quantization_config.method == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            **kwargs
        )
        
        model.eval()
        return model, tokenizer


class GPTQLoader(BaseLoader):
    """Loader for GPTQ quantized models."""
    
    def supports(self, model_path: str) -> bool:
        """Check if this is a GPTQ model."""
        path = Path(model_path) if Path(model_path).exists() else None
        if path:
            return (path / "quantize_config.json").exists()
        # For HF models, check if name contains GPTQ indicators
        return "gptq" in model_path.lower()
    
    def load(
        self,
        model_path: str,
        quantization_config: QuantizationConfig,
        hardware_config: HardwareConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load GPTQ quantized model."""
        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ImportError:
            raise ImportError("auto-gptq is required for GPTQ models. Install with: pip install auto-gptq")
        
        logger.info(f"Loading GPTQ model: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device_map="auto" if hardware_config.cpu_offload else None,
            use_safetensors=True,
            trust_remote_code=True,
        )
        
        model.eval()
        return model, tokenizer


class AWQLoader(BaseLoader):
    """Loader for AWQ quantized models."""
    
    def supports(self, model_path: str) -> bool:
        """Check if this is an AWQ model."""
        return "awq" in model_path.lower()
    
    def load(
        self,
        model_path: str,
        quantization_config: QuantizationConfig,
        hardware_config: HardwareConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load AWQ quantized model."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("autoawq is required for AWQ models. Install with: pip install autoawq")
        
        logger.info(f"Loading AWQ model: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            device_map="auto" if hardware_config.cpu_offload else None,
            fuse_layers=True,
        )
        
        model.eval()
        return model, tokenizer
