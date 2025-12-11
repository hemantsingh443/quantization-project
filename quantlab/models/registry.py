"""
Model Registry - Unified interface for loading and managing models.
"""

import logging
from typing import Optional, Dict, Any, Tuple, Callable
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from quantlab.config import QuantizationConfig, HardwareConfig


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for loading and managing models.
    
    Supports HuggingFace models with various quantization configurations.
    Extensible for custom model loaders.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the registry.
        
        Args:
            cache_dir: Directory for caching downloaded models
        """
        self.cache_dir = cache_dir
        self._custom_loaders: Dict[str, Callable] = {}
        self._loaded_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
    
    def register_loader(self, name: str, loader: Callable):
        """
        Register a custom model loader.
        
        Args:
            name: Unique name for the loader
            loader: Callable that takes (model_name, config) and returns (model, tokenizer)
        """
        self._custom_loaders[name] = loader
        logger.info(f"Registered custom loader: {name}")
    
    def load_model(
        self,
        model_name: str,
        revision: Optional[str] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        hardware_config: Optional[HardwareConfig] = None,
        use_cache: bool = True,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model with optional quantization.
        
        Args:
            model_name: HuggingFace model name or local path
            revision: Model revision/commit hash
            quantization_config: Quantization settings
            hardware_config: Hardware constraints
            use_cache: Whether to cache and reuse loaded models
            
        Returns:
            Tuple of (model, tokenizer)
        """
        quantization_config = quantization_config or QuantizationConfig()
        hardware_config = hardware_config or HardwareConfig()
        
        # Create cache key
        cache_key = f"{model_name}_{quantization_config.method}"
        
        # Check cache
        if use_cache and cache_key in self._loaded_models:
            logger.info(f"Using cached model: {cache_key}")
            return self._loaded_models[cache_key]
        
        # Check for custom loader
        if model_name in self._custom_loaders:
            model, tokenizer = self._custom_loaders[model_name](model_name, quantization_config)
        else:
            # Default: HuggingFace loading
            model, tokenizer = self._load_hf_model(
                model_name=model_name,
                revision=revision,
                quantization_config=quantization_config,
                hardware_config=hardware_config,
            )
        
        # Cache
        if use_cache:
            self._loaded_models[cache_key] = (model, tokenizer)
        
        return model, tokenizer
    
    def _load_hf_model(
        self,
        model_name: str,
        revision: Optional[str],
        quantization_config: QuantizationConfig,
        hardware_config: HardwareConfig,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a HuggingFace model with quantization."""
        
        # Build loading kwargs
        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if revision:
            kwargs["revision"] = revision
        
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        
        # Device map
        device_map = hardware_config.get_device_map()
        if device_map:
            kwargs["device_map"] = device_map
        
        # Max memory constraint
        if hardware_config.max_memory:
            kwargs["max_memory"] = hardware_config.max_memory
        elif hardware_config.gpu_memory_limit:
            # Set max memory based on limit
            kwargs["max_memory"] = {
                0: f"{hardware_config.gpu_memory_limit}GB",
                "cpu": "16GB"
            }
        
        # Quantization
        bnb_config = self._build_bnb_config(quantization_config)
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
        
        # Dtype
        if quantization_config.method == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif quantization_config.method == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif quantization_config.method in ["none", "fp32"]:
            kwargs["torch_dtype"] = torch.float32
        elif quantization_config.method in ["int8", "int4", "nf4"]:
            # BitsAndBytes handles dtype
            kwargs["torch_dtype"] = torch.float16
        
        logger.info(f"Loading model {model_name} with kwargs: {list(kwargs.keys())}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        
        # Move to device if no device_map
        if device_map is None and hardware_config.device != "cpu":
            if torch.cuda.is_available():
                model = model.cuda()
        
        model.eval()
        
        return model, tokenizer
    
    def _build_bnb_config(self, quant_config: QuantizationConfig) -> Optional[BitsAndBytesConfig]:
        """Build BitsAndBytesConfig from quantization config."""
        
        if quant_config.method == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=quant_config.int8_threshold,
                llm_int8_skip_modules=quant_config.skip_modules or None,
            )
        
        elif quant_config.method in ["int4", "nf4"]:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_config.quant_type,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=quant_config.double_quant,
            )
        
        return None
    
    def unload_model(self, model_name: str, quant_method: str = "none"):
        """Unload a cached model to free memory."""
        cache_key = f"{model_name}_{quant_method}"
        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {cache_key}")
    
    def clear_cache(self):
        """Clear all cached models."""
        self._loaded_models.clear()
        torch.cuda.empty_cache()
        logger.info("Cleared model cache")
    
    def list_cached_models(self) -> list:
        """List currently cached models."""
        return list(self._loaded_models.keys())


# Convenience function for quick loading
def load_model(
    model_name: str,
    quantization: str = "none",
    device: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Quick model loading convenience function.
    
    Args:
        model_name: Model name or path
        quantization: Quantization method
        device: Device to load on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    registry = ModelRegistry()
    quant_config = QuantizationConfig(method=quantization)
    hardware_config = HardwareConfig(device=device)
    
    return registry.load_model(
        model_name=model_name,
        quantization_config=quant_config,
        hardware_config=hardware_config,
        use_cache=False,
    )
