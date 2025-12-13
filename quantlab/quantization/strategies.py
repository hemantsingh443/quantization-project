"""
Quantization strategies - clean abstraction for load-time and post-training quantization.

This module provides a clear separation between:
1. Load-time quantization (BitsAndBytes INT8/INT4/NF4) - uses get_load_config()
2. Post-training quantization (GPTQ, AWQ) - uses quantize()
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

import torch
from torch import nn

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    BitsAndBytesConfig = None


logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    """Result of a quantization operation."""
    success: bool
    method: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    layer_stats: Dict[str, Any] = None
    errors: List[str] = None
    
    def __post_init__(self):
        self.layer_stats = self.layer_stats or {}
        self.errors = self.errors or []
    
    @property
    def size_reduction_pct(self) -> float:
        """Percentage of size reduction."""
        if self.original_size_mb == 0:
            return 0.0
        return (1 - self.quantized_size_mb / self.original_size_mb) * 100


class QuantizationStrategy(ABC):
    """
    Base class for quantization strategies.
    
    This provides a clean abstraction that handles both:
    - Load-time quantization (get_load_config returns BitsAndBytesConfig)
    - Post-training quantization (quantize transforms an existing model)
    
    Usage:
        strategy = NF4Strategy()
        
        # For load-time quantization:
        bnb_config = strategy.get_load_config()
        model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
        
        # For post-training quantization:
        quantized_model = strategy.quantize(model, calibration_data)
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self._is_load_time = True  # Most strategies are load-time
    
    @property
    def is_load_time_quantization(self) -> bool:
        """Whether this strategy applies at model load time."""
        return self._is_load_time
    
    @property
    def requires_calibration(self) -> bool:
        """Whether this strategy requires calibration data."""
        return False
    
    def get_load_config(self) -> Optional[Any]:
        """
        Get configuration for load-time quantization.
        
        Returns BitsAndBytesConfig or similar for strategies that 
        quantize at load time.
        
        Returns:
            Configuration object for model loading, or None if not applicable
        """
        return None
    
    def get_torch_dtype(self) -> Optional[torch.dtype]:
        """Get the torch dtype to use when loading."""
        return None
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Any = None,
        **kwargs
    ) -> nn.Module:
        """
        Apply post-training quantization to a model.
        
        Only applicable for strategies where is_load_time_quantization is False.
        
        Args:
            model: The model to quantize
            calibration_data: Representative data for calibration
            **kwargs: Additional configuration
            
        Returns:
            Quantized model
        """
        if self._is_load_time:
            logger.info(f"{self.name} is a load-time strategy. Use get_load_config() instead.")
            return model
        raise NotImplementedError("Subclasses must implement quantize() for PTQ")
    
    def estimate_memory_reduction(self) -> float:
        """Estimate memory reduction factor (e.g., 0.5 for 50% reduction)."""
        return 1.0
    
    def validate_hardware(self) -> List[str]:
        """
        Validate hardware requirements for this strategy.
        
        Returns:
            List of warning messages (empty if all requirements met)
        """
        return []


class NoQuantizationStrategy(QuantizationStrategy):
    """No quantization - use full precision (FP32)."""
    
    def __init__(self):
        super().__init__()
        self._is_load_time = True
    
    def get_torch_dtype(self) -> torch.dtype:
        return torch.float32
    
    def estimate_memory_reduction(self) -> float:
        return 1.0


class FP16Strategy(QuantizationStrategy):
    """Half precision (FP16) - 50% memory reduction."""
    
    def __init__(self):
        super().__init__()
        self._is_load_time = True
    
    def get_torch_dtype(self) -> torch.dtype:
        return torch.float16
    
    def estimate_memory_reduction(self) -> float:
        return 0.5


class BF16Strategy(QuantizationStrategy):
    """Brain Float16 - better numerical stability than FP16."""
    
    def __init__(self):
        super().__init__()
        self._is_load_time = True
        self.cuda_supported = (
            torch.cuda.is_available() and 
            torch.cuda.is_bf16_supported()
        )
    
    def get_torch_dtype(self) -> torch.dtype:
        if self.cuda_supported:
            return torch.bfloat16
        logger.warning("BF16 not supported, falling back to FP16")
        return torch.float16
    
    def validate_hardware(self) -> List[str]:
        if not self.cuda_supported:
            return ["BF16 not supported on this hardware, will use FP16"]
        return []
    
    def estimate_memory_reduction(self) -> float:
        return 0.5


class INT8Strategy(QuantizationStrategy):
    """8-bit integer quantization via BitsAndBytes."""
    
    def __init__(self, threshold: float = 6.0, skip_modules: List[str] = None):
        super().__init__()
        self._is_load_time = True
        self.threshold = threshold
        self.skip_modules = skip_modules or []
    
    def get_load_config(self) -> Optional[Any]:
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes is required for INT8 quantization")
        
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=self.threshold,
            llm_int8_skip_modules=self.skip_modules if self.skip_modules else None,
        )
    
    def get_torch_dtype(self) -> torch.dtype:
        return torch.float16
    
    def estimate_memory_reduction(self) -> float:
        return 0.5  # ~50% reduction


class INT4Strategy(QuantizationStrategy):
    """4-bit integer quantization via BitsAndBytes (FP4 format)."""
    
    def __init__(
        self,
        compute_dtype: torch.dtype = torch.float16,
        double_quant: bool = True,
    ):
        super().__init__()
        self._is_load_time = True
        self.compute_dtype = compute_dtype
        self.double_quant = double_quant
    
    def get_load_config(self) -> Optional[Any]:
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes is required for INT4 quantization")
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.double_quant,
        )
    
    def get_torch_dtype(self) -> torch.dtype:
        return self.compute_dtype
    
    def estimate_memory_reduction(self) -> float:
        return 0.25  # ~75% reduction


class NF4Strategy(QuantizationStrategy):
    """
    4-bit NormalFloat quantization via BitsAndBytes.
    
    Optimal for normally distributed weights (common in LLMs).
    Reference: QLoRA paper (https://arxiv.org/abs/2305.14314)
    """
    
    def __init__(
        self,
        compute_dtype: torch.dtype = torch.float16,
        double_quant: bool = True,
    ):
        super().__init__()
        self._is_load_time = True
        self.compute_dtype = compute_dtype
        self.double_quant = double_quant
    
    def get_load_config(self) -> Optional[Any]:
        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes is required for NF4 quantization")
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.double_quant,
        )
    
    def get_torch_dtype(self) -> torch.dtype:
        return self.compute_dtype
    
    def estimate_memory_reduction(self) -> float:
        return 0.25  # ~75% reduction


class GPTQStrategy(QuantizationStrategy):
    """
    GPTQ Quantization - Post-training quantization requiring calibration.
    
    Reference: https://arxiv.org/abs/2210.17323
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = True,
        damp_percent: float = 0.01,
    ):
        super().__init__()
        self._is_load_time = False  # This is PTQ!
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.damp_percent = damp_percent
    
    @property
    def requires_calibration(self) -> bool:
        return True
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Any = None,
        **kwargs
    ) -> nn.Module:
        """Apply GPTQ quantization with calibration."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError(
                "auto-gptq is required for GPTQ quantization. "
                "Install with: pip install auto-gptq"
            )
        
        if calibration_data is None:
            raise ValueError("GPTQ requires calibration_data")
        
        model_name = kwargs.get("model_name")
        if model_name is None:
            raise ValueError("model_name required in kwargs for GPTQ")
        
        logger.info(f"Running GPTQ quantization: {self.bits}-bit, group_size={self.group_size}")
        
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            damp_percent=self.damp_percent,
        )
        
        # Load fresh model for GPTQ
        gptq_model = AutoGPTQForCausalLM.from_pretrained(
            model_name,
            quantize_config,
        )
        
        # Quantize with calibration data
        gptq_model.quantize(calibration_data)
        
        logger.info("GPTQ quantization complete")
        return gptq_model
    
    def estimate_memory_reduction(self) -> float:
        if self.bits == 4:
            return 0.25
        elif self.bits == 8:
            return 0.5
        return 0.375  # 3-bit


class AWQStrategy(QuantizationStrategy):
    """
    AWQ (Activation-aware Weight Quantization) - PTQ with activation analysis.
    
    Reference: https://arxiv.org/abs/2306.00978
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        version: str = "gemm",
    ):
        super().__init__()
        self._is_load_time = False  # This is PTQ!
        self.bits = bits
        self.group_size = group_size
        self.version = version
    
    @property
    def requires_calibration(self) -> bool:
        return True
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Any = None,
        **kwargs
    ) -> nn.Module:
        """Apply AWQ quantization."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError(
                "autoawq is required for AWQ quantization. "
                "Install with: pip install autoawq"
            )
        
        model_name = kwargs.get("model_name")
        tokenizer = kwargs.get("tokenizer")
        
        if model_name is None:
            raise ValueError("model_name required in kwargs for AWQ")
        
        logger.info(f"Running AWQ quantization: {self.bits}-bit, group_size={self.group_size}")
        
        quant_config = {
            "zero_point": True,
            "q_group_size": self.group_size,
            "w_bit": self.bits,
            "version": self.version,
        }
        
        awq_model = AutoAWQForCausalLM.from_pretrained(model_name)
        
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        awq_model.quantize(tokenizer, quant_config=quant_config)
        
        logger.info("AWQ quantization complete")
        return awq_model
    
    def estimate_memory_reduction(self) -> float:
        return 0.25  # ~75% reduction for 4-bit


# Check for optional consumer inference dependencies
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

try:
    import exllamav2
    EXLLAMAV2_AVAILABLE = True
except (ImportError, RuntimeError, Exception):
    # ImportError: not installed
    # RuntimeError: CUDA/ninja issues during JIT compilation
    EXLLAMAV2_AVAILABLE = False
    exllamav2 = None


class GGUFStrategy(QuantizationStrategy):
    """
    GGUF format for llama.cpp inference.
    
    Gold standard for consumer-grade CPU/GPU inference. Uses llama-cpp-python
    for native integration with the llama.cpp library.
    
    Supported quantization types:
        - Q2_K: 2-bit (extreme compression, quality loss)
        - Q3_K_M: 3-bit medium
        - Q4_K_M: 4-bit medium (recommended balance)
        - Q5_K_M: 5-bit medium (good quality)
        - Q6_K: 6-bit (near-lossless)
        - Q8_0: 8-bit (minimal quality loss)
    
    Reference: https://github.com/ggerganov/llama.cpp
    """
    
    # Memory reduction estimates for each quant type (relative to FP16)
    QUANT_TYPE_MEMORY = {
        "Q2_K": 0.125,   # ~87.5% reduction
        "Q3_K_S": 0.1875,
        "Q3_K_M": 0.1875,
        "Q3_K_L": 0.1875,
        "Q4_0": 0.25,
        "Q4_K_S": 0.25,
        "Q4_K_M": 0.25,   # ~75% reduction
        "Q5_0": 0.3125,
        "Q5_K_S": 0.3125,
        "Q5_K_M": 0.3125, # ~68.75% reduction  
        "Q6_K": 0.375,    # ~62.5% reduction
        "Q8_0": 0.5,      # ~50% reduction
    }
    
    def __init__(
        self,
        quant_type: str = "Q4_K_M",
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_ctx: int = 2048,
        n_batch: int = 512,
        verbose: bool = False,
    ):
        super().__init__()
        self._is_load_time = True
        self.quant_type = quant_type.upper()
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.verbose = verbose
        
        if self.quant_type not in self.QUANT_TYPE_MEMORY:
            logger.warning(
                f"Unknown GGUF quant type: {quant_type}. "
                f"Known types: {list(self.QUANT_TYPE_MEMORY.keys())}"
            )
    
    @property
    def requires_calibration(self) -> bool:
        return False  # GGUF models are pre-quantized
    
    def get_load_config(self) -> Optional[Dict[str, Any]]:
        """
        Get configuration for loading GGUF models.
        
        Returns:
            Dict with llama.cpp loading parameters
        """
        return {
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "verbose": self.verbose,
        }
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a GGUF model using llama-cpp-python.
        
        Args:
            model_path: Path to .gguf file or HuggingFace model ID
            
        Returns:
            Llama model instance
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF support. "
                "Install with: pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )
        
        config = self.get_load_config()
        logger.info(f"Loading GGUF model: {model_path} with {self.quant_type}")
        
        model = Llama(
            model_path=model_path,
            **config,
        )
        
        return model
    
    def estimate_memory_reduction(self) -> float:
        """Estimate memory reduction based on quantization type."""
        return self.QUANT_TYPE_MEMORY.get(self.quant_type, 0.25)
    
    def validate_hardware(self) -> List[str]:
        warnings = []
        if not LLAMA_CPP_AVAILABLE:
            warnings.append("llama-cpp-python not installed")
        if self.n_gpu_layers != 0 and not torch.cuda.is_available():
            warnings.append("GPU layers requested but CUDA not available")
        return warnings


class ExLlamaV2Strategy(QuantizationStrategy):
    """
    ExLlamaV2 (EXL2) format for high-performance GPU inference.
    
    Optimized for consumer GPUs with mixed-precision quantization.
    Supports flexible bitrates from 2-8 bits, with per-layer optimization.
    
    Key features:
        - Mixed-precision quantization (different bits per layer)
        - Optimized CUDA kernels for fast inference
        - Flash attention support
        - Efficient KV cache management
    
    Reference: https://github.com/turboderp/exllamav2
    """
    
    def __init__(
        self,
        bits: float = 4.0,  # Target average bits (2.0-8.0)
        max_seq_len: int = 4096,
        rope_scale: float = 1.0,
        rope_alpha: float = 1.0,
        no_flash_attn: bool = False,
    ):
        super().__init__()
        self._is_load_time = True
        self.bits = bits
        self.max_seq_len = max_seq_len
        self.rope_scale = rope_scale
        self.rope_alpha = rope_alpha
        self.no_flash_attn = no_flash_attn
        
        if not 2.0 <= bits <= 8.0:
            logger.warning(f"EXL2 bits should be 2.0-8.0, got {bits}")
    
    @property
    def requires_calibration(self) -> bool:
        return False  # EXL2 models are pre-quantized
    
    def get_load_config(self) -> Optional[Dict[str, Any]]:
        """
        Get configuration for loading EXL2 models.
        
        Returns:
            Dict with ExLlamaV2 loading parameters
        """
        return {
            "max_seq_len": self.max_seq_len,
            "rope_scale": self.rope_scale,
            "rope_alpha": self.rope_alpha,
            "no_flash_attn": self.no_flash_attn,
        }
    
    def load_model(self, model_path: str) -> Any:
        """
        Load an EXL2 model using ExLlamaV2.
        
        Args:
            model_path: Path to EXL2 model directory
            
        Returns:
            ExLlamaV2 model and cache instances
        """
        if not EXLLAMAV2_AVAILABLE:
            raise ImportError(
                "exllamav2 is required for EXL2 support. "
                "Install with: pip install exllamav2"
            )
        
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
        from exllamav2.generator import ExLlamaV2StreamingGenerator
        
        logger.info(f"Loading EXL2 model: {model_path} ({self.bits}-bit avg)")
        
        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        
        if self.max_seq_len:
            config.max_seq_len = self.max_seq_len
        if self.rope_scale != 1.0:
            config.scale_pos_emb = self.rope_scale
        if self.rope_alpha != 1.0:
            config.scale_alpha_value = self.rope_alpha
        if self.no_flash_attn:
            config.no_flash_attn = True
        
        model = ExLlamaV2(config)
        model.load()
        
        cache = ExLlamaV2Cache(model, lazy=True)
        
        return {"model": model, "cache": cache, "config": config}
    
    def estimate_memory_reduction(self) -> float:
        """Estimate memory reduction based on target bits."""
        # FP16 is 16 bits, so reduction is bits/16
        return self.bits / 16.0
    
    def validate_hardware(self) -> List[str]:
        warnings = []
        if not EXLLAMAV2_AVAILABLE:
            warnings.append("exllamav2 not installed")
        if not torch.cuda.is_available():
            warnings.append("ExLlamaV2 requires CUDA GPU")
        return warnings


# Registry of available strategies
_STRATEGY_REGISTRY: Dict[str, type] = {
    "none": NoQuantizationStrategy,
    "fp32": NoQuantizationStrategy,
    "fp16": FP16Strategy,
    "bf16": BF16Strategy,
    "int8": INT8Strategy,
    "int4": INT4Strategy,
    "nf4": NF4Strategy,
    "gptq": GPTQStrategy,
    "awq": AWQStrategy,
    "gguf": GGUFStrategy,
    "exl2": ExLlamaV2Strategy,
}


def get_strategy(method: str, **kwargs) -> QuantizationStrategy:
    """
    Get a quantization strategy by name.
    
    Args:
        method: Strategy name (none, fp16, int8, int4, nf4, gptq, awq)
        **kwargs: Strategy-specific configuration
        
    Returns:
        Configured QuantizationStrategy instance
    """
    if method not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown quantization method: {method}. "
            f"Available: {list(_STRATEGY_REGISTRY.keys())}"
        )
    
    strategy_class = _STRATEGY_REGISTRY[method]
    return strategy_class(**kwargs)


def register_strategy(name: str, strategy_class: type):
    """Register a custom quantization strategy."""
    _STRATEGY_REGISTRY[name] = strategy_class


def list_strategies() -> List[str]:
    """List available quantization strategies."""
    return list(_STRATEGY_REGISTRY.keys())
