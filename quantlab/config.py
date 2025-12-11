"""
Configuration management for QuantLab.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import json


@dataclass
class HardwareConfig:
    """Hardware configuration and constraints."""
    device: str = "auto"  # auto, cuda, cpu
    gpu_memory_limit: Optional[float] = None  # GB
    cpu_offload: bool = True
    max_memory: Optional[Dict[str, str]] = None  # For device_map
    
    def get_device_map(self) -> Optional[Dict]:
        """Get device map for model loading."""
        if self.device == "cpu":
            return {"": "cpu"}
        if self.cpu_offload:
            return "auto"
        return None


@dataclass
class QuantizationConfig:
    """Quantization strategy configuration."""
    method: str = "none"  # none, fp16, bf16, int8, int4, nf4, gptq, awq
    
    # INT8 specific
    int8_threshold: float = 6.0
    
    # INT4/NF4 specific
    bits: int = 4
    group_size: int = 128
    double_quant: bool = True
    quant_type: str = "nf4"  # nf4, fp4
    
    # GPTQ specific
    gptq_bits: int = 4
    gptq_group_size: int = 128
    gptq_act_order: bool = True
    gptq_desc_act: bool = True
    
    # AWQ specific
    awq_bits: int = 4
    awq_group_size: int = 128
    awq_version: str = "gemm"
    
    # Layer-wise control
    skip_modules: List[str] = field(default_factory=list)
    quantize_modules: Optional[List[str]] = None  # None = all

    def to_bnb_config(self) -> Optional[Dict[str, Any]]:
        """Convert to bitsandbytes configuration."""
        if self.method == "int8":
            return {
                "load_in_8bit": True,
                "llm_int8_threshold": self.int8_threshold,
            }
        elif self.method in ["int4", "nf4"]:
            return {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": self.double_quant,
            }
        return None


@dataclass
class BenchmarkConfig:
    """Benchmarking configuration."""
    # Latency
    warmup_runs: int = 3
    benchmark_runs: int = 10
    
    # Input
    input_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    output_length: int = 50
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    
    # Evaluation
    eval_suites: List[str] = field(default_factory=lambda: ["simple"])
    eval_samples: int = 100  # Limit for quick testing
    
    # Memory
    track_memory: bool = True
    track_power: bool = False  # Requires pynvml


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Identification
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Model
    model_name: str = "facebook/opt-125m"
    model_revision: Optional[str] = None
    tokenizer_name: Optional[str] = None
    
    # Sub-configs
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        hardware = HardwareConfig(**data.get("hardware", {}))
        quantization = QuantizationConfig(**data.get("quantization", {}))
        benchmark = BenchmarkConfig(**data.get("benchmark", {}))
        
        return cls(
            name=data.get("name"),
            tags=data.get("tags", []),
            model_name=data.get("model_name", "facebook/opt-125m"),
            model_revision=data.get("model_revision"),
            tokenizer_name=data.get("tokenizer_name"),
            hardware=hardware,
            quantization=quantization,
            benchmark=benchmark,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class QuantLabConfig:
    """Global QuantLab configuration."""
    # Paths
    experiments_dir: Path = field(default_factory=lambda: Path("./experiments"))
    cache_dir: Optional[Path] = None
    
    # Database
    db_path: Optional[Path] = None  # SQLite database path
    
    # Logging
    log_level: str = "INFO"
    
    # Dashboard
    dashboard_port: int = 8501
    
    def __post_init__(self):
        self.experiments_dir = Path(self.experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        if self.db_path is None:
            self.db_path = self.experiments_dir / "quantlab.db"


# Default configurations for common scenarios
DEFAULT_CONFIGS = {
    "fp16": ExperimentConfig(
        quantization=QuantizationConfig(method="fp16")
    ),
    "int8": ExperimentConfig(
        quantization=QuantizationConfig(method="int8")
    ),
    "int4": ExperimentConfig(
        quantization=QuantizationConfig(method="int4", quant_type="fp4")
    ),
    "nf4": ExperimentConfig(
        quantization=QuantizationConfig(method="nf4", quant_type="nf4")
    ),
}


def get_default_config(name: str) -> ExperimentConfig:
    """Get a default configuration by name."""
    if name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(DEFAULT_CONFIGS.keys())}")
    return DEFAULT_CONFIGS[name]
