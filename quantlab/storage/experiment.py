"""
Experiment data model.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Experiment:
    """
    Represents a single quantization experiment with all metadata and results.
    
    Attributes:
        id: Unique experiment identifier
        model_name: HuggingFace model name or path
        model_size: Human-readable model size (e.g., "125M")
        quant_method: Quantization method used
        quant_config: Detailed quantization configuration
        hardware: Hardware information dict
        metrics: Benchmark metrics dict
        artifacts: List of artifact file paths
        name: Optional human-readable name
        tags: Tags for filtering/grouping
        status: Experiment status (pending, running, completed, failed)
        error: Error message if failed
        timestamp: When experiment was created
    """
    
    # Model info
    model_name: str
    model_size: Optional[str] = None
    
    # Quantization
    quant_method: str = "none"
    quant_config: Dict[str, Any] = field(default_factory=dict)
    
    # Environment
    hardware: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "quant_method": self.quant_method,
            "quant_config": self.quant_config,
            "hardware": self.hardware,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "name": self.name,
            "tags": self.tags,
            "status": self.status,
            "error": self.error,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            model_name=data["model_name"],
            model_size=data.get("model_size"),
            quant_method=data.get("quant_method", "none"),
            quant_config=data.get("quant_config", {}),
            hardware=data.get("hardware", {}),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", []),
            name=data.get("name"),
            tags=data.get("tags", []),
            status=data.get("status", "pending"),
            error=data.get("error"),
            timestamp=timestamp or datetime.now(),
        )
    
    def summary(self) -> str:
        """Get a formatted summary string."""
        lines = [
            f"Experiment: {self.id}",
            f"  Model: {self.model_name} ({self.model_size or 'unknown size'})",
            f"  Quantization: {self.quant_method}",
            f"  Status: {self.status}",
            f"  Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp else 'N/A'}",
        ]
        
        # Key metrics
        if "latency_mean_ms" in self.metrics:
            lines.append(f"  Latency: {self.metrics['latency_mean_ms']:.2f} ms")
        if "memory_mb" in self.metrics:
            lines.append(f"  Memory: {self.metrics['memory_mb']:.1f} MB")
        if "throughput_tps" in self.metrics:
            lines.append(f"  Throughput: {self.metrics['throughput_tps']:.1f} tokens/sec")
        
        if self.error:
            lines.append(f"  Error: {self.error}")
        
        return "\n".join(lines)
    
    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get a metric value with default."""
        return self.metrics.get(key, default)
    
    def add_artifact(self, path: str):
        """Add an artifact path."""
        if path not in self.artifacts:
            self.artifacts.append(path)
    
    def __str__(self) -> str:
        return f"Experiment({self.id}, {self.model_name}, {self.quant_method})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Comparison:
    """
    Comparison between multiple experiments.
    """
    
    experiments: List[Experiment]
    metrics_diff: Dict[str, List[float]] = field(default_factory=dict)
    
    def get_best_by_metric(self, metric: str, lower_is_better: bool = True) -> Optional[Experiment]:
        """Get the experiment with the best value for a metric."""
        valid_experiments = [
            exp for exp in self.experiments
            if metric in exp.metrics
        ]
        
        if not valid_experiments:
            return None
        
        if lower_is_better:
            return min(valid_experiments, key=lambda e: e.metrics[metric])
        else:
            return max(valid_experiments, key=lambda e: e.metrics[metric])
    
    def to_table(self, metrics: Optional[List[str]] = None) -> str:
        """Generate a text table for comparison."""
        if metrics is None:
            # Use common metrics
            metrics = ["latency_mean_ms", "memory_mb", "throughput_tps"]
        
        # Header
        header = ["ID", "Model", "Quant"] + metrics
        rows = [header]
        
        # Data rows
        for exp in self.experiments:
            row = [
                exp.id,
                exp.model_name.split("/")[-1][:15],
                exp.quant_method,
            ]
            for m in metrics:
                val = exp.metrics.get(m)
                if val is not None:
                    if isinstance(val, float):
                        row.append(f"{val:.2f}")
                    else:
                        row.append(str(val))
                else:
                    row.append("N/A")
            rows.append(row)
        
        # Format as table
        col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
        
        lines = []
        for row in rows:
            line = " | ".join(
                str(cell).ljust(col_widths[i])
                for i, cell in enumerate(row)
            )
            lines.append(line)
        
        # Add separator after header
        separator = "-+-".join("-" * w for w in col_widths)
        lines.insert(1, separator)
        
        return "\n".join(lines)
