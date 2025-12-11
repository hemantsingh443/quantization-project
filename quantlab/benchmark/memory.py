"""
Memory benchmarking for model footprint analysis.
"""

import logging
from typing import Dict, Any, Optional

import torch
from torch import nn


logger = logging.getLogger(__name__)


class MemoryBenchmark:
    """
    Measures model memory footprint.
    
    Captures:
    - Model parameter memory
    - Buffer memory
    - GPU memory usage (allocated and reserved)
    - Peak memory during inference
    """
    
    def run(
        self,
        model: nn.Module,
        include_inference_peak: bool = True,
    ) -> Dict[str, Any]:
        """
        Run memory benchmark.
        
        Args:
            model: Model to measure
            include_inference_peak: Whether to measure peak memory during inference
            
        Returns:
            Dictionary with memory metrics in MB
        """
        results = {}
        
        # Model parameter memory
        param_memory = self._calculate_param_memory(model)
        results.update(param_memory)
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_memory = self._get_gpu_memory()
            results.update(gpu_memory)
        
        # Memory by dtype
        dtype_breakdown = self._memory_by_dtype(model)
        results["memory_by_dtype"] = dtype_breakdown
        
        return results
    
    def _calculate_param_memory(self, model: nn.Module) -> Dict[str, float]:
        """Calculate memory used by parameters and buffers."""
        param_bytes = 0
        buffer_bytes = 0
        
        for param in model.parameters():
            param_bytes += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_bytes += buffer.numel() * buffer.element_size()
        
        total_bytes = param_bytes + buffer_bytes
        
        return {
            "param_memory_mb": param_bytes / (1024 * 1024),
            "buffer_memory_mb": buffer_bytes / (1024 * 1024),
            "total_model_memory_mb": total_bytes / (1024 * 1024),
            "memory_mb": total_bytes / (1024 * 1024),  # Alias for convenience
        }
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        # Force garbage collection
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Get device properties
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024 * 1024)
        
        return {
            "gpu_allocated_mb": allocated,
            "gpu_reserved_mb": reserved,
            "gpu_max_allocated_mb": max_allocated,
            "gpu_total_mb": total_memory,
            "gpu_utilization_pct": (allocated / total_memory) * 100 if total_memory > 0 else 0,
        }
    
    def _memory_by_dtype(self, model: nn.Module) -> Dict[str, float]:
        """Break down memory usage by data type."""
        dtype_bytes = {}
        
        for param in model.parameters():
            dtype_str = str(param.dtype)
            if dtype_str not in dtype_bytes:
                dtype_bytes[dtype_str] = 0
            dtype_bytes[dtype_str] += param.numel() * param.element_size()
        
        # Convert to MB
        return {
            dtype: bytes_val / (1024 * 1024)
            for dtype, bytes_val in dtype_bytes.items()
        }
    
    def measure_inference_peak(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measure peak memory during inference.
        
        Args:
            model: Model to measure
            input_ids: Sample input tensor
            
        Returns:
            Peak memory metrics
        """
        if not torch.cuda.is_available():
            return {}
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Run inference
        with torch.no_grad():
            _ = model(input_ids)
        
        torch.cuda.synchronize()
        
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        
        return {
            "inference_peak_allocated_mb": peak_allocated,
            "inference_peak_reserved_mb": peak_reserved,
        }


def get_memory_summary(model: nn.Module) -> str:
    """Get a formatted memory summary string."""
    benchmark = MemoryBenchmark()
    results = benchmark.run(model, include_inference_peak=False)
    
    lines = [
        "Memory Summary:",
        f"  Parameters: {results.get('param_memory_mb', 0):.1f} MB",
        f"  Buffers: {results.get('buffer_memory_mb', 0):.1f} MB",
        f"  Total Model: {results.get('total_model_memory_mb', 0):.1f} MB",
    ]
    
    if "gpu_allocated_mb" in results:
        lines.extend([
            f"  GPU Allocated: {results['gpu_allocated_mb']:.1f} MB",
            f"  GPU Utilization: {results.get('gpu_utilization_pct', 0):.1f}%",
        ])
    
    dtype_breakdown = results.get("memory_by_dtype", {})
    if dtype_breakdown:
        lines.append("  By dtype:")
        for dtype, mb in dtype_breakdown.items():
            lines.append(f"    {dtype}: {mb:.1f} MB")
    
    return "\n".join(lines)


def check_memory_available(required_mb: float) -> bool:
    """Check if enough GPU memory is available."""
    if not torch.cuda.is_available():
        return True  # Assume CPU has enough
    
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024 * 1024)
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    
    available = total - allocated
    return available >= required_mb
