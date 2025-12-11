"""
Low VRAM Usage Example

Demonstrates techniques for running larger models on limited VRAM (4GB):
1. 4-bit quantization
2. CPU offloading
3. Memory-efficient loading
"""

import sys
from pathlib import Path
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from quantlab import QuantLab
from quantlab.config import ExperimentConfig, QuantizationConfig, HardwareConfig


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_model_with_constraints(
    model_name: str,
    quantization: str = "nf4",
    max_memory_gb: float = 3.5,
):
    """
    Test a model with strict memory constraints.
    
    Args:
        model_name: Model to test
        quantization: Quantization method
        max_memory_gb: Maximum GPU memory to use
    """
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print(f"Quantization: {quantization}")
    print(f"Max GPU Memory: {max_memory_gb} GB")
    print("=" * 50)
    
    clear_memory()
    
    # Configure for low VRAM
    config = ExperimentConfig(
        model_name=model_name,
        hardware=HardwareConfig(
            device="auto",
            gpu_memory_limit=max_memory_gb,
            cpu_offload=True,
        ),
        quantization=QuantizationConfig(
            method=quantization,
            double_quant=True,  # Extra compression
        ),
    )
    
    lab = QuantLab()
    
    try:
        result = lab.run_experiment(
            model_name=model_name,
            quantization=quantization,
            config=config,
            name=f"{model_name.split('/')[-1]}-low-vram",
            tags=["low-vram", "4gb"],
            run_benchmark=True,
        )
        
        print(f"\n✓ Success!")
        print(f"  Memory: {result.metrics.get('memory_mb', 'N/A'):.1f} MB")
        print(f"  GPU Allocated: {result.metrics.get('gpu_allocated_mb', 'N/A'):.1f} MB")
        print(f"  Latency: {result.metrics.get('latency_mean_ms', 'N/A'):.1f} ms")
        
        return result
        
    except torch.cuda.OutOfMemoryError:
        print("\n✗ Out of memory! Try:")
        print("  - More aggressive quantization (int4, nf4)")
        print("  - Smaller model variant")
        print("  - Lower batch size")
        return None
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None
    
    finally:
        clear_memory()


def main():
    """Test various models with 4GB VRAM constraint."""
    
    print("=" * 60)
    print("Low VRAM (4GB) Model Testing")
    print("=" * 60)
    
    # Models to test (smallest to largest)
    test_cases = [
        # Easy - should work in FP16
        ("facebook/opt-125m", "fp16"),
        
        # Medium - needs INT8
        ("facebook/opt-350m", "int8"),
        
        # Challenging - needs NF4
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "nf4"),
        
        # Difficult - needs NF4 + offload
        # ("microsoft/phi-2", "nf4"),  # Uncomment if you want to push limits
    ]
    
    results = []
    
    for model_name, quant_method in test_cases:
        result = test_model_with_constraints(
            model_name=model_name,
            quantization=quant_method,
            max_memory_gb=3.5,  # Leave some headroom
        )
        
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print(f"\nSuccessful: {len(results)} / {len(test_cases)}")
    
    for result in results:
        memory = result.metrics.get("memory_mb", 0)
        print(f"  {result.model_name.split('/')[-1]}: {memory:.0f} MB")
    
    print("\nRecommendations for 4GB VRAM:")
    print("  - Up to 350M params: FP16 or INT8")
    print("  - 500M - 1.5B params: NF4 or INT4")
    print("  - 1.5B - 3B params: NF4 + CPU offload")
    print("  - 3B+ params: May need more aggressive techniques")


if __name__ == "__main__":
    main()
