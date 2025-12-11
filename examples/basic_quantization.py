"""
Basic Quantization Example

This example demonstrates the core workflow:
1. Load a small model (OPT-125M)
2. Apply different quantization methods
3. Benchmark each variant
4. Compare results

Designed to work on consumer GPUs with limited VRAM.
"""

import sys
from pathlib import Path

# Add parent directory to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab import QuantLab


def main():
    print("=" * 60)
    print("QuantLab - Basic Quantization Example")
    print("=" * 60)
    
    # Initialize QuantLab
    lab = QuantLab()
    
    # Use a small model that fits in 4GB VRAM
    model_name = "facebook/opt-125m"
    
    print(f"\nModel: {model_name}")
    print("\nRunning experiments with different quantization methods...")
    print("-" * 60)
    
    # Quantization methods to test
    methods = ["fp16", "int8", "nf4"]
    
    experiments = []
    
    for method in methods:
        print(f"\n[{method.upper()}] Starting...")
        
        try:
            result = lab.run_experiment(
                model_name=model_name,
                quantization=method,
                name=f"opt-125m-{method}",
                tags=["basic-example", "opt"],
                run_benchmark=True,
                run_eval=False,
            )
            
            experiments.append(result)
            
            # Print summary
            memory = result.metrics.get("memory_mb", "N/A")
            latency = result.metrics.get("latency_mean_ms", "N/A")
            
            print(f"[{method.upper()}] Complete!")
            print(f"  Memory: {memory:.1f} MB" if isinstance(memory, float) else f"  Memory: {memory}")
            print(f"  Latency: {latency:.1f} ms" if isinstance(latency, float) else f"  Latency: {latency}")
            print(f"  Experiment ID: {result.id}")
            
        except Exception as e:
            print(f"[{method.upper()}] Failed: {e}")
    
    # Compare results
    if len(experiments) >= 2:
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        
        comparison = lab.compare([e.id for e in experiments])
        
        print(f"\n{'Method':<10} {'Memory (MB)':<15} {'Latency (ms)':<15}")
        print("-" * 40)
        
        for i, exp in enumerate(experiments):
            method = exp.quant_method
            memory = comparison["metrics"].get("memory_mb", [None])[i]
            latency = comparison["metrics"].get("latency_mean_ms", [None])[i]
            
            mem_str = f"{memory:.1f}" if memory else "N/A"
            lat_str = f"{latency:.1f}" if latency else "N/A"
            
            print(f"{method:<10} {mem_str:<15} {lat_str:<15}")
    
    print("\n" + "=" * 60)
    print("Done! Use 'quantlab list' to see all experiments.")
    print("=" * 60)


if __name__ == "__main__":
    main()
