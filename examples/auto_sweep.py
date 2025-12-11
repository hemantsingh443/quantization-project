"""
Auto-Sweep: Automated Experimentation

This example demonstrates:
1. Automated sweep across quantization methods
2. Finding optimal settings for a given memory constraint
3. Batch experimentation with scriptable pipelines
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab import QuantLab
from quantlab.benchmark.memory import check_memory_available


def run_sweep(
    model_name: str,
    methods: list,
    memory_limit_mb: float = None,
):
    """
    Run a sweep of quantization methods on a model.
    
    Args:
        model_name: Model to test
        methods: List of quantization methods
        memory_limit_mb: Optional memory constraint
    """
    print("=" * 60)
    print(f"Auto-Sweep: {model_name}")
    print(f"Methods: {', '.join(methods)}")
    if memory_limit_mb:
        print(f"Memory Limit: {memory_limit_mb} MB")
    print("=" * 60)
    
    lab = QuantLab()
    
    results = []
    
    for method in methods:
        print(f"\n>>> Testing {method}...")
        
        try:
            result = lab.run_experiment(
                model_name=model_name,
                quantization=method,
                tags=["auto-sweep"],
                run_benchmark=True,
                run_eval=False,
            )
            
            results.append(result)
            
            memory = result.metrics.get("memory_mb", float("inf"))
            latency = result.metrics.get("latency_mean_ms", float("inf"))
            
            status = "✓"
            if memory_limit_mb and memory > memory_limit_mb:
                status = "⚠ (exceeds limit)"
            
            print(f"    {status} Memory: {memory:.1f} MB, Latency: {latency:.1f} ms")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("Sweep Analysis")
    print("=" * 60)
    
    if not results:
        print("No successful experiments.")
        return
    
    # Find best by memory
    valid_results = results
    if memory_limit_mb:
        valid_results = [
            r for r in results
            if r.metrics.get("memory_mb", float("inf")) <= memory_limit_mb
        ]
    
    if not valid_results:
        print(f"No methods fit within {memory_limit_mb} MB limit.")
        print("Consider using more aggressive quantization (int4, nf4).")
        return
    
    # Best by memory
    best_memory = min(valid_results, key=lambda r: r.metrics.get("memory_mb", float("inf")))
    print(f"\nBest Memory: {best_memory.quant_method}")
    print(f"  {best_memory.metrics.get('memory_mb', 'N/A'):.1f} MB")
    
    # Best by latency
    best_latency = min(valid_results, key=lambda r: r.metrics.get("latency_mean_ms", float("inf")))
    print(f"\nBest Latency: {best_latency.quant_method}")
    print(f"  {best_latency.metrics.get('latency_mean_ms', 'N/A'):.1f} ms")
    
    # Recommendation
    print("\n" + "-" * 40)
    print("Recommendation:")
    
    # Simple heuristic: best memory/latency tradeoff
    scores = []
    for r in valid_results:
        memory = r.metrics.get("memory_mb", float("inf"))
        latency = r.metrics.get("latency_mean_ms", float("inf"))
        # Normalize and combine (lower is better)
        score = (memory / 1000) + (latency / 1000)  # Rough normalization
        scores.append((r, score))
    
    scores.sort(key=lambda x: x[1])
    recommended = scores[0][0]
    
    print(f"  Method: {recommended.quant_method}")
    print(f"  Memory: {recommended.metrics.get('memory_mb', 'N/A'):.1f} MB")
    print(f"  Latency: {recommended.metrics.get('latency_mean_ms', 'N/A'):.1f} ms")
    print(f"  Experiment ID: {recommended.id}")
    
    return recommended


def main():
    parser = argparse.ArgumentParser(description="Auto-sweep quantization methods")
    parser.add_argument("model", help="Model name (e.g., facebook/opt-125m)")
    parser.add_argument(
        "--methods",
        default="fp16,int8,nf4",
        help="Comma-separated quantization methods"
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help="Maximum memory in MB"
    )
    
    args = parser.parse_args()
    
    methods = [m.strip() for m in args.methods.split(",")]
    
    run_sweep(
        model_name=args.model,
        methods=methods,
        memory_limit_mb=args.memory_limit,
    )


if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        # Default demo
        run_sweep(
            model_name="facebook/opt-125m",
            methods=["fp16", "int8", "nf4"],
            memory_limit_mb=500,  # 500 MB limit
        )
    else:
        main()
