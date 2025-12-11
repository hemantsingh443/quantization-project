"""
Model Comparison Workflow

This example shows how to:
1. Run experiments on multiple models
2. Compare across quantization methods
3. Generate a comparison report
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab import QuantLab
from quantlab.dashboard.plots import create_comparison_report


def main():
    print("=" * 60)
    print("QuantLab - Model Comparison Workflow")
    print("=" * 60)
    
    lab = QuantLab()
    
    # Models suitable for 4GB VRAM
    models = [
        "facebook/opt-125m",
        "gpt2",
    ]
    
    # Quantization methods
    methods = ["fp16", "int8"]
    
    all_experiments = []
    
    for model in models:
        print(f"\n{'='*40}")
        print(f"Testing: {model}")
        print("=" * 40)
        
        for method in methods:
            print(f"\n  [{method}] Running...")
            
            try:
                result = lab.run_experiment(
                    model_name=model,
                    quantization=method,
                    name=f"{model.split('/')[-1]}-{method}",
                    tags=["comparison", "multi-model"],
                    run_benchmark=True,
                    run_eval=False,
                )
                
                all_experiments.append(result)
                
                memory = result.metrics.get("memory_mb", "N/A")
                print(f"  [{method}] Done - Memory: {memory:.1f} MB" if isinstance(memory, float) else f"  [{method}] Done - Memory: {memory}")
                
            except Exception as e:
                print(f"  [{method}] Failed: {e}")
    
    # Generate comparison report
    if all_experiments:
        print("\n" + "=" * 60)
        print("Generating Comparison Report")
        print("=" * 60)
        
        # Prepare experiment data for report
        exp_data = []
        for exp in all_experiments:
            data = exp.to_dict()
            data.update(exp.metrics)
            exp_data.append(data)
        
        report_dir = Path("./experiments/reports")
        report_path = create_comparison_report(
            experiments=exp_data,
            output_dir=report_dir,
            include_plots=True,
        )
        
        print(f"\nReport generated: {report_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)
    
    print(f"\n{'Model':<20} {'Method':<10} {'Memory (MB)':<12} {'Latency (ms)':<12}")
    print("-" * 55)
    
    for exp in all_experiments:
        model_short = exp.model_name.split("/")[-1][:18]
        memory = exp.metrics.get("memory_mb")
        latency = exp.metrics.get("latency_mean_ms")
        
        print(
            f"{model_short:<20} {exp.quant_method:<10} "
            f"{memory:.1f if memory else 'N/A':<12} "
            f"{latency:.1f if latency else 'N/A':<12}"
        )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
