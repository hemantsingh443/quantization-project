"""
GGUF (llama.cpp) Example

This example demonstrates loading and using GGUF quantized models
for high-performance CPU/GPU inference using llama-cpp-python.

Prerequisites:
    pip install llama-cpp-python
    
    For GPU support (NVIDIA):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

GGUF models can be downloaded from HuggingFace, e.g.:
    https://huggingface.co/TheBloke/Llama-2-7B-GGUF
"""

import sys
from pathlib import Path

# Add parent directory to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.quantization import GGUFStrategy, list_strategies


def check_dependencies():
    """Check if llama-cpp-python is installed."""
    try:
        from llama_cpp import Llama
        return True
    except ImportError:
        return False


def demonstrate_strategy_api():
    """Show GGUFStrategy API without requiring actual model."""
    print("=" * 60)
    print("GGUF Strategy API Demonstration")
    print("=" * 60)
    
    # Show available strategies
    strategies = list_strategies()
    print(f"\nAvailable strategies: {strategies}")
    print(f"GGUF available: {'gguf' in strategies}")
    
    # Create strategies with different quant types
    print("\n--- Memory Reduction Estimates ---")
    quant_types = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    
    for qtype in quant_types:
        strategy = GGUFStrategy(quant_type=qtype)
        reduction = strategy.estimate_memory_reduction()
        reduction_pct = (1 - reduction) * 100
        print(f"  {qtype:8s}: {reduction:.3f}x FP16 ({reduction_pct:.1f}% reduction)")
    
    # Show configuration options
    print("\n--- Configuration Example ---")
    strategy = GGUFStrategy(
        quant_type="Q4_K_M",  # Recommended balance of quality/size
        n_gpu_layers=-1,      # -1 = all layers on GPU
        n_ctx=4096,           # Context window size
        n_batch=512,          # Batch size for prompt processing
        verbose=False,
    )
    
    config = strategy.get_load_config()
    print(f"  Quant type: {strategy.quant_type}")
    print(f"  Load config: {config}")
    print(f"  Requires calibration: {strategy.requires_calibration}")
    
    # Hardware validation
    print("\n--- Hardware Validation ---")
    warnings = strategy.validate_hardware()
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  ✓ All requirements met")


def run_inference(model_path: str):
    """Run inference with a GGUF model."""
    print("\n" + "=" * 60)
    print("GGUF Inference Example")
    print("=" * 60)
    
    # Create strategy
    strategy = GGUFStrategy(
        quant_type="Q4_K_M",
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=2048,
    )
    
    print(f"\nLoading model: {model_path}")
    print(f"Quant type: {strategy.quant_type}")
    
    try:
        # Load model
        model = strategy.load_model(model_path)
        
        # Run inference
        prompt = "The capital of France is"
        print(f"\nPrompt: {prompt}")
        
        output = model(
            prompt,
            max_tokens=32,
            temperature=0.7,
            top_p=0.9,
            echo=False,
        )
        
        response = output["choices"][0]["text"]
        print(f"Response: {response}")
        
        # Show stats
        print(f"\n--- Generation Stats ---")
        print(f"  Tokens generated: {output['usage']['completion_tokens']}")
        print(f"  Total tokens: {output['usage']['total_tokens']}")
        
    except ImportError as e:
        print(f"\n Error: {e}")
        print("\nTo install llama-cpp-python:")
        print("  pip install llama-cpp-python")
        print("\nFor GPU support:")
        print('  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF Example")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to GGUF model file (e.g., model.Q4_K_M.gguf)"
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only show API demonstration (no model loading)"
    )
    
    args = parser.parse_args()
    
    # Always show API demo
    demonstrate_strategy_api()
    
    # Check if we should run inference
    if args.demo_only:
        print("\n[Demo-only mode - skipping inference]")
        return
    
    if not check_dependencies():
        print("\n" + "=" * 60)
        print("llama-cpp-python Not Installed")
        print("=" * 60)
        print("\nTo enable GGUF support, install llama-cpp-python:")
        print("  pip install llama-cpp-python")
        print("\nFor GPU acceleration (recommended):")
        print('  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall')
        return
    
    if args.model:
        run_inference(args.model)
    else:
        print("\n" + "-" * 60)
        print("To run inference, provide a GGUF model path:")
        print("  python gguf_example.py --model /path/to/model.gguf")
        print("\nDownload GGUF models from HuggingFace:")
        print("  https://huggingface.co/TheBloke")
        print("-" * 60)


if __name__ == "__main__":
    main()
