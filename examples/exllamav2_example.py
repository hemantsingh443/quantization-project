"""
ExLlamaV2 (EXL2) Example

This example demonstrates loading and using EXL2 quantized models
for high-performance GPU inference using ExLlamaV2.

Prerequisites:
    pip install exllamav2
    
    Requires CUDA GPU!

EXL2 models can be found on HuggingFace, e.g.:
    https://huggingface.co/turboderp
"""

import sys
from pathlib import Path

# Add parent directory to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.quantization import ExLlamaV2Strategy, list_strategies


def check_dependencies():
    """Check if exllamav2 is installed."""
    try:
        import exllamav2
        return True
    except ImportError:
        return False


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def demonstrate_strategy_api():
    """Show ExLlamaV2Strategy API without requiring actual model."""
    print("=" * 60)
    print("ExLlamaV2 (EXL2) Strategy API Demonstration")
    print("=" * 60)
    
    # Show available strategies
    strategies = list_strategies()
    print(f"\nAvailable strategies: {strategies}")
    print(f"EXL2 available: {'exl2' in strategies}")
    
    # Create strategies with different bit widths
    print("\n--- Memory Reduction Estimates ---")
    bit_widths = [2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0]
    
    for bits in bit_widths:
        strategy = ExLlamaV2Strategy(bits=bits)
        reduction = strategy.estimate_memory_reduction()
        reduction_pct = (1 - reduction) * 100
        print(f"  {bits:.1f}-bit: {reduction:.3f}x FP16 ({reduction_pct:.1f}% reduction)")
    
    # Show configuration options
    print("\n--- Configuration Example ---")
    strategy = ExLlamaV2Strategy(
        bits=4.0,             # 4-bit average quantization
        max_seq_len=4096,     # Maximum sequence length
        rope_scale=1.0,       # RoPE scaling factor
        rope_alpha=1.0,       # RoPE alpha value
        no_flash_attn=False,  # Use flash attention if available
    )
    
    config = strategy.get_load_config()
    print(f"  Bits: {strategy.bits}")
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
    
    # CUDA check
    print("\n--- CUDA Status ---")
    if check_cuda():
        import torch
        print(f"  ✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ✗ CUDA not available (ExLlamaV2 requires CUDA)")


def run_inference(model_path: str):
    """Run inference with an EXL2 model."""
    print("\n" + "=" * 60)
    print("ExLlamaV2 Inference Example")
    print("=" * 60)
    
    # Create strategy
    strategy = ExLlamaV2Strategy(
        bits=4.0,
        max_seq_len=2048,
    )
    
    print(f"\nLoading model: {model_path}")
    print(f"Target bits: {strategy.bits}")
    
    try:
        # Load model
        result = strategy.load_model(model_path)
        model = result["model"]
        cache = result["cache"]
        config = result["config"]
        
        # Create generator
        from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
        from exllamav2 import ExLlamaV2Tokenizer
        
        tokenizer = ExLlamaV2Tokenizer(config)
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        
        # Configure sampling
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.7
        settings.top_p = 0.9
        settings.top_k = 40
        
        # Run generation
        prompt = "The capital of France is"
        print(f"\nPrompt: {prompt}")
        
        output = generator.generate_simple(prompt, settings, num_tokens=32)
        
        print(f"Response: {output}")
        
    except ImportError as e:
        print(f"\n Error: {e}")
        print("\nTo install exllamav2:")
        print("  pip install exllamav2")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ExLlamaV2 Example")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to EXL2 model directory"
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
    
    if not check_cuda():
        print("\n" + "=" * 60)
        print("CUDA Not Available")
        print("=" * 60)
        print("\nExLlamaV2 requires a CUDA-capable GPU.")
        print("Please ensure you have:")
        print("  1. An NVIDIA GPU")
        print("  2. CUDA drivers installed")
        print("  3. PyTorch with CUDA support")
        return
    
    if not check_dependencies():
        print("\n" + "=" * 60)
        print("ExLlamaV2 Not Installed")
        print("=" * 60)
        print("\nTo enable EXL2 support, install exllamav2:")
        print("  pip install exllamav2")
        return
    
    if args.model:
        run_inference(args.model)
    else:
        print("\n" + "-" * 60)
        print("To run inference, provide an EXL2 model path:")
        print("  python exllamav2_example.py --model /path/to/exl2/model")
        print("\nDownload EXL2 models from HuggingFace:")
        print("  https://huggingface.co/turboderp")
        print("-" * 60)


if __name__ == "__main__":
    main()
