"""
Consumer Inference Comparison

This example demonstrates all consumer-focused quantization strategies
available in QuantLab, showing their configurations and trade-offs.

Strategies covered:
1. GGUF (llama.cpp) - CPU/GPU optimized, various quant types
2. EXL2 (ExLlamaV2) - CUDA GPU optimized, mixed precision

No model loading required - this is a demonstration of the APIs.
"""

import sys
from pathlib import Path

# Add parent directory to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.quantization import (
    GGUFStrategy,
    ExLlamaV2Strategy,
    list_strategies,
    get_strategy,
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    print(f"\n--- {title} ---")


def compare_strategies():
    """Compare all consumer inference strategies."""
    
    print_header("Consumer Inference Strategy Comparison")
    
    # Show all available strategies
    all_strategies = list_strategies()
    consumer_strategies = ["gguf", "exl2"]
    
    print(f"\nAll available strategies: {all_strategies}")
    print(f"Consumer inference strategies: {consumer_strategies}")
    
    print_section("Strategy Comparison Table")
    
    print(f"\n{'Strategy':<12} {'Format':<10} {'Backend':<15} {'Hardware':<15} {'Use Case':<25}")
    print("-" * 77)
    print(f"{'gguf':<12} {'GGUF':<10} {'llama.cpp':<15} {'CPU/GPU':<15} {'Local deployment':<25}")
    print(f"{'exl2':<12} {'EXL2':<10} {'ExLlamaV2':<15} {'CUDA GPU':<15} {'High-perf GPU inference':<25}")
    
    # GGUF Details
    print_header("GGUF (llama.cpp) Details")
    
    print_section("Supported Quantization Types")
    gguf_types = [
        ("Q2_K", "2-bit", "Extreme compression, quality loss"),
        ("Q3_K_M", "3-bit", "Medium quality"),
        ("Q4_K_M", "4-bit", "★ Recommended balance"),
        ("Q5_K_M", "5-bit", "Good quality"),
        ("Q6_K", "6-bit", "Near-lossless"),
        ("Q8_0", "8-bit", "Minimal quality loss"),
    ]
    
    print(f"\n{'Type':<12} {'Bits':<8} {'Memory':<15} {'Notes':<30}")
    print("-" * 65)
    
    for qtype, bits, notes in gguf_types:
        strategy = GGUFStrategy(quant_type=qtype)
        reduction = strategy.estimate_memory_reduction()
        mem_str = f"{reduction:.3f}x FP16"
        print(f"{qtype:<12} {bits:<8} {mem_str:<15} {notes:<30}")
    
    print_section("Configuration Options")
    print("""
    GGUFStrategy(
        quant_type="Q4_K_M",    # Quantization type
        n_gpu_layers=-1,        # GPU layers (-1 = all, 0 = CPU only)
        n_ctx=2048,             # Context window size
        n_batch=512,            # Batch size for prompt processing
        verbose=False,          # Enable verbose logging
    )
    """)
    
    print_section("Installation")
    print("""
    # CPU only:
    pip install llama-cpp-python
    
    # With CUDA (recommended for GPU):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
    
    # With Metal (macOS):
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall
    """)
    
    # ExLlamaV2 Details
    print_header("ExLlamaV2 (EXL2) Details")
    
    print_section("Bit Width Options")
    exl2_bits = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]
    
    print(f"\n{'Bits':<10} {'Memory':<15} {'Quality':<20}")
    print("-" * 45)
    
    for bits in exl2_bits:
        strategy = ExLlamaV2Strategy(bits=bits)
        reduction = strategy.estimate_memory_reduction()
        mem_str = f"{reduction:.3f}x FP16"
        
        if bits <= 2.5:
            quality = "Poor"
        elif bits <= 3.5:
            quality = "Moderate"
        elif bits <= 4.5:
            quality = "Good"
        elif bits <= 6.0:
            quality = "Very Good"
        else:
            quality = "Excellent"
        
        marker = " ★" if bits == 4.0 else ""
        print(f"{bits:<10.1f} {mem_str:<15} {quality:<20}{marker}")
    
    print_section("Configuration Options")
    print("""
    ExLlamaV2Strategy(
        bits=4.0,              # Target average bit width (2.0-8.0)
        max_seq_len=4096,      # Maximum sequence length
        rope_scale=1.0,        # RoPE scaling factor
        rope_alpha=1.0,        # RoPE alpha for NTK-aware scaling
        no_flash_attn=False,   # Disable flash attention
    )
    """)
    
    print_section("Installation")
    print("""
    # Requires CUDA GPU
    pip install exllamav2
    """)
    
    # Comparing use cases
    print_header("When to Use Each Strategy")
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │ Use GGUF when:                                                  │
    │   • Running on CPU or mixed CPU/GPU                             │
    │   • Memory is limited (laptop, consumer PC)                     │
    │   • Need cross-platform compatibility                           │
    │   • Want easy model swapping (single .gguf file)                │
    │   • Running on macOS with Apple Silicon (Metal support)         │
    ├─────────────────────────────────────────────────────────────────┤
    │ Use EXL2 when:                                                  │
    │   • Have NVIDIA GPU with CUDA                                   │
    │   • Need maximum inference speed                                │
    │   • Want fine-grained control over quantization per layer       │
    │   • Building high-throughput inference servers                  │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print_header("Quick Start Examples")
    
    print("""
    # GGUF Example:
    from quantlab.quantization import GGUFStrategy
    
    strategy = GGUFStrategy(quant_type="Q4_K_M", n_gpu_layers=-1)
    model = strategy.load_model("path/to/model.Q4_K_M.gguf")
    
    # EXL2 Example:
    from quantlab.quantization import ExLlamaV2Strategy
    
    strategy = ExLlamaV2Strategy(bits=4.0)
    result = strategy.load_model("path/to/exl2-model")
    model = result["model"]
    """)


def main():
    compare_strategies()
    print("\n" + "=" * 70)
    print(" Run individual examples for more details:")
    print("   python examples/gguf_example.py --demo-only")
    print("   python examples/exllamav2_example.py --demo-only")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
