# QuantLab - LLM Quantization Experimentation Platform

A comprehensive, modular platform for rapid experimentation with model quantization strategies. Designed for research on consumer hardware with scalability to larger systems.

## Features

-  **Unified Model Interface**: Load any HuggingFace model with consistent API
-  **Multiple Quantization Strategies**: FP16, INT8, INT4, NF4, GPTQ, AWQ
-  **Consumer Inference Formats**: GGUF (llama.cpp) and ExLlamaV2 (EXL2) support
-  **Comprehensive Benchmarking**: Latency, throughput, memory, accuracy
-  **Experiment Versioning**: Track and compare all experiments
-  **CLI + Web Dashboard**: Browse and visualize results
-  **Extensible**: Add custom quantizers, benchmarks, and evaluations

## Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For consumer inference formats:

```bash
# GGUF (llama.cpp) - CPU/GPU inference
pip install llama-cpp-python

# With CUDA GPU support:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

# ExLlamaV2 (EXL2) - High-performance CUDA inference
pip install exllamav2
```

## Quick Start

```python
from quantlab import QuantLab

# Initialize the platform
lab = QuantLab()

# Run a quantization experiment
result = lab.run_experiment(
    model_name="facebook/opt-125m",
    quantization="int8",
    eval_suite="simple"
)

# Compare with baseline
lab.compare(result.id, baseline="fp16")
```

## Consumer Inference Formats

QuantLab now supports high-performance consumer inference formats for local deployment.

### GGUF (llama.cpp)

Gold standard for CPU/GPU inference on consumer hardware:

```python
from quantlab.quantization import GGUFStrategy

# Create strategy with Q4_K_M quantization (recommended)
strategy = GGUFStrategy(
    quant_type="Q4_K_M",  # Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0
    n_gpu_layers=-1,      # -1 = all layers on GPU
    n_ctx=2048,
)

# Load and run inference
model = strategy.load_model("path/to/model.Q4_K_M.gguf")
```

**Supported quant types:**
| Type | Bits | Memory | Quality |
|------|------|--------|---------|
| Q2_K | 2-bit | 0.125x | Poor |
| Q4_K_M | 4-bit | 0.25x | Good ★ |
| Q5_K_M | 5-bit | 0.31x | Very Good |
| Q8_0 | 8-bit | 0.5x | Excellent |

### ExLlamaV2 (EXL2)

Optimized for high-performance CUDA GPU inference:

```python
from quantlab.quantization import ExLlamaV2Strategy

# Create strategy with 4-bit average quantization
strategy = ExLlamaV2Strategy(
    bits=4.0,          # 2.0 to 8.0
    max_seq_len=4096,
)

# Load model
result = strategy.load_model("path/to/exl2-model")
model = result["model"]
```

### Running Examples

```bash
# Compare all consumer inference strategies
python examples/consumer_inference_comparison.py

# GGUF demo
python examples/gguf_example.py --demo-only

# ExLlamaV2 demo  
python examples/exllamav2_example.py --demo-only
```

## CLI Usage

```bash
# Install the package first
pip install -e .

# Then use CLI commands
quantlab run facebook/opt-125m --quant int8
quantlab list-experiments --limit 10
quantlab compare <exp_id_1> <exp_id_2>
quantlab sweep gpt2 --methods "fp16,int8,nf4"

# Or run via python module (without pip install)
python -m quantlab.dashboard.cli run facebook/opt-125m --quant int8
python -m quantlab.dashboard.cli list-experiments --limit 10
```

### Example Output

```
$ quantlab list-experiments --limit 10
                                        Experiments                                        
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ ID       ┃ Model    ┃ Quant ┃ Status    ┃ Memory (MB) ┃ Latency (ms) ┃ Timestamp        ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 0d68d2e1 │ opt-125m │ nf4   │ completed │       117.4 │      22150.5 │ 2025-12-11 16:51 │
│ 77f8ef71 │ opt-125m │ int8  │ completed │       157.9 │      26384.9 │ 2025-12-11 16:22 │
│ 4ce44066 │ opt-125m │ fp16  │ completed │       238.9 │       5132.4 │ 2025-12-11 16:12 │
└──────────┴──────────┴───────┴───────────┴─────────────┴──────────────┴──────────────────┘
```

## Quantization Strategies

| Strategy | Type | Format | Hardware | Use Case |
|----------|------|--------|----------|----------|
| fp16 | Load-time | PyTorch | GPU | Baseline |
| int8 | Load-time | BitsAndBytes | GPU | Memory savings |
| nf4 | Load-time | BitsAndBytes | GPU | Best quality/size |
| gptq | PTQ | AutoGPTQ | GPU | Production |
| awq | PTQ | AutoAWQ | GPU | Production |
| **gguf** | Pre-quant | llama.cpp | CPU/GPU | Consumer inference |
| **exl2** | Pre-quant | ExLlamaV2 | CUDA GPU | High-performance |

## Supported Models (Tested on 4GB VRAM)

| Model | Size | Quantization | Notes |
|-------|------|--------------|-------|
| GPT-2 | 124M | FP16/INT8 | Full precision OK |
| GPT-2 Medium | 355M | FP16/INT8 | Full precision OK |
| OPT-125M | 125M | FP16/INT8 | Full precision OK |
| OPT-350M | 350M | INT8/INT4 | Use 4-bit for safety |
| TinyLLaMA | 1.1B | INT4/NF4 | 4-bit required |
| Phi-2 | 2.7B | INT4+offload | CPU offload needed |
| Gemma 2B | 2B | INT4+offload | CPU offload needed |

The platform also includes:
- low_vram.yaml config for memory-constrained GPUs
- low_vram_example.py demonstrating optimal settings
- Automatic CPU offloading support

## Project Structure

```
quantlab/
├── models/          # Model registry and loaders
├── quantization/    # Quantization strategies (incl. GGUF, EXL2)
├── pipeline/        # Orchestration pipeline
├── benchmark/       # Performance measurement
├── storage/         # Experiment storage
└── dashboard/       # CLI and web UI

examples/
├── basic_quantization.py       # Core workflow demo
├── consumer_inference_comparison.py  # GGUF vs EXL2 comparison
├── gguf_example.py             # GGUF usage demo
├── exllamav2_example.py        # ExLlamaV2 usage demo
└── low_vram_example.py         # Memory-constrained setup

tests/
└── test_components.py    # 46 unit tests
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=quantlab
```

## License

MIT License

