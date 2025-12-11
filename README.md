# QuantLab - LLM Quantization Experimentation Platform

A comprehensive, modular platform for rapid experimentation with model quantization strategies. Designed for research on consumer hardware with scalability to larger systems.

## Features

-  **Unified Model Interface**: Load any HuggingFace model with consistent API
-  **Multiple Quantization Strategies**: FP16, INT8, INT4, NF4, GPTQ, AWQ
-  **Comprehensive Benchmarking**: Latency, throughput, memory, accuracy
-  **Experiment Versioning**: Track and compare all experiments
-  **CLI + Web Dashboard**: Browse and visualize results
-  **Extensible**: Add custom quantizers, benchmarks, and evaluations

## Installation

```bash
pip install -r requirements.txt
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
├── quantization/    # Quantization strategies
├── pipeline/        # Orchestration pipeline
├── benchmark/       # Performance measurement
├── storage/         # Experiment storage
└── dashboard/       # CLI and web UI
```

## License

MIT License
