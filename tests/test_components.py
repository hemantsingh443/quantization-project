"""
Unit tests for QuantLab components.

Run with: pytest tests/ -v
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
import tempfile
import shutil


class TestExperiment:
    """Tests for Experiment dataclass."""
    
    def test_experiment_creation(self):
        """Test creating an experiment."""
        from quantlab.storage.experiment import Experiment
        
        exp = Experiment(
            model_name="facebook/opt-125m",
            quant_method="int8",
            name="Test Experiment",
        )
        
        assert exp.model_name == "facebook/opt-125m"
        assert exp.quant_method == "int8"
        assert exp.name == "Test Experiment"
        assert exp.status == "pending"
        assert isinstance(exp.id, str)
        assert len(exp.id) == 8
    
    def test_experiment_to_dict(self):
        """Test serialization to dict."""
        from quantlab.storage.experiment import Experiment
        
        exp = Experiment(
            model_name="gpt2",
            quant_method="fp16",
            metrics={"memory_mb": 250.0, "latency_mean_ms": 100.0},
        )
        
        data = exp.to_dict()
        
        assert isinstance(data, dict)
        assert data["model_name"] == "gpt2"
        assert data["quant_method"] == "fp16"
        assert data["metrics"]["memory_mb"] == 250.0
    
    def test_experiment_from_dict(self):
        """Test deserialization from dict."""
        from quantlab.storage.experiment import Experiment
        
        data = {
            "id": "abc12345",
            "model_name": "facebook/opt-350m",
            "quant_method": "nf4",
            "status": "completed",
            "metrics": {"memory_mb": 150.0},
            "timestamp": "2025-01-01T10:00:00",
        }
        
        exp = Experiment.from_dict(data)
        
        assert exp.id == "abc12345"
        assert exp.model_name == "facebook/opt-350m"
        assert exp.quant_method == "nf4"
        assert exp.status == "completed"
        assert exp.metrics["memory_mb"] == 150.0
        assert isinstance(exp.timestamp, datetime)
    
    def test_experiment_summary(self):
        """Test summary generation."""
        from quantlab.storage.experiment import Experiment
        
        exp = Experiment(
            model_name="gpt2",
            quant_method="int8",
            model_size="124M",
            status="completed",
            metrics={
                "memory_mb": 120.0,
                "latency_mean_ms": 50.0,
                "throughput_tps": 20.0,
            },
        )
        
        summary = exp.summary()
        
        assert "gpt2" in summary
        assert "int8" in summary
        assert "124M" in summary
        assert "completed" in summary


class TestExperimentStore:
    """Tests for ExperimentStore."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        from quantlab.storage.store import ExperimentStore
        
        temp_dir = tempfile.mkdtemp()
        store = ExperimentStore(experiments_dir=Path(temp_dir))
        yield store
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_save_and_load(self, temp_store):
        """Test saving and loading an experiment."""
        from quantlab.storage.experiment import Experiment
        
        exp = Experiment(
            model_name="test-model",
            quant_method="fp16",
            status="completed",
        )
        
        # Save
        exp_id = temp_store.save_experiment(exp)
        assert exp_id == exp.id
        
        # Load
        loaded = temp_store.load_experiment(exp_id)
        assert loaded is not None
        assert loaded.model_name == "test-model"
        assert loaded.quant_method == "fp16"
    
    def test_list_experiments(self, temp_store):
        """Test listing experiments."""
        from quantlab.storage.experiment import Experiment
        
        # Create several experiments
        for i, method in enumerate(["fp16", "int8", "nf4"]):
            exp = Experiment(
                model_name=f"model-{i}",
                quant_method=method,
            )
            temp_store.save_experiment(exp)
        
        # List all
        all_exps = temp_store.list_experiments(limit=10)
        assert len(all_exps) == 3
        
        # Filter by quant method
        int8_exps = temp_store.list_experiments(quant_method="int8")
        assert len(int8_exps) == 1
        assert int8_exps[0].quant_method == "int8"
    
    def test_delete_experiment(self, temp_store):
        """Test deleting an experiment."""
        from quantlab.storage.experiment import Experiment
        
        exp = Experiment(model_name="to-delete", quant_method="fp16")
        temp_store.save_experiment(exp)
        
        # Verify it exists
        assert temp_store.load_experiment(exp.id) is not None
        
        # Delete
        result = temp_store.delete_experiment(exp.id)
        assert result is True
        
        # Verify it's gone
        assert temp_store.load_experiment(exp.id) is None
    
    def test_get_statistics(self, temp_store):
        """Test getting storage statistics."""
        from quantlab.storage.experiment import Experiment
        
        # Create experiments
        for method in ["fp16", "fp16", "int8"]:
            exp = Experiment(model_name="test", quant_method=method)
            temp_store.save_experiment(exp)
        
        stats = temp_store.get_statistics()
        
        assert stats["total_experiments"] == 3
        assert stats["by_quant_method"]["fp16"] == 2
        assert stats["by_quant_method"]["int8"] == 1


class TestQuantizationStrategies:
    """Tests for QuantizationStrategy classes."""
    
    def test_strategy_registry(self):
        """Test getting strategies from registry."""
        from quantlab.quantization import get_strategy, list_strategies
        
        strategies = list_strategies()
        assert "fp16" in strategies
        assert "int8" in strategies
        assert "nf4" in strategies
        
        fp16 = get_strategy("fp16")
        assert fp16 is not None
        assert fp16.is_load_time_quantization is True
    
    def test_fp16_strategy(self):
        """Test FP16Strategy."""
        from quantlab.quantization import FP16Strategy
        import torch
        
        strategy = FP16Strategy()
        
        assert strategy.is_load_time_quantization is True
        assert strategy.get_torch_dtype() == torch.float16
        assert strategy.estimate_memory_reduction() == 0.5
        assert strategy.get_load_config() is None  # No BNB config needed
    
    def test_nf4_strategy(self):
        """Test NF4Strategy."""
        from quantlab.quantization import NF4Strategy
        
        strategy = NF4Strategy(double_quant=True)
        
        assert strategy.is_load_time_quantization is True
        assert strategy.estimate_memory_reduction() == 0.25
        
        # Should return BitsAndBytesConfig
        config = strategy.get_load_config()
        assert config is not None
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
    
    def test_gptq_strategy(self):
        """Test GPTQStrategy is marked as PTQ."""
        from quantlab.quantization import GPTQStrategy
        
        strategy = GPTQStrategy(bits=4, group_size=128)
        
        assert strategy.is_load_time_quantization is False
        assert strategy.requires_calibration is True
    
    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        from quantlab.quantization import (
            QuantizationStrategy, 
            register_strategy, 
            get_strategy
        )
        
        class CustomStrategy(QuantizationStrategy):
            def __init__(self):
                super().__init__()
                self._is_load_time = True
        
        register_strategy("custom", CustomStrategy)
        
        custom = get_strategy("custom")
        assert custom is not None
        assert isinstance(custom, CustomStrategy)


class TestConfig:
    """Tests for configuration classes."""
    
    def test_quantization_config(self):
        """Test QuantizationConfig."""
        from quantlab.config import QuantizationConfig
        
        config = QuantizationConfig(
            method="nf4",
            bits=4,
            double_quant=True,
        )
        
        assert config.method == "nf4"
        assert config.bits == 4
        assert config.double_quant is True
    
    def test_experiment_config(self):
        """Test ExperimentConfig."""
        from quantlab.config import ExperimentConfig
        
        config = ExperimentConfig(model_name="gpt2")
        
        assert config.model_name == "gpt2"
        assert config.quantization is not None
        assert config.hardware is not None
        assert config.benchmark is not None


class TestComparison:
    """Tests for experiment comparison."""
    
    def test_comparison_to_table(self):
        """Test Comparison table generation."""
        from quantlab.storage.experiment import Experiment, Comparison
        
        exp1 = Experiment(
            model_name="gpt2",
            quant_method="fp16",
            metrics={"latency_mean_ms": 100, "memory_mb": 250},
        )
        exp2 = Experiment(
            model_name="gpt2", 
            quant_method="int8",
            metrics={"latency_mean_ms": 80, "memory_mb": 150},
        )
        
        comparison = Comparison(experiments=[exp1, exp2])
        table = comparison.to_table()
        
        assert "fp16" in table
        assert "int8" in table
    
    def test_get_best_by_metric(self):
        """Test finding best experiment by metric."""
        from quantlab.storage.experiment import Experiment, Comparison
        
        exp1 = Experiment(
            model_name="model1",
            quant_method="fp16",
            metrics={"memory_mb": 300},
        )
        exp2 = Experiment(
            model_name="model2",
            quant_method="int8",
            metrics={"memory_mb": 150},
        )
        
        comparison = Comparison(experiments=[exp1, exp2])
        
        best_memory = comparison.get_best_by_metric("memory_mb", lower_is_better=True)
        assert best_memory.quant_method == "int8"


class TestGGUFStrategy:
    """Tests for GGUF strategy (without requiring actual models)."""
    
    def test_gguf_strategy_initialization(self):
        """Test GGUF strategy with default parameters."""
        from quantlab.quantization import GGUFStrategy
        
        strategy = GGUFStrategy()
        
        assert strategy.quant_type == "Q4_K_M"
        assert strategy.n_gpu_layers == -1
        assert strategy.n_ctx == 2048
        assert strategy.n_batch == 512
        assert strategy._is_load_time is True
    
    def test_gguf_strategy_custom_params(self):
        """Test GGUF strategy with custom parameters."""
        from quantlab.quantization import GGUFStrategy
        
        strategy = GGUFStrategy(
            quant_type="Q8_0",
            n_gpu_layers=20,
            n_ctx=4096,
        )
        
        assert strategy.quant_type == "Q8_0"
        assert strategy.n_gpu_layers == 20
        assert strategy.n_ctx == 4096
    
    def test_gguf_memory_estimation(self):
        """Test memory reduction estimates for GGUF types."""
        from quantlab.quantization import GGUFStrategy
        
        # Q4_K_M should have 0.25 reduction (4-bit)
        q4 = GGUFStrategy(quant_type="Q4_K_M")
        assert q4.estimate_memory_reduction() == 0.25
        
        # Q8_0 should have 0.5 reduction (8-bit)
        q8 = GGUFStrategy(quant_type="Q8_0")
        assert q8.estimate_memory_reduction() == 0.5
        
        # Q2_K should have 0.125 reduction (2-bit)
        q2 = GGUFStrategy(quant_type="Q2_K")
        assert q2.estimate_memory_reduction() == 0.125
    
    def test_gguf_get_load_config(self):
        """Test GGUF load config generation."""
        from quantlab.quantization import GGUFStrategy
        
        strategy = GGUFStrategy(n_gpu_layers=10, n_ctx=1024, verbose=True)
        config = strategy.get_load_config()
        
        assert config["n_gpu_layers"] == 10
        assert config["n_ctx"] == 1024
        assert config["verbose"] is True
    
    def test_gguf_quant_type_normalization(self):
        """Test that quant type is normalized to uppercase."""
        from quantlab.quantization import GGUFStrategy
        
        strategy = GGUFStrategy(quant_type="q4_k_m")
        assert strategy.quant_type == "Q4_K_M"
    
    def test_gguf_requires_no_calibration(self):
        """Test that GGUF doesn't require calibration (pre-quantized)."""
        from quantlab.quantization import GGUFStrategy
        
        strategy = GGUFStrategy()
        assert strategy.requires_calibration is False


class TestExLlamaV2Strategy:
    """Tests for ExLlamaV2 strategy."""
    
    def test_exl2_strategy_initialization(self):
        """Test EXL2 strategy with default parameters."""
        from quantlab.quantization import ExLlamaV2Strategy
        
        strategy = ExLlamaV2Strategy()
        
        assert strategy.bits == 4.0
        assert strategy.max_seq_len == 4096
        assert strategy.rope_scale == 1.0
        assert strategy._is_load_time is True
    
    def test_exl2_strategy_custom_bits(self):
        """Test EXL2 strategy with custom bit widths."""
        from quantlab.quantization import ExLlamaV2Strategy
        
        strategy = ExLlamaV2Strategy(bits=3.5, max_seq_len=8192)
        
        assert strategy.bits == 3.5
        assert strategy.max_seq_len == 8192
    
    def test_exl2_memory_estimation(self):
        """Test memory reduction estimates."""
        from quantlab.quantization import ExLlamaV2Strategy
        
        # 4-bit should be 4/16 = 0.25
        exl2_4bit = ExLlamaV2Strategy(bits=4.0)
        assert exl2_4bit.estimate_memory_reduction() == 0.25
        
        # 8-bit should be 8/16 = 0.5
        exl2_8bit = ExLlamaV2Strategy(bits=8.0)
        assert exl2_8bit.estimate_memory_reduction() == 0.5
        
        # 3-bit should be 3/16 = 0.1875
        exl2_3bit = ExLlamaV2Strategy(bits=3.0)
        assert abs(exl2_3bit.estimate_memory_reduction() - 0.1875) < 0.001
    
    def test_exl2_get_load_config(self):
        """Test EXL2 load config generation."""
        from quantlab.quantization import ExLlamaV2Strategy
        
        strategy = ExLlamaV2Strategy(
            max_seq_len=2048,
            rope_scale=2.0,
            no_flash_attn=True,
        )
        config = strategy.get_load_config()
        
        assert config["max_seq_len"] == 2048
        assert config["rope_scale"] == 2.0
        assert config["no_flash_attn"] is True
    
    def test_exl2_requires_no_calibration(self):
        """Test that EXL2 doesn't require calibration (pre-quantized)."""
        from quantlab.quantization import ExLlamaV2Strategy
        
        strategy = ExLlamaV2Strategy()
        assert strategy.requires_calibration is False


class TestConsumerStrategiesInRegistry:
    """Tests for consumer strategies in the registry."""
    
    def test_gguf_in_registry(self):
        """Test that GGUF is available in strategy registry."""
        from quantlab.quantization import get_strategy, list_strategies
        
        strategies = list_strategies()
        assert "gguf" in strategies
        
        gguf = get_strategy("gguf")
        assert gguf is not None
    
    def test_exl2_in_registry(self):
        """Test that EXL2 is available in strategy registry."""
        from quantlab.quantization import get_strategy, list_strategies
        
        strategies = list_strategies()
        assert "exl2" in strategies
        
        exl2 = get_strategy("exl2")
        assert exl2 is not None
    
    def test_get_strategy_with_kwargs(self):
        """Test getting strategies with custom kwargs."""
        from quantlab.quantization import get_strategy
        
        gguf = get_strategy("gguf", quant_type="Q8_0", n_gpu_layers=10)
        assert gguf.quant_type == "Q8_0"
        assert gguf.n_gpu_layers == 10


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for testing."""
        from quantlab.models.registry import ModelRegistry
        return ModelRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry initializes with empty caches."""
        assert len(registry._loaded_models) == 0
        assert len(registry._custom_loaders) == 0
    
    def test_register_custom_loader(self, registry):
        """Test registering custom loaders."""
        def custom_loader(model_name, config):
            return ("mock_model", "mock_tokenizer")
        
        registry.register_loader("custom", custom_loader)
        
        assert "custom" in registry._custom_loaders
    
    def test_list_cached_models_empty(self, registry):
        """Test listing cached models when empty."""
        cached = registry.list_cached_models()
        assert cached == []
    
    def test_clear_cache(self, registry):
        """Test clearing the model cache."""
        # Add something to cache manually
        registry._loaded_models["test_key"] = ("model", "tokenizer")
        
        registry.clear_cache()
        
        assert len(registry._loaded_models) == 0


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""
    
    def test_benchmark_runner_initialization(self):
        """Test BenchmarkRunner initializes with all benchmarks."""
        from quantlab.benchmark.runner import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        assert hasattr(runner, "latency")
        assert hasattr(runner, "throughput")
        assert hasattr(runner, "memory")
        assert hasattr(runner, "deviation")
    
    def test_simple_eval_prompts_exist(self):
        """Test that simple eval has reasonable prompts."""
        from quantlab.benchmark.runner import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        # Access the _run_simple_eval method to check its content
        import inspect
        source = inspect.getsource(runner._run_simple_eval)
        
        assert "capital of France" in source
        assert "Water boils" in source
    
    def test_perplexity_texts_exist(self):
        """Test that perplexity eval has test texts."""
        from quantlab.benchmark.runner import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        import inspect
        source = inspect.getsource(runner._run_perplexity_eval)
        
        assert "quick brown fox" in source


class TestPipeline:
    """Tests for pipeline components."""
    
    def test_experiment_runner_import(self):
        """Test ExperimentRunner can be imported."""
        try:
            from quantlab.pipeline.runner import ExperimentRunner
            assert ExperimentRunner is not None
        except ImportError:
            pytest.skip("ExperimentRunner not available")
    
    def test_config_validation(self):
        """Test QuantizationConfig validation."""
        from quantlab.config import QuantizationConfig
        
        config = QuantizationConfig(method="int8", bits=8)
        assert config.method == "int8"
        assert config.bits == 8
    
    def test_experiment_config_defaults(self):
        """Test ExperimentConfig has sensible defaults."""
        from quantlab.config import ExperimentConfig
        
        config = ExperimentConfig(model_name="test-model")
        
        assert config.model_name == "test-model"
        assert config.quantization is not None
        assert config.benchmark is not None
    
    def test_benchmark_config_defaults(self):
        """Test BenchmarkConfig defaults."""
        from quantlab.config import BenchmarkConfig
        
        config = BenchmarkConfig()
        
        assert config.warmup_runs >= 1
        assert config.benchmark_runs >= 1
        assert len(config.input_lengths) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_unknown_strategy_raises(self):
        """Test that unknown strategy raises ValueError."""
        from quantlab.quantization import get_strategy
        
        with pytest.raises(ValueError, match="Unknown quantization method"):
            get_strategy("nonexistent_strategy")
    
    def test_experiment_with_empty_metrics(self):
        """Test experiment handles empty metrics."""
        from quantlab.storage.experiment import Experiment
        
        exp = Experiment(
            model_name="test",
            quant_method="fp16",
            metrics={},
        )
        
        data = exp.to_dict()
        assert data["metrics"] == {}
    
    def test_quantization_result_size_reduction(self):
        """Test QuantizationResult size reduction calculation."""
        from quantlab.quantization import QuantizationResult
        
        result = QuantizationResult(
            success=True,
            method="nf4",
            original_size_mb=1000.0,
            quantized_size_mb=250.0,
            compression_ratio=4.0,
        )
        
        assert result.size_reduction_pct == 75.0
    
    def test_quantization_result_zero_original(self):
        """Test QuantizationResult handles zero original size."""
        from quantlab.quantization import QuantizationResult
        
        result = QuantizationResult(
            success=True,
            method="test",
            original_size_mb=0.0,
            quantized_size_mb=0.0,
            compression_ratio=0.0,
        )
        
        assert result.size_reduction_pct == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

