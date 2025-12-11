"""
Quantization Pipeline - Orchestrates the full quantization workflow.

Stages:
1. Preprocessing / Tokenization
2. Quantization Transform
3. Calibration (if required)
4. Error Analysis
5. Benchmark Execution
6. Logging + Visualization
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from quantlab.config import ExperimentConfig, QuantizationConfig
from quantlab.models.registry import ModelRegistry
from quantlab.quantization import get_quantizer
from quantlab.benchmark import BenchmarkRunner
from quantlab.storage import Experiment


logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """A single stage in the pipeline."""
    name: str
    function: Callable
    enabled: bool = True
    config: Dict[str, Any] = None


class QuantizationPipeline:
    """
    Modular pipeline for quantization experiments.
    
    Allows customization of each stage and supports hooks for
    extensibility.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Experiment configuration
        """
        self.config = config or ExperimentConfig()
        self.registry = ModelRegistry()
        self.benchmark_runner = BenchmarkRunner()
        
        # Pipeline stages
        self.stages: List[PipelineStage] = []
        
        # Hooks
        self._pre_hooks: Dict[str, List[Callable]] = {}
        self._post_hooks: Dict[str, List[Callable]] = {}
        
        # Results
        self.results: Dict[str, Any] = {}
        
        # Set up default stages
        self._setup_default_stages()
    
    def _setup_default_stages(self):
        """Set up the default pipeline stages."""
        self.stages = [
            PipelineStage("load_model", self._stage_load_model),
            PipelineStage("quantize", self._stage_quantize),
            PipelineStage("calibrate", self._stage_calibrate, enabled=False),
            PipelineStage("analyze_errors", self._stage_analyze_errors),
            PipelineStage("benchmark", self._stage_benchmark),
        ]
    
    def add_hook(self, stage_name: str, hook: Callable, position: str = "post"):
        """
        Add a hook to a pipeline stage.
        
        Args:
            stage_name: Name of the stage to hook into
            hook: Function to call. Receives (pipeline, stage_name, context)
            position: "pre" or "post"
        """
        hooks = self._pre_hooks if position == "pre" else self._post_hooks
        if stage_name not in hooks:
            hooks[stage_name] = []
        hooks[stage_name].append(hook)
    
    def enable_stage(self, stage_name: str):
        """Enable a pipeline stage."""
        for stage in self.stages:
            if stage.name == stage_name:
                stage.enabled = True
                return
        raise ValueError(f"Unknown stage: {stage_name}")
    
    def disable_stage(self, stage_name: str):
        """Disable a pipeline stage."""
        for stage in self.stages:
            if stage.name == stage_name:
                stage.enabled = False
                return
        raise ValueError(f"Unknown stage: {stage_name}")
    
    def run(
        self,
        model_name: Optional[str] = None,
        calibration_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline.
        
        Args:
            model_name: Model to load (overrides config)
            calibration_data: Data for calibration stage
            
        Returns:
            Dictionary with all results
        """
        if model_name:
            self.config.model_name = model_name
        
        # Initialize context
        context = {
            "config": self.config,
            "model": None,
            "tokenizer": None,
            "calibration_data": calibration_data,
            "metrics": {},
        }
        
        logger.info(f"Starting pipeline for {self.config.model_name}")
        
        # Run each stage
        for stage in self.stages:
            if not stage.enabled:
                logger.info(f"Skipping disabled stage: {stage.name}")
                continue
            
            logger.info(f"Running stage: {stage.name}")
            
            # Pre-hooks
            for hook in self._pre_hooks.get(stage.name, []):
                hook(self, stage.name, context)
            
            # Run stage
            try:
                stage.function(context)
            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                context["error"] = str(e)
                break
            
            # Post-hooks
            for hook in self._post_hooks.get(stage.name, []):
                hook(self, stage.name, context)
        
        self.results = context
        return context
    
    def _stage_load_model(self, context: Dict[str, Any]):
        """Load model stage."""
        model, tokenizer = self.registry.load_model(
            model_name=context["config"].model_name,
            quantization_config=context["config"].quantization,
            hardware_config=context["config"].hardware,
        )
        
        context["model"] = model
        context["tokenizer"] = tokenizer
        
        # Store model info
        context["model_info"] = {
            "name": context["config"].model_name,
            "params": sum(p.numel() for p in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
        }
    
    def _stage_quantize(self, context: Dict[str, Any]):
        """Apply quantization stage."""
        quant_config = context["config"].quantization
        
        if quant_config.method in ["none", "fp32"]:
            logger.info("No quantization applied")
            return
        
        # For BNB-based quantization, it's done at load time
        # This stage can apply additional post-load quantization
        quantizer = get_quantizer(quant_config.method)
        
        if quantizer is None:
            logger.info(f"Quantization {quant_config.method} applied at load time")
            return
        
        # Apply quantization
        context["model"] = quantizer.quantize(
            context["model"],
            quant_config.to_dict() if hasattr(quant_config, 'to_dict') else {},
        )
    
    def _stage_calibrate(self, context: Dict[str, Any]):
        """Calibration stage for methods that require it."""
        calibration_data = context.get("calibration_data")
        
        if calibration_data is None:
            logger.warning("No calibration data provided, skipping")
            return
        
        quant_config = context["config"].quantization
        quantizer = get_quantizer(quant_config.method)
        
        if quantizer and hasattr(quantizer, "calibrate"):
            context["model"] = quantizer.calibrate(
                context["model"],
                calibration_data,
                {"num_samples": context["config"].benchmark.eval_samples},
            )
    
    def _stage_analyze_errors(self, context: Dict[str, Any]):
        """Analyze quantization errors."""
        # Basic error analysis - check for NaN/Inf
        model = context["model"]
        
        error_report = {
            "nan_count": 0,
            "inf_count": 0,
            "layers_checked": 0,
        }
        
        for name, param in model.named_parameters():
            error_report["layers_checked"] += 1
            if torch.isnan(param).any():
                error_report["nan_count"] += 1
            if torch.isinf(param).any():
                error_report["inf_count"] += 1
        
        context["error_report"] = error_report
        
        if error_report["nan_count"] > 0 or error_report["inf_count"] > 0:
            logger.warning(f"Found numerical issues: {error_report}")
    
    def _stage_benchmark(self, context: Dict[str, Any]):
        """Run benchmarks stage."""
        metrics = self.benchmark_runner.run_all(
            model=context["model"],
            tokenizer=context["tokenizer"],
            config=context["config"].benchmark,
        )
        
        context["metrics"].update(metrics)
    
    def get_experiment(self) -> Experiment:
        """Convert pipeline results to an Experiment object."""
        if not self.results:
            raise ValueError("Pipeline hasn't been run yet")
        
        return Experiment(
            model_name=self.config.model_name,
            model_size=self.results.get("model_info", {}).get("params"),
            quant_method=self.config.quantization.method,
            quant_config=self.config.quantization.to_dict() if hasattr(self.config.quantization, 'to_dict') else {},
            metrics=self.results.get("metrics", {}),
        )


# Convenience function
def run_pipeline(
    model_name: str,
    quantization: str = "fp16",
    benchmark: bool = True,
    calibration_data: Any = None,
) -> Dict[str, Any]:
    """
    Quick pipeline execution.
    
    Args:
        model_name: Model to load
        quantization: Quantization method
        benchmark: Whether to run benchmarks
        calibration_data: Optional calibration data
        
    Returns:
        Pipeline results
    """
    config = ExperimentConfig(
        model_name=model_name,
        quantization=QuantizationConfig(method=quantization),
    )
    
    pipeline = QuantizationPipeline(config)
    
    if not benchmark:
        pipeline.disable_stage("benchmark")
    
    return pipeline.run(calibration_data=calibration_data)
