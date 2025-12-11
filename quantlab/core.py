"""
Core QuantLab orchestration class.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from quantlab.config import QuantLabConfig, ExperimentConfig, get_default_config
from quantlab.models.registry import ModelRegistry
from quantlab.quantization import get_quantizer
from quantlab.benchmark import BenchmarkRunner
from quantlab.storage import ExperimentStore, Experiment


logger = logging.getLogger(__name__)


class QuantLab:
    """
    Main orchestration class for the quantization experimentation platform.
    
    Usage:
        lab = QuantLab()
        result = lab.run_experiment(
            model_name="facebook/opt-125m",
            quantization="int8"
        )
    """
    
    def __init__(self, config: Optional[QuantLabConfig] = None):
        """Initialize QuantLab with optional configuration."""
        self.config = config or QuantLabConfig()
        self._setup_logging()
        
        # Initialize components
        self.registry = ModelRegistry(cache_dir=self.config.cache_dir)
        self.store = ExperimentStore(
            experiments_dir=self.config.experiments_dir,
            db_path=self.config.db_path
        )
        self.benchmark_runner = BenchmarkRunner()
        
        logger.info(f"QuantLab initialized. Experiments dir: {self.config.experiments_dir}")
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def run_experiment(
        self,
        model_name: str,
        quantization: str = "none",
        config: Optional[ExperimentConfig] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        run_benchmark: bool = True,
        run_eval: bool = False,
        baseline_id: Optional[str] = None,
    ) -> Experiment:
        """
        Run a complete quantization experiment.
        
        Args:
            model_name: HuggingFace model name or local path
            quantization: Quantization method (none, fp16, int8, int4, nf4, gptq, awq)
            config: Full experiment configuration (overrides other params)
            name: Experiment name for identification
            tags: Tags for filtering experiments
            run_benchmark: Whether to run performance benchmarks
            run_eval: Whether to run evaluation suites
            baseline_id: Optional baseline experiment ID for comparison
            
        Returns:
            Experiment object with results
        """
        # Build configuration
        if config is None:
            try:
                config = get_default_config(quantization)
            except ValueError:
                config = ExperimentConfig()
                config.quantization.method = quantization
        
        config.model_name = model_name
        if name:
            config.name = name
        if tags:
            config.tags = tags
        
        logger.info(f"Starting experiment: {config.model_name} with {config.quantization.method}")
        
        # Create experiment
        experiment = Experiment(
            model_name=config.model_name,
            model_size=None,  # Will be populated after loading
            quant_method=config.quantization.method,
            quant_config=config.quantization.to_dict() if hasattr(config.quantization, 'to_dict') else {},
            hardware=self._get_hardware_info(),
            name=config.name,
            tags=config.tags,
        )
        
        try:
            # Step 1: Load model
            logger.info("Loading model...")
            model, tokenizer = self.registry.load_model(
                model_name=config.model_name,
                revision=config.model_revision,
                quantization_config=config.quantization,
                hardware_config=config.hardware,
            )
            
            # Update model size
            experiment.model_size = self._get_model_size(model)
            logger.info(f"Model loaded. Size: {experiment.model_size}")
            
            # Step 2: Run benchmarks
            if run_benchmark:
                logger.info("Running benchmarks...")
                benchmark_results = self.benchmark_runner.run_all(
                    model=model,
                    tokenizer=tokenizer,
                    config=config.benchmark,
                )
                experiment.metrics.update(benchmark_results)
                logger.info(f"Benchmarks complete. Latency: {benchmark_results.get('latency_mean_ms', 'N/A')}ms")
            
            # Step 3: Run evaluations
            if run_eval and config.benchmark.eval_suites:
                logger.info("Running evaluations...")
                eval_results = self.benchmark_runner.run_evals(
                    model=model,
                    tokenizer=tokenizer,
                    suites=config.benchmark.eval_suites,
                    samples=config.benchmark.eval_samples,
                )
                experiment.metrics.update(eval_results)
            
            # Step 4: Compute deviation from baseline
            if baseline_id:
                logger.info(f"Computing deviation from baseline {baseline_id}...")
                baseline = self.store.load_experiment(baseline_id)
                if baseline:
                    deviation = self._compute_deviation(experiment, baseline)
                    experiment.metrics["baseline_comparison"] = deviation
            
            experiment.status = "completed"
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            experiment.status = "failed"
            experiment.error = str(e)
            raise
        
        finally:
            # Save experiment
            self.store.save_experiment(experiment)
            logger.info(f"Experiment saved: {experiment.id}")
        
        return experiment
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Collect hardware information."""
        import platform
        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }
        
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except ImportError:
            pass
        
        return info
    
    def _get_model_size(self, model) -> str:
        """Get model size in human-readable format."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            if total_params >= 1e9:
                return f"{total_params/1e9:.1f}B"
            elif total_params >= 1e6:
                return f"{total_params/1e6:.0f}M"
            else:
                return f"{total_params/1e3:.0f}K"
        except:
            return "unknown"
    
    def _compute_deviation(self, exp: Experiment, baseline: Experiment) -> Dict[str, Any]:
        """Compute metrics deviation from baseline."""
        deviation = {}
        for key in ["latency_mean_ms", "memory_mb", "throughput_tps"]:
            if key in exp.metrics and key in baseline.metrics:
                exp_val = exp.metrics[key]
                base_val = baseline.metrics[key]
                if base_val > 0:
                    deviation[f"{key}_change_pct"] = ((exp_val - base_val) / base_val) * 100
        return deviation
    
    # Experiment management
    def list_experiments(
        self,
        model_name: Optional[str] = None,
        quant_method: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Experiment]:
        """List experiments with optional filters."""
        return self.store.list_experiments(
            model_name=model_name,
            quant_method=quant_method,
            tags=tags,
            status=status,
            limit=limit,
        )
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get a specific experiment by ID."""
        return self.store.load_experiment(experiment_id)
    
    def compare(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (default: all)
            
        Returns:
            Comparison dictionary with metrics for each experiment
        """
        experiments = [self.store.load_experiment(eid) for eid in experiment_ids]
        experiments = [e for e in experiments if e is not None]
        
        if not experiments:
            return {}
        
        if metrics is None:
            # Get all metrics from first experiment
            metrics = list(experiments[0].metrics.keys())
        
        comparison = {
            "experiments": [],
            "metrics": {},
        }
        
        for exp in experiments:
            comparison["experiments"].append({
                "id": exp.id,
                "model": exp.model_name,
                "quant": exp.quant_method,
                "timestamp": exp.timestamp.isoformat() if exp.timestamp else None,
            })
        
        for metric in metrics:
            comparison["metrics"][metric] = [
                exp.metrics.get(metric) for exp in experiments
            ]
        
        return comparison
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        return self.store.delete_experiment(experiment_id)
    
    # Sweep functionality
    def sweep(
        self,
        model_name: str,
        quantization_methods: List[str],
        **kwargs
    ) -> List[Experiment]:
        """
        Run experiments across multiple quantization methods.
        
        Args:
            model_name: Model to test
            quantization_methods: List of quantization methods to try
            **kwargs: Additional arguments passed to run_experiment
            
        Returns:
            List of experiment results
        """
        results = []
        
        # Run baseline first
        if "none" not in quantization_methods and "fp32" not in quantization_methods:
            logger.info("Running FP16 baseline...")
            baseline = self.run_experiment(
                model_name=model_name,
                quantization="fp16",
                **kwargs
            )
            results.append(baseline)
            baseline_id = baseline.id
        else:
            baseline_id = None
        
        # Run each quantization method
        for quant_method in quantization_methods:
            logger.info(f"Running {quant_method}...")
            try:
                exp = self.run_experiment(
                    model_name=model_name,
                    quantization=quant_method,
                    baseline_id=baseline_id,
                    **kwargs
                )
                results.append(exp)
            except Exception as e:
                logger.error(f"Failed to run {quant_method}: {e}")
        
        return results
