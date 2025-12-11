"""
Benchmark runner - orchestrates all benchmark suites.
"""

import logging
from typing import Dict, Any, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from quantlab.config import BenchmarkConfig
from quantlab.benchmark.latency import LatencyBenchmark
from quantlab.benchmark.throughput import ThroughputBenchmark
from quantlab.benchmark.memory import MemoryBenchmark
from quantlab.benchmark.deviation import DeviationBenchmark


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates running of all benchmark suites.
    
    Collects latency, throughput, memory, and accuracy metrics.
    """
    
    def __init__(self):
        self.latency = LatencyBenchmark()
        self.throughput = ThroughputBenchmark()
        self.memory = MemoryBenchmark()
        self.deviation = DeviationBenchmark()
    
    def run_all(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: BenchmarkConfig,
    ) -> Dict[str, Any]:
        """
        Run all benchmarks.
        
        Args:
            model: Model to benchmark
            tokenizer: Model tokenizer
            config: Benchmark configuration
            
        Returns:
            Dictionary with all benchmark results
        """
        results = {}
        
        # Memory benchmark (run first, before any forward passes)
        logger.info("Running memory benchmark...")
        memory_results = self.memory.run(model)
        results.update(memory_results)
        
        # Latency benchmark
        logger.info("Running latency benchmark...")
        latency_results = self.latency.run(
            model=model,
            tokenizer=tokenizer,
            input_lengths=config.input_lengths,
            output_length=config.output_length,
            warmup_runs=config.warmup_runs,
            benchmark_runs=config.benchmark_runs,
        )
        results.update(latency_results)
        
        # Throughput benchmark
        logger.info("Running throughput benchmark...")
        throughput_results = self.throughput.run(
            model=model,
            tokenizer=tokenizer,
            input_lengths=config.input_lengths,
            batch_sizes=config.batch_sizes,
            output_length=config.output_length,
        )
        results.update(throughput_results)
        
        return results
    
    def run_evals(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        suites: List[str],
        samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Run evaluation suites.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            suites: List of eval suite names ('simple', 'perplexity', 'mmlu')
            samples: Number of samples per suite
            
        Returns:
            Evaluation results
        """
        results = {}
        
        for suite in suites:
            logger.info(f"Running eval suite: {suite}")
            
            if suite == "simple":
                results.update(self._run_simple_eval(model, tokenizer, samples))
            elif suite == "perplexity":
                results.update(self._run_perplexity_eval(model, tokenizer, samples))
            else:
                logger.warning(f"Unknown eval suite: {suite}")
        
        return results
    
    def _run_simple_eval(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        samples: int,
    ) -> Dict[str, Any]:
        """Run simple coherence evaluation."""
        prompts = [
            "The capital of France is",
            "Water boils at",
            "The sun rises in the",
            "One plus one equals",
            "The largest planet in our solar system is",
        ]
        
        results = {
            "simple_eval_samples": min(samples, len(prompts)),
            "simple_eval_responses": [],
        }
        
        model.eval()
        with torch.no_grad():
            for prompt in prompts[:samples]:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    results["simple_eval_responses"].append({
                        "prompt": prompt,
                        "response": response,
                    })
                except Exception as e:
                    logger.error(f"Generation failed for '{prompt}': {e}")
        
        return results
    
    def _run_perplexity_eval(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        samples: int,
    ) -> Dict[str, Any]:
        """Calculate perplexity on sample texts."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
        ]
        
        total_loss = 0.0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for text in texts[:samples]:
                inputs = tokenizer(text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                try:
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    num_tokens = inputs["input_ids"].numel()
                    
                    total_loss += loss * num_tokens
                    total_tokens += num_tokens
                except Exception as e:
                    logger.error(f"Perplexity calc failed: {e}")
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        else:
            perplexity = float("inf")
        
        return {
            "perplexity": perplexity,
            "perplexity_samples": min(samples, len(texts)),
        }
    
    def compare_models(
        self,
        baseline_model: PreTrainedModel,
        quantized_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: BenchmarkConfig,
    ) -> Dict[str, Any]:
        """
        Compare baseline and quantized models.
        
        Args:
            baseline_model: Original model
            quantized_model: Quantized version
            tokenizer: Shared tokenizer
            config: Benchmark configuration
            
        Returns:
            Comparison results with deviations
        """
        logger.info("Comparing baseline vs quantized model...")
        
        # Run benchmarks on both
        baseline_results = self.run_all(baseline_model, tokenizer, config)
        quantized_results = self.run_all(quantized_model, tokenizer, config)
        
        # Compute deviation
        deviation_results = self.deviation.run(
            baseline_model=baseline_model,
            quantized_model=quantized_model,
            tokenizer=tokenizer,
        )
        
        return {
            "baseline": baseline_results,
            "quantized": quantized_results,
            "deviation": deviation_results,
        }
