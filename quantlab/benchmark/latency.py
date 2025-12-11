"""
Latency benchmarking for LLM inference.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import statistics

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.getLogger(__name__)


class LatencyBenchmark:
    """
    Measures inference latency with various configurations.
    
    Captures:
    - Time to first token (TTFT)
    - Per-token latency
    - Total generation time
    - Statistics (mean, std, p50, p95, p99)
    """
    
    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        input_lengths: List[int] = [32, 64, 128],
        output_length: int = 50,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Run latency benchmark.
        
        Args:
            model: Model to benchmark
            tokenizer: Tokenizer
            input_lengths: List of input sequence lengths to test
            output_length: Number of tokens to generate
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            
        Returns:
            Dictionary with latency metrics
        """
        results = {}
        
        model.eval()
        device = next(model.parameters()).device
        
        for input_len in input_lengths:
            logger.info(f"Benchmarking input_length={input_len}")
            
            # Create dummy input
            prompt = "Hello " * (input_len // 2)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=input_len,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Warmup
            logger.debug(f"Running {warmup_runs} warmup iterations...")
            with torch.no_grad():
                for _ in range(warmup_runs):
                    try:
                        _ = model.generate(
                            **inputs,
                            max_new_tokens=min(output_length, 10),
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    except Exception as e:
                        logger.warning(f"Warmup failed: {e}")
                        break
            
            # Synchronize before benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark runs
            latencies = []
            ttft_times = []  # Time to first token
            
            for run_idx in range(benchmark_runs):
                try:
                    # Use CUDA events for precise timing
                    if torch.cuda.is_available():
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        first_token_event = torch.cuda.Event(enable_timing=True)
                        
                        start_event.record()
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=output_length,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        
                        end_event.record()
                        torch.cuda.synchronize()
                        
                        elapsed_ms = start_event.elapsed_time(end_event)
                    else:
                        # CPU timing
                        start_time = time.perf_counter()
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=output_length,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                    
                    latencies.append(elapsed_ms)
                    
                    # Estimate TTFT (first token)
                    num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
                    if num_generated > 0:
                        ttft = elapsed_ms / num_generated
                        ttft_times.append(ttft)
                    
                except Exception as e:
                    logger.error(f"Benchmark run {run_idx} failed: {e}")
            
            # Calculate statistics
            if latencies:
                sorted_latencies = sorted(latencies)
                
                results[f"latency_{input_len}_mean_ms"] = statistics.mean(latencies)
                results[f"latency_{input_len}_std_ms"] = (
                    statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                )
                results[f"latency_{input_len}_min_ms"] = min(latencies)
                results[f"latency_{input_len}_max_ms"] = max(latencies)
                results[f"latency_{input_len}_p50_ms"] = sorted_latencies[len(latencies) // 2]
                results[f"latency_{input_len}_p95_ms"] = sorted_latencies[int(len(latencies) * 0.95)]
                results[f"latency_{input_len}_p99_ms"] = sorted_latencies[int(len(latencies) * 0.99)]
                
                # Per-token latency
                results[f"per_token_{input_len}_ms"] = results[f"latency_{input_len}_mean_ms"] / output_length
                
                if ttft_times:
                    results[f"ttft_{input_len}_ms"] = statistics.mean(ttft_times)
        
        # Overall summary using first input length as reference
        if input_lengths and f"latency_{input_lengths[0]}_mean_ms" in results:
            ref_len = input_lengths[0]
            results["latency_mean_ms"] = results[f"latency_{ref_len}_mean_ms"]
            results["latency_std_ms"] = results[f"latency_{ref_len}_std_ms"]
            results["per_token_ms"] = results[f"per_token_{ref_len}_ms"]
        
        return results


def quick_latency_test(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str = "Hello, how are you?",
    output_tokens: int = 20,
) -> float:
    """
    Quick single-run latency test.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer
        prompt: Test prompt
        output_tokens: Number of tokens to generate
        
    Returns:
        Latency in milliseconds
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=output_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end)
    else:
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=output_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return (time.perf_counter() - start) * 1000
