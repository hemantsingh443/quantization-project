"""
Throughput benchmarking for LLM inference.
"""

import time
import logging
from typing import Dict, Any, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.getLogger(__name__)


class ThroughputBenchmark:
    """
    Measures inference throughput (tokens per second).
    
    Tests batch processing efficiency and scaling.
    """
    
    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        input_lengths: List[int] = [32],
        batch_sizes: List[int] = [1],
        output_length: int = 50,
        num_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Run throughput benchmark.
        
        Args:
            model: Model to benchmark
            tokenizer: Tokenizer
            input_lengths: Input sequence lengths to test
            batch_sizes: Batch sizes to test
            output_length: Tokens to generate per sequence
            num_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with throughput metrics
        """
        results = {}
        
        model.eval()
        device = next(model.parameters()).device
        
        for input_len in input_lengths:
            for batch_size in batch_sizes:
                logger.info(f"Throughput test: input_len={input_len}, batch={batch_size}")
                
                # Create batched inputs
                prompts = ["Hello world " * (input_len // 6)] * batch_size
                
                try:
                    inputs = tokenizer(
                        prompts,
                        return_tensors="pt",
                        max_length=input_len,
                        truncation=True,
                        padding="max_length",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception as e:
                    logger.error(f"Tokenization failed: {e}")
                    continue
                
                # Synchronize
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Run benchmark
                total_tokens = 0
                total_time_ms = 0
                
                for _ in range(num_iterations):
                    try:
                        start_time = time.perf_counter()
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=output_length,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        
                        # Count generated tokens
                        tokens_generated = (
                            outputs.shape[0] * 
                            (outputs.shape[1] - inputs["input_ids"].shape[1])
                        )
                        
                        total_tokens += tokens_generated
                        total_time_ms += elapsed_ms
                        
                    except Exception as e:
                        logger.error(f"Generation failed: {e}")
                        break
                
                # Calculate throughput
                if total_time_ms > 0:
                    throughput_tps = (total_tokens / total_time_ms) * 1000
                    results[f"throughput_b{batch_size}_l{input_len}_tps"] = throughput_tps
                    
                    # Sequences per second
                    total_sequences = batch_size * num_iterations
                    sequences_per_sec = (total_sequences / total_time_ms) * 1000
                    results[f"sequences_b{batch_size}_l{input_len}_ps"] = sequences_per_sec
                    
                    logger.info(f"Throughput: {throughput_tps:.1f} tokens/sec")
        
        # Overall throughput (using first config as reference)
        if batch_sizes and input_lengths:
            ref_key = f"throughput_b{batch_sizes[0]}_l{input_lengths[0]}_tps"
            if ref_key in results:
                results["throughput_tps"] = results[ref_key]
        
        return results


def estimate_theoretical_throughput(
    model: PreTrainedModel,
    gpu_bandwidth_gbps: float = 448.0,  # RTX 3050 bandwidth
) -> Dict[str, float]:
    """
    Estimate theoretical maximum throughput based on memory bandwidth.
    
    This provides an upper bound for memory-bound inference.
    
    Args:
        model: Model to analyze
        gpu_bandwidth_gbps: GPU memory bandwidth in GB/s
        
    Returns:
        Theoretical throughput estimates
    """
    # Calculate model size
    total_bytes = sum(
        p.numel() * p.element_size() 
        for p in model.parameters()
    )
    model_size_gb = total_bytes / (1024**3)
    
    # For autoregressive inference, we need to load the full model for each token
    # Theoretical max tokens/sec = bandwidth / model_size
    theoretical_tps = gpu_bandwidth_gbps / model_size_gb
    
    return {
        "model_size_gb": model_size_gb,
        "theoretical_max_tps": theoretical_tps,
        "gpu_bandwidth_gbps": gpu_bandwidth_gbps,
    }
