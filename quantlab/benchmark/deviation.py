"""
Logit deviation analysis for comparing quantized models.
"""

import logging
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.getLogger(__name__)


class DeviationBenchmark:
    """
    Measures logit-level deviation between baseline and quantized models.
    
    Useful for understanding where quantization introduces errors.
    """
    
    def run(
        self,
        baseline_model: PreTrainedModel,
        quantized_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        test_prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare logit outputs between two models.
        
        Args:
            baseline_model: Original (higher precision) model
            quantized_model: Quantized model to compare
            tokenizer: Shared tokenizer
            test_prompts: Prompts to test on
            
        Returns:
            Deviation metrics
        """
        if test_prompts is None:
            test_prompts = [
                "The capital of France is",
                "Machine learning is",
                "def fibonacci(n):",
                "In the year 2024,",
            ]
        
        results = {
            "num_prompts": len(test_prompts),
            "kl_divergences": [],
            "mse_values": [],
            "cosine_similarities": [],
            "top1_agreement": [],
            "top5_agreement": [],
        }
        
        baseline_model.eval()
        quantized_model.eval()
        
        baseline_device = next(baseline_model.parameters()).device
        quantized_device = next(quantized_model.parameters()).device
        
        with torch.no_grad():
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Get baseline logits
                baseline_inputs = {k: v.to(baseline_device) for k, v in inputs.items()}
                baseline_outputs = baseline_model(**baseline_inputs)
                baseline_logits = baseline_outputs.logits[:, -1, :]  # Last token logits
                
                # Get quantized logits
                quant_inputs = {k: v.to(quantized_device) for k, v in inputs.items()}
                quant_outputs = quantized_model(**quant_inputs)
                quant_logits = quant_outputs.logits[:, -1, :]
                
                # Move to same device for comparison
                baseline_logits = baseline_logits.float().cpu()
                quant_logits = quant_logits.float().cpu()
                
                # KL Divergence
                baseline_probs = F.softmax(baseline_logits, dim=-1)
                quant_probs = F.softmax(quant_logits, dim=-1)
                kl_div = F.kl_div(
                    quant_probs.log(),
                    baseline_probs,
                    reduction="batchmean"
                ).item()
                results["kl_divergences"].append(kl_div)
                
                # MSE
                mse = F.mse_loss(quant_logits, baseline_logits).item()
                results["mse_values"].append(mse)
                
                # Cosine similarity
                cosine_sim = F.cosine_similarity(
                    baseline_logits,
                    quant_logits,
                    dim=-1
                ).mean().item()
                results["cosine_similarities"].append(cosine_sim)
                
                # Top-K agreement
                baseline_top5 = baseline_logits.topk(5, dim=-1).indices
                quant_top5 = quant_logits.topk(5, dim=-1).indices
                
                # Top-1 agreement
                top1_match = (baseline_top5[:, 0] == quant_top5[:, 0]).float().mean().item()
                results["top1_agreement"].append(top1_match)
                
                # Top-5 agreement (any overlap in top 5)
                baseline_set = set(baseline_top5[0].tolist())
                quant_set = set(quant_top5[0].tolist())
                top5_overlap = len(baseline_set & quant_set) / 5
                results["top5_agreement"].append(top5_overlap)
        
        # Aggregate statistics
        import statistics
        
        if results["kl_divergences"]:
            results["kl_divergence_mean"] = statistics.mean(results["kl_divergences"])
            results["mse_mean"] = statistics.mean(results["mse_values"])
            results["cosine_similarity_mean"] = statistics.mean(results["cosine_similarities"])
            results["top1_agreement_mean"] = statistics.mean(results["top1_agreement"])
            results["top5_agreement_mean"] = statistics.mean(results["top5_agreement"])
        
        return results


def per_layer_deviation(
    baseline_model: PreTrainedModel,
    quantized_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str = "Hello, world!",
) -> Dict[str, Dict[str, float]]:
    """
    Analyze deviation at each layer.
    
    Uses hooks to capture intermediate activations.
    
    Args:
        baseline_model: Baseline model
        quantized_model: Quantized model
        tokenizer: Tokenizer
        prompt: Test prompt
        
    Returns:
        Per-layer deviation metrics
    """
    baseline_activations = {}
    quantized_activations = {}
    
    # Register hooks to capture activations
    def make_hook(storage, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[name] = output[0].detach().cpu().float()
            else:
                storage[name] = output.detach().cpu().float()
        return hook
    
    baseline_hooks = []
    quantized_hooks = []
    
    # Hook into transformer layers
    for name, module in baseline_model.named_modules():
        if "layers" in name and name.count(".") <= 2:
            handle = module.register_forward_hook(
                make_hook(baseline_activations, name)
            )
            baseline_hooks.append(handle)
    
    for name, module in quantized_model.named_modules():
        if "layers" in name and name.count(".") <= 2:
            handle = module.register_forward_hook(
                make_hook(quantized_activations, name)
            )
            quantized_hooks.append(handle)
    
    # Run forward pass
    inputs = tokenizer(prompt, return_tensors="pt")
    
    baseline_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        baseline_input = {k: v.to(next(baseline_model.parameters()).device) for k, v in inputs.items()}
        _ = baseline_model(**baseline_input)
        
        quant_input = {k: v.to(next(quantized_model.parameters()).device) for k, v in inputs.items()}
        _ = quantized_model(**quant_input)
    
    # Remove hooks
    for handle in baseline_hooks + quantized_hooks:
        handle.remove()
    
    # Compute per-layer deviation
    layer_deviations = {}
    
    for name in baseline_activations:
        if name in quantized_activations:
            baseline_act = baseline_activations[name]
            quant_act = quantized_activations[name]
            
            # Ensure same shape
            if baseline_act.shape != quant_act.shape:
                continue
            
            mse = F.mse_loss(quant_act, baseline_act).item()
            cosine = F.cosine_similarity(
                baseline_act.flatten(),
                quant_act.flatten(),
                dim=0
            ).item()
            
            layer_deviations[name] = {
                "mse": mse,
                "cosine_similarity": cosine,
                "max_diff": (baseline_act - quant_act).abs().max().item(),
            }
    
    return layer_deviations
