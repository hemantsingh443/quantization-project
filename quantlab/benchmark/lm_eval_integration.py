"""
lm-evaluation-harness integration.

Provides a wrapper around EleutherAI/lm-evaluation-harness for 
standardized LLM evaluation (Hellaswag, MMLU, etc.)
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# Check if lm-eval is available
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval import evaluator
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False


def check_lm_eval():
    """Check if lm-evaluation-harness is installed."""
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-evaluation-harness is required for standard benchmarks. "
            "Install with: pip install lm-eval"
        )


def get_available_tasks() -> List[str]:
    """Get list of available evaluation tasks."""
    check_lm_eval()
    
    # Common tasks for LLM evaluation
    common_tasks = [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
        "openbookqa",
        "boolq",
        "mmlu",
        "gsm8k",
        "truthfulqa_mc",
    ]
    
    return common_tasks


def run_lm_eval(
    model,
    tokenizer,
    tasks: List[str],
    num_fewshot: int = 0,
    batch_size: int = 1,
    limit: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness on a model.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        tasks: List of task names to run
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        limit: Limit number of samples per task (for testing)
        device: Device to run on
        
    Returns:
        Dictionary with evaluation results
    """
    check_lm_eval()
    
    logger.info(f"Running lm-eval on tasks: {tasks}")
    
    # Create HFLM wrapper
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )
    
    logger.info("lm-eval completed")
    
    # Extract key metrics
    extracted = extract_metrics(results)
    
    return {
        "raw_results": results,
        "metrics": extracted,
    }


def extract_metrics(results: Dict) -> Dict[str, float]:
    """Extract key metrics from lm-eval results."""
    metrics = {}
    
    if "results" not in results:
        return metrics
    
    for task_name, task_results in results["results"].items():
        # Get accuracy or other primary metric
        if "acc" in task_results:
            metrics[f"{task_name}_acc"] = task_results["acc"]
        if "acc_norm" in task_results:
            metrics[f"{task_name}_acc_norm"] = task_results["acc_norm"]
        if "perplexity" in task_results:
            metrics[f"{task_name}_ppl"] = task_results["perplexity"]
    
    return metrics


def quick_eval(
    model,
    tokenizer,
    batch_size: int = 1,
    limit: int = 100,
) -> Dict[str, float]:
    """
    Run a quick evaluation with common lightweight tasks.
    
    Good for rapid iteration - runs a subset of samples.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        batch_size: Batch size
        limit: Number of samples per task
        
    Returns:
        Dictionary of metrics
    """
    check_lm_eval()
    
    # Use lightweight tasks for quick iteration
    quick_tasks = ["hellaswag", "arc_easy", "boolq"]
    
    results = run_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=quick_tasks,
        batch_size=batch_size,
        limit=limit,
    )
    
    return results["metrics"]


def full_eval(
    model,
    tokenizer,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Run full evaluation with standard benchmark suite.
    
    Takes significant time - use for final evaluation.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        batch_size: Batch size
        
    Returns:
        Dictionary of metrics
    """
    check_lm_eval()
    
    full_tasks = [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
        "boolq",
    ]
    
    results = run_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=full_tasks,
        batch_size=batch_size,
    )
    
    return results["metrics"]


class LMEvalBenchmark:
    """
    Benchmark class for lm-evaluation-harness integration.
    
    Provides a consistent interface with other QuantLab benchmarks.
    """
    
    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ):
        """
        Initialize the benchmark.
        
        Args:
            tasks: Tasks to run (default: quick evaluation tasks)
            num_fewshot: Number of few-shot examples
            limit: Limit samples per task
        """
        self.tasks = tasks or ["hellaswag", "arc_easy"]
        self.num_fewshot = num_fewshot
        self.limit = limit
    
    def run(
        self,
        model,
        tokenizer,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch_size: Batch size
            
        Returns:
            Benchmark results
        """
        if not LM_EVAL_AVAILABLE:
            logger.warning(
                "lm-eval not installed. Returning empty results. "
                "Install with: pip install lm-eval"
            )
            return {"skipped": True, "reason": "lm-eval not installed"}
        
        return run_lm_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            batch_size=batch_size,
            limit=self.limit,
        )
