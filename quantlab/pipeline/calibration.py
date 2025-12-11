"""
Calibration utilities for quantization methods that require representative data.
"""

import logging
from typing import Optional, List, Any, Iterator
import random

import torch
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


class CalibrationDataset(Dataset):
    """
    Dataset for calibration samples.
    
    Can be initialized with raw text or pre-tokenized inputs.
    """
    
    def __init__(
        self,
        samples: List[Any],
        tokenizer: Any = None,
        max_length: int = 128,
    ):
        """
        Args:
            samples: List of text strings or pre-tokenized inputs
            tokenizer: Optional tokenizer for text samples
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize if tokenizer provided
        if tokenizer and isinstance(samples[0], str):
            self._tokenize_samples()
    
    def _tokenize_samples(self):
        """Tokenize text samples."""
        tokenized = []
        for text in self.samples:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized.append({k: v.squeeze(0) for k, v in tokens.items()})
        self.samples = tokenized
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Any:
        return self.samples[idx]


def collect_calibration_data(
    tokenizer: Any,
    num_samples: int = 128,
    max_length: int = 128,
    source: str = "wikitext",
) -> CalibrationDataset:
    """
    Collect calibration data from a standard source.
    
    Args:
        tokenizer: Tokenizer to use
        num_samples: Number of samples to collect
        max_length: Maximum sequence length
        source: Data source ("wikitext", "c4", "custom")
        
    Returns:
        CalibrationDataset ready for calibration
    """
    samples = []
    
    if source == "wikitext":
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            
            # Filter for sufficient length
            for item in dataset:
                text = item["text"].strip()
                if len(text) > 50:  # Minimum length
                    samples.append(text)
                    if len(samples) >= num_samples:
                        break
            
            logger.info(f"Collected {len(samples)} samples from wikitext")
            
        except Exception as e:
            logger.warning(f"Failed to load wikitext: {e}, using default samples")
            samples = _get_default_samples(num_samples)
    
    elif source == "c4":
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("c4", "en", split="train", streaming=True)
            
            for item in dataset:
                text = item["text"].strip()
                if len(text) > 50:
                    samples.append(text[:512])  # Limit length
                    if len(samples) >= num_samples:
                        break
            
            logger.info(f"Collected {len(samples)} samples from C4")
            
        except Exception as e:
            logger.warning(f"Failed to load C4: {e}, using default samples")
            samples = _get_default_samples(num_samples)
    
    else:
        samples = _get_default_samples(num_samples)
    
    return CalibrationDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def _get_default_samples(num_samples: int = 128) -> List[str]:
    """Get default calibration samples."""
    base_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Python is a high-level programming language known for its simple syntax and readability.",
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Climate change is one of the most pressing issues facing our planet today.",
        "The Internet has revolutionized the way we communicate and access information.",
        "Quantum computing promises to solve problems that are intractable for classical computers.",
        "The human brain contains approximately 86 billion neurons.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "Deep learning has achieved remarkable success in computer vision and natural language processing.",
        "The scientific method involves observation, hypothesis, experimentation, and conclusion.",
        "Artificial general intelligence remains an open research challenge.",
        "Data privacy is increasingly important in our digital age.",
        "The discovery of DNA structure was one of the most important scientific breakthroughs.",
        "Space exploration has led to numerous technological innovations.",
    ]
    
    # Repeat and shuffle to get enough samples
    samples = []
    while len(samples) < num_samples:
        samples.extend(base_samples)
    
    random.shuffle(samples)
    return samples[:num_samples]


def create_calibration_dataloader(
    dataset: CalibrationDataset,
    batch_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for calibration.
    
    Args:
        dataset: Calibration dataset
        batch_size: Batch size (usually 1 for calibration)
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def collect_activation_ranges(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
    num_batches: int = 100,
) -> dict:
    """
    Collect activation ranges for static quantization.
    
    Args:
        model: Model to analyze
        calibration_loader: Calibration data loader
        num_batches: Number of batches to process
        
    Returns:
        Dictionary of layer_name -> (min, max) ranges
    """
    ranges = {}
    
    # Hook to collect activations
    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                if name not in ranges:
                    ranges[name] = {
                        "min": float("inf"),
                        "max": float("-inf"),
                    }
                ranges[name]["min"] = min(ranges[name]["min"], output.min().item())
                ranges[name]["max"] = max(ranges[name]["max"], output.max().item())
        return fn
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            handles.append(module.register_forward_hook(hook_fn(name)))
    
    # Run calibration
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                # Move to device
                device = next(model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
            else:
                batch = batch.to(next(model.parameters()).device)
                model(batch)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    logger.info(f"Collected activation ranges for {len(ranges)} layers")
    return ranges
