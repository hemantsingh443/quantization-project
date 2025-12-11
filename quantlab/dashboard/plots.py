"""
Visualization utilities for experiment results.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


logger = logging.getLogger(__name__)


def plot_memory_comparison(
    experiments: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Any]:
    """
    Plot memory usage comparison across experiments.
    
    Args:
        experiments: List of experiment dicts with 'name' and 'memory_mb' keys
        output_path: Optional path to save the plot
        show: Whether to display the plot
        
    Returns:
        Figure object if available
    """
    if PLOTLY_AVAILABLE:
        names = [e.get("name", e.get("id", "Unknown")) for e in experiments]
        memories = [e.get("memory_mb", 0) for e in experiments]
        quant_methods = [e.get("quant_method", "unknown") for e in experiments]
        
        fig = px.bar(
            x=names,
            y=memories,
            color=quant_methods,
            labels={"x": "Experiment", "y": "Memory (MB)", "color": "Quantization"},
            title="Memory Usage Comparison",
        )
        
        if output_path:
            fig.write_image(str(output_path))
        
        if show:
            fig.show()
        
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        names = [e.get("name", e.get("id", "Unknown"))[:15] for e in experiments]
        memories = [e.get("memory_mb", 0) for e in experiments]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, memories)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    else:
        logger.warning("No plotting library available")
        return None


def plot_latency_comparison(
    experiments: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Any]:
    """
    Plot latency comparison across experiments.
    
    Args:
        experiments: List of experiment dicts
        output_path: Optional path to save the plot
        show: Whether to display the plot
        
    Returns:
        Figure object if available
    """
    if PLOTLY_AVAILABLE:
        names = [e.get("name", e.get("id", "Unknown")) for e in experiments]
        latencies = [e.get("latency_mean_ms", 0) for e in experiments]
        latency_stds = [e.get("latency_std_ms", 0) for e in experiments]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names,
            y=latencies,
            error_y=dict(type="data", array=latency_stds),
            name="Latency",
        ))
        
        fig.update_layout(
            title="Latency Comparison",
            xaxis_title="Experiment",
            yaxis_title="Latency (ms)",
        )
        
        if output_path:
            fig.write_image(str(output_path))
        
        if show:
            fig.show()
        
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        names = [e.get("name", e.get("id", "Unknown"))[:15] for e in experiments]
        latencies = [e.get("latency_mean_ms", 0) for e in experiments]
        stds = [e.get("latency_std_ms", 0) for e in experiments]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, latencies, yerr=stds, capsize=5)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    return None


def plot_performance_vs_accuracy(
    experiments: List[Dict[str, Any]],
    performance_metric: str = "memory_mb",
    accuracy_metric: str = "perplexity",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Any]:
    """
    Plot performance metric vs accuracy metric.
    
    Args:
        experiments: List of experiment dicts
        performance_metric: Which performance metric to use (x-axis)
        accuracy_metric: Which accuracy metric to use (y-axis)
        output_path: Optional path to save
        show: Whether to display
        
    Returns:
        Figure object
    """
    if PLOTLY_AVAILABLE:
        x_vals = [e.get(performance_metric, 0) for e in experiments]
        y_vals = [e.get(accuracy_metric, 0) for e in experiments]
        names = [e.get("name", e.get("id", "Unknown")) for e in experiments]
        quant_methods = [e.get("quant_method", "unknown") for e in experiments]
        
        fig = px.scatter(
            x=x_vals,
            y=y_vals,
            color=quant_methods,
            text=names,
            labels={
                "x": performance_metric.replace("_", " ").title(),
                "y": accuracy_metric.replace("_", " ").title(),
                "color": "Quantization",
            },
            title=f"{performance_metric} vs {accuracy_metric}",
        )
        
        fig.update_traces(textposition="top center")
        
        if output_path:
            fig.write_image(str(output_path))
        
        if show:
            fig.show()
        
        return fig
    
    return None


def plot_layer_deviation_heatmap(
    layer_deviations: Dict[str, Dict[str, float]],
    metric: str = "mse",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Any]:
    """
    Plot heatmap of per-layer deviation.
    
    Args:
        layer_deviations: Dict of layer_name -> {metric: value}
        metric: Which metric to plot (mse, cosine_similarity, max_diff)
        output_path: Optional save path
        show: Whether to display
        
    Returns:
        Figure object
    """
    if not layer_deviations:
        return None
    
    layers = list(layer_deviations.keys())
    values = [layer_deviations[l].get(metric, 0) for l in layers]
    
    if PLOTLY_AVAILABLE:
        # Reshape for heatmap (single row)
        fig = go.Figure(data=go.Heatmap(
            z=[values],
            x=layers,
            y=[metric],
            colorscale="RdYlGn_r" if metric == "mse" else "RdYlGn",
        ))
        
        fig.update_layout(
            title=f"Per-Layer {metric.upper()}",
            xaxis_title="Layer",
            height=200,
        )
        
        if output_path:
            fig.write_image(str(output_path))
        
        if show:
            fig.show()
        
        return fig
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Simple bar chart for layers
        ax.bar(range(len(layers)), values)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([l.split(".")[-1][:10] for l in layers], rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Per-Layer {metric.upper()}")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    return None


def create_comparison_report(
    experiments: List[Dict[str, Any]],
    output_dir: Path,
    include_plots: bool = True,
) -> Path:
    """
    Create a comprehensive comparison report.
    
    Args:
        experiments: List of experiment dicts
        output_dir: Directory to save report files
        include_plots: Whether to generate plots
        
    Returns:
        Path to the generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Quantization Comparison Report",
        "",
        f"**Number of experiments:** {len(experiments)}",
        "",
        "## Summary Table",
        "",
        "| ID | Model | Quantization | Memory (MB) | Latency (ms) | Throughput (t/s) |",
        "|-----|-------|--------------|-------------|--------------|------------------|",
    ]
    
    for exp in experiments:
        report_lines.append(
            f"| {exp.get('id', 'N/A')} | "
            f"{exp.get('model_name', 'N/A').split('/')[-1][:20]} | "
            f"{exp.get('quant_method', 'N/A')} | "
            f"{exp.get('memory_mb', 'N/A'):.1f if exp.get('memory_mb') else 'N/A'} | "
            f"{exp.get('latency_mean_ms', 'N/A'):.1f if exp.get('latency_mean_ms') else 'N/A'} | "
            f"{exp.get('throughput_tps', 'N/A'):.1f if exp.get('throughput_tps') else 'N/A'} |"
        )
    
    report_lines.extend(["", "## Detailed Metrics", ""])
    
    for exp in experiments:
        report_lines.append(f"### {exp.get('id', 'Unknown')} - {exp.get('quant_method', 'Unknown')}")
        report_lines.append("")
        for key, value in sorted(exp.items()):
            if key not in ["id", "model_name", "quant_method", "artifacts", "hardware"]:
                if isinstance(value, float):
                    report_lines.append(f"- **{key}**: {value:.4f}")
                elif isinstance(value, (int, str, bool)):
                    report_lines.append(f"- **{key}**: {value}")
        report_lines.append("")
    
    # Generate plots
    if include_plots:
        report_lines.append("## Visualizations")
        report_lines.append("")
        
        # Memory plot
        mem_path = output_dir / "memory_comparison.png"
        plot_memory_comparison(experiments, output_path=mem_path, show=False)
        if mem_path.exists():
            report_lines.append(f"![Memory Comparison](memory_comparison.png)")
            report_lines.append("")
        
        # Latency plot
        lat_path = output_dir / "latency_comparison.png"
        plot_latency_comparison(experiments, output_path=lat_path, show=False)
        if lat_path.exists():
            report_lines.append(f"![Latency Comparison](latency_comparison.png)")
            report_lines.append("")
    
    # Write report
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Report generated: {report_path}")
    return report_path
