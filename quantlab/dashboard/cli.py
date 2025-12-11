"""
Command Line Interface for QuantLab.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from quantlab.core import QuantLab
from quantlab.config import QuantLabConfig, ExperimentConfig


# Initialize CLI app
app = typer.Typer(
    name="quantlab",
    help="LLM Quantization Experimentation Platform",
    add_completion=False,
)

console = Console()


def get_lab() -> QuantLab:
    """Get QuantLab instance."""
    return QuantLab()


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or path"),
    quant: str = typer.Option("fp16", "--quant", "-q", help="Quantization method"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Experiment name"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip evaluation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run a quantization experiment.
    
    Examples:
        quantlab run facebook/opt-125m --quant int8
        quantlab run gpt2 --quant nf4 --name "GPT2 NF4 Test"
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    console.print(f"[bold blue]Running experiment on {model}[/]")
    
    lab = get_lab()
    
    tag_list = tags.split(",") if tags else []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        
        try:
            result = lab.run_experiment(
                model_name=model,
                quantization=quant,
                name=name,
                tags=tag_list,
                run_benchmark=True,
                run_eval=not no_eval,
            )
            
            progress.update(task, description="Complete!")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            raise typer.Exit(1)
    
    # Display results
    console.print(Panel(
        result.summary(),
        title=f"Experiment Results: {result.id}",
        border_style="green",
    ))


@app.command()
def list_experiments(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model"),
    quant: Optional[str] = typer.Option(None, "--quant", "-q", help="Filter by method"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max results"),
):
    """
    List saved experiments.
    
    Examples:
        quantlab list
        quantlab list --model gpt2 --limit 5
    """
    lab = get_lab()
    experiments = lab.list_experiments(
        model_name=model,
        quant_method=quant,
        limit=limit,
    )
    
    if not experiments:
        console.print("[yellow]No experiments found.[/]")
        return
    
    # Create table
    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Quant", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Timestamp")
    
    for exp in experiments:
        table.add_row(
            exp.id,
            exp.model_name.split("/")[-1][:20],
            exp.quant_method,
            exp.status,
            f"{exp.metrics.get('memory_mb', 'N/A'):.1f}" if exp.metrics.get('memory_mb') else "N/A",
            f"{exp.metrics.get('latency_mean_ms', 'N/A'):.1f}" if exp.metrics.get('latency_mean_ms') else "N/A",
            exp.timestamp.strftime("%Y-%m-%d %H:%M") if exp.timestamp else "N/A",
        )
    
    console.print(table)


@app.command()
def inspect(
    experiment_id: str = typer.Argument(..., help="Experiment ID to inspect"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """
    Inspect a specific experiment.
    
    Examples:
        quantlab inspect abc123
        quantlab inspect abc123 --json
    """
    lab = get_lab()
    exp = lab.get_experiment(experiment_id)
    
    if exp is None:
        console.print(f"[red]Experiment {experiment_id} not found.[/]")
        raise typer.Exit(1)
    
    if json_output:
        console.print_json(json.dumps(exp.to_dict(), indent=2, default=str))
    else:
        console.print(Panel(
            exp.summary(),
            title=f"Experiment: {exp.id}",
            border_style="blue",
        ))
        
        # Show all metrics
        if exp.metrics:
            console.print("\n[bold]All Metrics:[/]")
            for key, value in sorted(exp.metrics.items()):
                if isinstance(value, float):
                    console.print(f"  {key}: {value:.4f}")
                elif not isinstance(value, (dict, list)):
                    console.print(f"  {key}: {value}")
        
        # Show hardware
        if exp.hardware:
            console.print("\n[bold]Hardware:[/]")
            for key, value in exp.hardware.items():
                console.print(f"  {key}: {value}")


@app.command()
def compare(
    ids: List[str] = typer.Argument(..., help="Experiment IDs to compare"),
):
    """
    Compare multiple experiments.
    
    Examples:
        quantlab compare abc123 def456 ghi789
    """
    lab = get_lab()
    comparison = lab.compare(ids)
    
    if not comparison.get("experiments"):
        console.print("[red]No valid experiments found.[/]")
        raise typer.Exit(1)
    
    # Create comparison table
    table = Table(title="Experiment Comparison")
    
    # Add columns
    table.add_column("Metric", style="cyan")
    for exp_info in comparison["experiments"]:
        table.add_column(f"{exp_info['id']}\n{exp_info['quant']}", justify="right")
    
    # Add rows for each metric
    for metric, values in comparison["metrics"].items():
        row = [metric]
        for val in values:
            if val is None:
                row.append("N/A")
            elif isinstance(val, float):
                row.append(f"{val:.2f}")
            else:
                row.append(str(val))
        table.add_row(*row)
    
    console.print(table)


@app.command()
def sweep(
    model: str = typer.Argument(..., help="Model to sweep"),
    methods: str = typer.Option("fp16,int8,nf4", help="Comma-separated methods"),
):
    """
    Run experiments across multiple quantization methods.
    
    Examples:
        quantlab sweep facebook/opt-125m
        quantlab sweep gpt2 --methods "fp16,int8,int4,nf4"
    """
    method_list = [m.strip() for m in methods.split(",")]
    
    console.print(f"[bold blue]Sweeping {model} with methods: {method_list}[/]")
    
    lab = get_lab()
    
    results = []
    for method in method_list:
        console.print(f"\n[yellow]Testing {method}...[/]")
        try:
            result = lab.run_experiment(
                model_name=model,
                quantization=method,
                run_benchmark=True,
                run_eval=False,
            )
            results.append(result)
            console.print(f"[green]✓ {method}: {result.metrics.get('memory_mb', 'N/A'):.1f} MB[/]")
        except Exception as e:
            console.print(f"[red]✗ {method}: {e}[/]")
    
    # Compare all results
    if len(results) > 1:
        console.print("\n")
        comparison = lab.compare([r.id for r in results])
        
        table = Table(title="Sweep Results")
        table.add_column("Method", style="cyan")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Status", style="green")
        
        for i, exp_info in enumerate(comparison["experiments"]):
            metrics = comparison["metrics"]
            table.add_row(
                exp_info["quant"],
                f"{metrics.get('memory_mb', [None])[i] or 'N/A'}",
                f"{metrics.get('latency_mean_ms', [None])[i] or 'N/A'}",
                "✓",
            )
        
        console.print(table)


@app.command()
def delete(
    experiment_id: str = typer.Argument(..., help="Experiment ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Delete an experiment.
    """
    if not force:
        confirm = typer.confirm(f"Delete experiment {experiment_id}?")
        if not confirm:
            raise typer.Abort()
    
    lab = get_lab()
    if lab.delete_experiment(experiment_id):
        console.print(f"[green]Deleted experiment {experiment_id}[/]")
    else:
        console.print(f"[red]Experiment {experiment_id} not found[/]")
        raise typer.Exit(1)


@app.command()
def stats():
    """
    Show storage statistics.
    """
    lab = get_lab()
    statistics = lab.store.get_statistics()
    
    console.print(Panel(
        f"Total experiments: {statistics['total_experiments']}",
        title="Storage Statistics",
    ))
    
    if statistics.get("by_quant_method"):
        console.print("\n[bold]By Quantization Method:[/]")
        for method, count in statistics["by_quant_method"].items():
            console.print(f"  {method}: {count}")
    
    if statistics.get("by_model"):
        console.print("\n[bold]By Model:[/]")
        for model, count in list(statistics["by_model"].items())[:10]:
            console.print(f"  {model.split('/')[-1][:30]}: {count}")


@app.command()
def dashboard():
    """
    Launch the web dashboard.
    """
    console.print("[bold blue]Launching web dashboard...[/]")
    console.print("Run: streamlit run quantlab/dashboard/web.py")
    console.print("\nOr use the Python API:")
    console.print("  from quantlab.dashboard.web import launch_dashboard")
    console.print("  launch_dashboard()")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
