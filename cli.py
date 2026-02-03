"""
Model Autopsy CLI

Command-line interface for running model diagnostics.
"""
import sys
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="ml-autopsy")
def cli():
    """
    Model Autopsy - Diagnostic suite for ML model debugging.
    
    Stop shipping black boxes. Start performing autopsies.
    """
    pass


@cli.command()
@click.option("--port", default=8501, help="Port to run the server on")
def serve(port: int):
    """Launch the interactive Streamlit dashboard."""
    console.print(Panel.fit(
        "[bold green]Starting Model Autopsy Dashboard[/bold green]\n"
        f"[dim]Running on http://localhost:{port}[/dim]",
        title="☠️ Model Autopsy",
        border_style="green"
    ))
    
    ui_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(ui_path), 
        "--server.port", str(port)
    ])


@cli.command()
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True),
              help="Path to dataset CSV (features)")
@click.option("--predictions", "-p", required=True, type=click.Path(exists=True),
              help="Path to predictions CSV")
@click.option("--labels", "-l", type=click.Path(exists=True), default=None,
              help="Optional: Path to labels CSV if separate from dataset/predictions")
@click.option("--output-dir", "-o", default="autopsy_report",
              help="Output directory for report")
@click.option("--threshold", "-t", default=0.5, type=float,
              help="Classification threshold for deriving predictions from scores")
@click.option("--min-support", default=10, type=int,
              help="Minimum support for slice detection")
@click.option("--n-clusters", default=3, type=int,
              help="Number of clusters for error analysis")
@click.option("--format", "output_format", default="markdown", 
              type=click.Choice(["markdown", "json", "both"]),
              help="Output format for the report")
def audit(
    dataset: str,
    predictions: str,
    labels: Optional[str],
    output_dir: str,
    threshold: float,
    min_support: int,
    n_clusters: int,
    output_format: str,
):
    """
    Run a headless model autopsy and generate a report.
    
    Example:
        ml-autopsy audit -d data.csv -p preds.csv -o report/
    """
    run_id = f"cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        f"[bold]Run ID:[/bold] {run_id}\n"
        f"[bold]Dataset:[/bold] {dataset}\n"
        f"[bold]Predictions:[/bold] {predictions}\n"
        f"[bold]Output:[/bold] {output_path.absolute()}",
        title="☠️ Model Autopsy - Audit",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Ingest data
        task = progress.add_task("[cyan]Ingesting data...", total=None)
        
        try:
            from engine.ingest import ingest_uploaded_files, ValidationError
            from io import BytesIO
            
            # Read files into BytesIO to simulate upload
            with open(dataset, 'rb') as f:
                dataset_bytes = BytesIO(f.read())
            with open(predictions, 'rb') as f:
                predictions_bytes = BytesIO(f.read())
            labels_bytes = None
            if labels:
                with open(labels, 'rb') as f:
                    labels_bytes = BytesIO(f.read())
            
            canonical, config, feature_cols = ingest_uploaded_files(
                dataset_file=dataset_bytes,
                predictions_file=predictions_bytes,
                labels_file=labels_bytes,
                threshold=threshold,
            )
            progress.update(task, description=f"[green]✓ Ingested {config['n_rows']} samples")
            
        except ValidationError as e:
            console.print(f"[red]Validation Error:[/red] {e}")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)
        
        # Step 2: Compute metrics
        task = progress.add_task("[cyan]Computing metrics...", total=None)
        
        from engine.metrics import compute_classification_metrics, compute_auc_metrics
        
        metrics = compute_classification_metrics(
            canonical["y_true"], canonical["y_pred"]
        )
        
        if "y_score" in canonical.columns:
            auc_metrics = compute_auc_metrics(
                canonical["y_true"], canonical["y_score"]
            )
            metrics.update(auc_metrics)
        
        progress.update(task, description="[green]✓ Metrics computed")
        
        # Step 3: Slice analysis
        task = progress.add_task("[cyan]Discovering failure slices...", total=None)
        
        from engine.slicing import generate_all_slices, rank_slices
        
        slices = generate_all_slices(canonical, feature_cols)
        dynamic_min_support = max(min_support, len(canonical) // 20)
        ranked_slices = rank_slices(slices, min_support=dynamic_min_support, top_n=10)
        
        progress.update(task, description=f"[green]✓ Found {len(ranked_slices)} significant slices")
        
        # Step 4: Clustering
        task = progress.add_task("[cyan]Clustering errors...", total=None)
        
        from engine.clustering import run_error_clustering
        
        actual_clusters = min(n_clusters, max(1, canonical["is_error"].sum()))
        errors_df, profiles, X_2d = run_error_clustering(
            canonical, feature_cols=feature_cols, n_clusters=actual_clusters
        )
        
        progress.update(task, description=f"[green]✓ Identified {len(profiles)} error clusters")
        
        # Step 5: Attribution
        task = progress.add_task("[cyan]Running attribution analysis...", total=None)
        
        from engine.attribution import run_attribution_analysis
        
        global_imp, error_imp, delta = run_attribution_analysis(
            canonical, feature_cols=feature_cols, n_repeats=5
        )
        
        progress.update(task, description="[green]✓ Attribution complete")
        
        # Step 6: Blind spots
        task = progress.add_task("[cyan]Detecting blind spots...", total=None)
        
        from engine.blindspots import run_blindspot_analysis
        
        confident_wrong, divergence = run_blindspot_analysis(canonical)
        
        progress.update(task, description="[green]✓ Blind spot analysis complete")
        
        # Step 7: Generate report
        task = progress.add_task("[cyan]Generating report...", total=None)
        
        from engine.report import generate_report, save_report, generate_results_index
        
        slices_df = pd.DataFrame(ranked_slices) if ranked_slices else pd.DataFrame()
        
        report = generate_report(
            run_id=run_id,
            metrics=metrics,
            slices_df=slices_df,
            profiles=profiles,
            global_importance=global_imp,
            delta=delta,
            divergence_df=divergence,
            total_samples=len(canonical),
            error_count=int(canonical["is_error"].sum()),
        )
        
        results_index = generate_results_index(run_id, output_path, {})
        saved_paths = save_report(report, results_index, output_path)
        
        progress.update(task, description="[green]✓ Report generated")
    
    # Summary
    console.print()
    
    # Metrics table
    table = Table(title="Key Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.3f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)
    console.print()
    
    # Output files
    console.print(Panel.fit(
        f"[green]Report saved to:[/green] {saved_paths['report']}\n"
        f"[green]Results index:[/green] {saved_paths['index']}",
        title="✅ Autopsy Complete",
        border_style="green"
    ))


@cli.command()
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True),
              help="Path to dataset CSV")
@click.option("--predictions", "-p", required=True, type=click.Path(exists=True),
              help="Path to predictions CSV")
def validate(dataset: str, predictions: str):
    """
    Validate input files without running full analysis.
    
    Useful for checking if your files will work before a full audit.
    """
    console.print("[cyan]Validating files...[/cyan]")
    
    try:
        from engine.ingest import ingest_uploaded_files, ValidationError
        from io import BytesIO
        
        with open(dataset, 'rb') as f:
            dataset_bytes = BytesIO(f.read())
        with open(predictions, 'rb') as f:
            predictions_bytes = BytesIO(f.read())
        
        canonical, config, feature_cols = ingest_uploaded_files(
            dataset_file=dataset_bytes,
            predictions_file=predictions_bytes,
        )
        
        console.print(f"[green]✓ Validation passed![/green]")
        console.print(f"  Samples: {config['n_rows']}")
        console.print(f"  Features: {config['n_features']}")
        console.print(f"  Task type: {config['task_type']}")
        console.print(f"  Detected columns: y_true={config.get('y_true_col')}, y_pred={config.get('y_pred_col')}")
        
    except ValidationError as e:
        console.print(f"[red]✗ Validation failed:[/red] {e}")
        raise SystemExit(1)


@cli.command()
def demo():
    """
    Run the autopsy on demo data to see an example output.
    """
    demo_dir = Path(__file__).parent / "examples" / "demo"
    dataset_path = demo_dir / "adult_dataset.csv"
    predictions_path = demo_dir / "adult_predictions.csv"
    
    if not dataset_path.exists():
        console.print("[red]Demo files not found. Make sure you're running from the project root.[/red]")
        raise SystemExit(1)
    
    console.print("[cyan]Running demo autopsy on UCI Adult dataset...[/cyan]")
    
    # Call audit command with demo data
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(audit, [
        "--dataset", str(dataset_path),
        "--predictions", str(predictions_path),
        "--output-dir", "demo_report",
    ])
    
    if result.exit_code != 0:
        console.print(f"[red]Demo failed:[/red] {result.output}")
        raise SystemExit(1)
    
    console.print(result.output)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
