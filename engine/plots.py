"""
Plots Module for Model Autopsy

Generates diagnostic visualizations for model evaluation.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use


def plot_confusion_matrix(
    confusion_data: Dict[str, Any],
    output_path: Path,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple = (8, 6),
) -> Path:
    """
    Generate and save a confusion matrix plot.
    
    Args:
        confusion_data: Dict with 'matrix' and 'labels' keys
        output_path: Path to save the plot
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    matrix = np.array(confusion_data["matrix"])
    labels = confusion_data.get("labels", list(range(len(matrix))))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if matrix[i, j] > thresh else "black",
                   fontsize=12, fontweight='bold')
    
    # For binary classification, add FP/FN/TP/TN labels
    if matrix.shape == (2, 2):
        annotations = [
            (0, 0, "TN"),
            (0, 1, "FP"),
            (1, 0, "FN"),
            (1, 1, "TP"),
        ]
        for i, j, label in annotations:
            ax.text(j, i + 0.35, f"({label})",
                   ha="center", va="center",
                   color="white" if matrix[i, j] > thresh else "gray",
                   fontsize=9)
    
    fig.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_calibration_curve(
    calibration_data: Dict[str, Any],
    output_path: Path,
    title: str = "Calibration Curve",
    figsize: tuple = (10, 5),
) -> Path:
    """
    Generate and save a calibration curve plot.
    
    Args:
        calibration_data: Dict with 'bins' containing bin statistics
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    bins = calibration_data.get("bins", [])
    
    if not bins:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No calibration data available",
               ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return Path(output_path)
    
    # Extract data
    mean_predicted = [b["mean_predicted"] for b in bins]
    mean_actual = [b["mean_actual"] for b in bins]
    counts = [b["count"] for b in bins]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    ax1.plot(mean_predicted, mean_actual, 'o-', color='#2196F3', 
            label='Model', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Right: Histogram of predictions
    bin_edges = [b["bin_lower"] for b in bins] + [bins[-1]["bin_upper"]]
    ax2.bar(mean_predicted, counts, width=0.08, alpha=0.7, color='#4CAF50', edgecolor='black')
    ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14)
    ax2.set_xlim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def generate_all_plots(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate all diagnostic plots from computed metrics.
    
    Args:
        metrics: Dictionary from compute_all_metrics()
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to paths
    """
    output_dir = Path(output_dir)
    plots = {}
    
    # Confusion matrix
    if "confusion_matrix" in metrics:
        confusion_path = output_dir / "confusion.png"
        plot_confusion_matrix(metrics["confusion_matrix"], confusion_path)
        plots["confusion"] = confusion_path
    
    # Calibration curve
    if "calibration" in metrics:
        calibration_path = output_dir / "calibration.png"
        plot_calibration_curve(metrics["calibration"], calibration_path)
        plots["calibration"] = calibration_path
    
    return plots
