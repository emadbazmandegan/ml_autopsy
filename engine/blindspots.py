"""
Blind Spots Module for Model Autopsy

Detects regions where the model is confidently wrong or distributionally weak.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def identify_confident_wrong(
    df: pd.DataFrame,
    y_score_col: str = "y_score",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    confidence_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Identify samples where model was confident but wrong.
    
    Args:
        df: Canonical DataFrame
        y_score_col: Prediction probability column
        y_true_col: True label column
        y_pred_col: Predicted label column
        confidence_threshold: Threshold for "confident" (default 0.8)
        
    Returns:
        DataFrame with only confident-wrong samples
    """
    result = df.copy()
    
    # Calculate confidence (max of p and 1-p)
    if y_score_col in result.columns:
        result["confidence"] = result[y_score_col].apply(
            lambda p: max(p, 1 - p) if pd.notna(p) else 0.5
        )
    else:
        # No scores, can't determine confidence
        result["confidence"] = 0.5
    
    # Identify wrong predictions
    result["is_wrong"] = result[y_true_col] != result[y_pred_col]
    
    # Filter to confident and wrong
    confident_wrong = result[
        (result["confidence"] >= confidence_threshold) & 
        (result["is_wrong"] == True)
    ].copy()
    
    return confident_wrong


def compute_psi(
    baseline: pd.Series,
    comparison: pd.Series,
    n_bins: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Population Stability Index (PSI) for numeric features.
    
    PSI measures distribution shift between two populations.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change
    - PSI >= 0.25: Significant change
    
    Args:
        baseline: Baseline distribution
        comparison: Comparison distribution
        n_bins: Number of bins
        epsilon: Small value to avoid log(0)
        
    Returns:
        PSI value (>= 0)
    """
    # Handle edge cases
    if len(baseline) == 0 or len(comparison) == 0:
        return 0.0
    
    # Create bins from baseline
    combined = pd.concat([baseline, comparison]).dropna()
    if len(combined) == 0:
        return 0.0
    
    try:
        _, bin_edges = pd.qcut(baseline.dropna(), q=n_bins, retbins=True, duplicates='drop')
    except ValueError:
        # Fall back to equal-width bins
        min_val = combined.min()
        max_val = combined.max()
        if min_val == max_val:
            return 0.0
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Compute proportions
    baseline_hist, _ = np.histogram(baseline.dropna(), bins=bin_edges)
    comparison_hist, _ = np.histogram(comparison.dropna(), bins=bin_edges)
    
    baseline_pct = (baseline_hist + epsilon) / (baseline_hist.sum() + epsilon * len(baseline_hist))
    comparison_pct = (comparison_hist + epsilon) / (comparison_hist.sum() + epsilon * len(comparison_hist))
    
    # PSI formula
    psi = np.sum((comparison_pct - baseline_pct) * np.log(comparison_pct / baseline_pct))
    
    return float(max(0, psi))  # Ensure non-negative


def compute_jsd(
    baseline: pd.Series,
    comparison: pd.Series,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Jensen-Shannon Divergence for categorical features.
    
    JSD is symmetric and bounded [0, 1] (when using log base 2).
    
    Args:
        baseline: Baseline distribution
        comparison: Comparison distribution
        epsilon: Small value for stability
        
    Returns:
        JSD value in [0, 1]
    """
    # Get all unique categories
    all_categories = set(baseline.dropna().unique()) | set(comparison.dropna().unique())
    
    if len(all_categories) == 0:
        return 0.0
    
    # Compute frequency distributions
    baseline_counts = baseline.value_counts()
    comparison_counts = comparison.value_counts()
    
    p = np.array([baseline_counts.get(c, 0) for c in all_categories], dtype=float)
    q = np.array([comparison_counts.get(c, 0) for c in all_categories], dtype=float)
    
    # Normalize
    p = (p + epsilon) / (p.sum() + epsilon * len(p))
    q = (q + epsilon) / (q.sum() + epsilon * len(q))
    
    # JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)
    m = 0.5 * (p + q)
    jsd = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
    
    return float(min(1.0, max(0.0, jsd)))  # Clamp to [0, 1]


def compute_feature_divergence(
    df: pd.DataFrame,
    confident_wrong: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute divergence between full data and confident-wrong subset.
    
    Args:
        df: Full DataFrame
        confident_wrong: Confident-wrong subset
        feature_cols: Feature columns to analyze
        
    Returns:
        DataFrame with divergence scores per feature
    """
    results = []
    
    for col in feature_cols:
        if col not in df.columns or col not in confident_wrong.columns:
            continue
        
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if is_numeric:
            divergence = compute_psi(df[col], confident_wrong[col])
            divergence_type = "PSI"
        else:
            divergence = compute_jsd(df[col], confident_wrong[col])
            divergence_type = "JSD"
        
        results.append({
            "feature": col,
            "divergence": divergence,
            "divergence_type": divergence_type,
            "is_numeric": is_numeric,
        })
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) > 0:
        result_df = result_df.sort_values("divergence", ascending=False)
        result_df["rank"] = range(1, len(result_df) + 1)
    
    return result_df


def run_blindspot_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    y_score_col: str = "y_score",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    confidence_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full blind spot analysis.
    
    Args:
        df: Canonical DataFrame
        feature_cols: Feature columns (auto-detect if None)
        exclude_cols: Columns to exclude
        y_score_col: Prediction probability column
        y_true_col: True label column
        y_pred_col: Predicted label column
        confidence_threshold: Confidence threshold
        
    Returns:
        Tuple of (confident_wrong_df, divergence_df)
    """
    # Default exclusions
    default_exclude = {
        y_true_col, y_pred_col, y_score_col, "is_error", "error_type",
        "id", "index", "_id", "cluster", "confidence", "is_wrong"
    }
    exclude_set = default_exclude | set(exclude_cols or [])
    
    # Auto-detect feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude_set]
    
    # Identify confident-wrong samples
    confident_wrong = identify_confident_wrong(
        df, y_score_col, y_true_col, y_pred_col, confidence_threshold
    )
    
    if len(confident_wrong) == 0:
        empty_div = pd.DataFrame(columns=["feature", "divergence", "divergence_type", "rank"])
        return confident_wrong, empty_div
    
    # Compute feature divergence
    divergence = compute_feature_divergence(df, confident_wrong, feature_cols)
    
    return confident_wrong, divergence


def plot_confident_wrong_distribution(
    df: pd.DataFrame,
    confident_wrong: pd.DataFrame,
    feature: str,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> Path:
    """
    Plot distribution comparison for a feature.
    
    Args:
        df: Full DataFrame
        confident_wrong: Confident-wrong subset
        feature: Feature to plot
        output_path: Path to save plot
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    is_numeric = pd.api.types.is_numeric_dtype(df[feature])
    
    if is_numeric:
        ax.hist(df[feature].dropna(), bins=20, alpha=0.5, label='All Data', density=True)
        ax.hist(confident_wrong[feature].dropna(), bins=20, alpha=0.7, 
               label='Confident Wrong', density=True, color='red')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
    else:
        # Categorical: bar chart
        all_counts = df[feature].value_counts(normalize=True).head(10)
        cw_counts = confident_wrong[feature].value_counts(normalize=True).head(10)
        
        x = np.arange(len(all_counts))
        width = 0.35
        
        ax.bar(x - width/2, all_counts.values, width, label='All Data', alpha=0.7)
        cw_vals = [cw_counts.get(cat, 0) for cat in all_counts.index]
        ax.bar(x + width/2, cw_vals, width, label='Confident Wrong', alpha=0.7, color='red')
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Proportion')
    
    ax.set_title(f'Distribution Shift: {feature}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def save_blindspot_results(
    confident_wrong: pd.DataFrame,
    divergence: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: Path,
    top_features: int = 5,
) -> Dict[str, Path]:
    """
    Save blind spot analysis artifacts.
    
    Args:
        confident_wrong: Confident-wrong samples
        divergence: Feature divergence DataFrame
        df: Full DataFrame
        output_dir: Output directory
        top_features: Number of top features to plot
        
    Returns:
        Dictionary of artifact paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save divergence CSV
    drift_path = output_dir / "drift_features.csv"
    divergence.to_csv(drift_path, index=False)
    paths["drift"] = drift_path
    
    # Save confident-wrong subset
    cw_path = output_dir / "confident_wrong.csv"
    confident_wrong.to_csv(cw_path, index=False)
    paths["confident_wrong"] = cw_path
    
    # Plot top divergent features
    if len(divergence) > 0 and len(confident_wrong) > 0:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for _, row in divergence.head(top_features).iterrows():
            feature = row["feature"]
            plot_path = plots_dir / f"drift_{feature}.png"
            plot_confident_wrong_distribution(df, confident_wrong, feature, plot_path)
            paths[f"plot_{feature}"] = plot_path
    
    return paths
