"""
Report Module for Model Autopsy

Generates comprehensive markdown reports and results indexes.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import pandas as pd


def generate_executive_summary(
    metrics: Dict[str, Any],
    error_rate: float,
    total_samples: int,
) -> str:
    """
    Generate executive summary section.
    
    Args:
        metrics: Dictionary of performance metrics
        error_rate: Overall error rate
        total_samples: Total number of samples
        
    Returns:
        Markdown string
    """
    lines = [
        "## Executive Summary\n",
        f"This report analyzes model performance across **{total_samples:,}** samples.\n",
    ]
    
    # Key metrics
    if metrics:
        lines.append("### Key Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.3f} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")
    
    # Error summary
    lines.append(f"**Overall Error Rate:** {error_rate:.1%}\n")
    
    return "\n".join(lines)


def generate_slices_section(
    slices_df: pd.DataFrame,
    top_n: int = 10,
) -> str:
    """
    Generate top failure slices section.
    
    Args:
        slices_df: DataFrame with slice data
        top_n: Number of top slices to show
        
    Returns:
        Markdown string
    """
    lines = [
        "## Top Failure Slices\n",
        "Slices ranked by lift (error rate relative to baseline).\n",
    ]
    
    if len(slices_df) == 0:
        lines.append("*No significant failure slices detected.*\n")
        return "\n".join(lines)
    
    # Table header
    lines.append("| Rank | Slice | Support | Error Rate | Lift |")
    lines.append("|------|-------|---------|------------|------|")
    
    for i, row in slices_df.head(top_n).iterrows():
        rank = row.get("rank", i + 1)
        rule = row.get("rule", row.get("slice", "N/A"))
        support = row.get("support", 0)
        error_rate = row.get("error_rate", 0)
        lift = row.get("lift", 0)
        
        lines.append(f"| {rank} | {rule} | {support:,} | {error_rate:.1%} | {lift:.2f}x |")
    
    lines.append("")
    return "\n".join(lines)


def generate_clusters_section(
    profiles: List[Dict[str, Any]],
) -> str:
    """
    Generate cluster profiles section.
    
    Args:
        profiles: List of cluster profile dictionaries
        
    Returns:
        Markdown string
    """
    lines = [
        "## Error Clusters\n",
        "Distinct failure modes identified through clustering.\n",
    ]
    
    if not profiles:
        lines.append("*No clusters generated.*\n")
        return "\n".join(lines)
    
    for profile in profiles:
        cluster_id = profile.get("cluster_id", 0)
        size = profile.get("size", 0)
        size_pct = profile.get("size_pct", 0)
        dominant_error = profile.get("dominant_error", "Unknown")
        
        lines.append(f"### Cluster {cluster_id}")
        lines.append(f"- **Size:** {size:,} ({size_pct:.1%} of errors)")
        lines.append(f"- **Dominant Error:** {dominant_error}")
        
        top_features = profile.get("top_features", [])[:3]
        if top_features:
            lines.append("- **Key Features:**")
            for feat in top_features:
                name = feat.get("feature", "")
                z = feat.get("z_diff", 0)
                direction = "↑" if z > 0 else "↓"
                lines.append(f"  - {name} {direction} ({z:+.2f}σ)")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_attribution_section(
    global_importance: pd.DataFrame,
    delta: pd.DataFrame,
    top_n: int = 10,
) -> str:
    """
    Generate attribution highlights section.
    
    Args:
        global_importance: Global feature importance
        delta: Delta attribution (error - global)
        top_n: Number of top features to show
        
    Returns:
        Markdown string
    """
    lines = [
        "## Feature Attribution\n",
        "Features driving model errors.\n",
    ]
    
    # Global importance
    if len(global_importance) > 0:
        lines.append("### Global Importance\n")
        lines.append("| Feature | Importance |")
        lines.append("|---------|------------|")
        
        for _, row in global_importance.head(top_n).iterrows():
            lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} |")
        lines.append("")
    
    # Delta attribution
    if len(delta) > 0:
        lines.append("### Error-Specific Features\n")
        lines.append("Features with highest difference between error and global importance.\n")
        lines.append("| Feature | Delta | Direction |")
        lines.append("|---------|-------|-----------|")
        
        for _, row in delta.head(top_n).iterrows():
            direction = "↑ More important for errors" if row['delta'] > 0 else "↓ Less important for errors"
            lines.append(f"| {row['feature']} | {row['delta']:+.4f} | {direction} |")
        lines.append("")
    
    return "\n".join(lines)


def generate_recommendations(
    slices_df: pd.DataFrame,
    divergence_df: pd.DataFrame,
    profiles: List[Dict[str, Any]],
) -> str:
    """
    Generate actionable recommendations.
    
    Args:
        slices_df: Slice data
        divergence_df: Feature divergence data
        profiles: Cluster profiles
        
    Returns:
        Markdown string
    """
    lines = [
        "## Recommendations\n",
    ]
    
    recommendations = []
    
    # Based on slices
    if len(slices_df) > 0:
        top_slice = slices_df.iloc[0]
        rule = top_slice.get("rule", "")
        lift = top_slice.get("lift", 0)
        if lift > 2:
            recommendations.append(
                f"**High-Risk Slice:** Investigate samples matching `{rule}` "
                f"which have {lift:.1f}x higher error rate."
            )
    
    # Based on divergence
    if len(divergence_df) > 0:
        top_drift = divergence_df.iloc[0]
        feature = top_drift.get("feature", "")
        div = top_drift.get("divergence", 0)
        if div > 0.1:
            recommendations.append(
                f"**Distribution Shift:** Feature `{feature}` shows significant "
                f"divergence in confident-wrong samples (divergence={div:.3f})."
            )
    
    # Based on clusters
    if profiles:
        dominant_errors = [p.get("dominant_error") for p in profiles]
        if dominant_errors.count("FN") > dominant_errors.count("FP"):
            recommendations.append(
                "**Error Pattern:** Model tends toward false negatives. "
                "Consider adjusting decision threshold or rebalancing training data."
            )
        elif dominant_errors.count("FP") > dominant_errors.count("FN"):
            recommendations.append(
                "**Error Pattern:** Model tends toward false positives. "
                "Consider increasing precision focus."
            )
    
    if not recommendations:
        recommendations.append("No critical issues detected. Continue monitoring.")
    
    for rec in recommendations:
        lines.append(f"- {rec}")
    
    lines.append("")
    return "\n".join(lines)


def generate_report(
    run_id: str,
    metrics: Dict[str, Any],
    slices_df: Optional[pd.DataFrame] = None,
    profiles: Optional[List[Dict[str, Any]]] = None,
    global_importance: Optional[pd.DataFrame] = None,
    delta: Optional[pd.DataFrame] = None,
    divergence_df: Optional[pd.DataFrame] = None,
    total_samples: int = 0,
    error_count: int = 0,
) -> str:
    """
    Generate complete markdown report.
    
    Args:
        run_id: Run identifier
        metrics: Performance metrics
        slices_df: Slice data
        profiles: Cluster profiles
        global_importance: Global feature importance
        delta: Delta attribution
        divergence_df: Feature divergence
        total_samples: Total samples
        error_count: Error count
        
    Returns:
        Complete markdown report
    """
    error_rate = error_count / total_samples if total_samples > 0 else 0
    
    sections = [
        f"# Model Autopsy Report\n",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---\n",
    ]
    
    # Executive summary
    sections.append(generate_executive_summary(metrics, error_rate, total_samples))
    
    # Top slices
    if slices_df is not None:
        sections.append(generate_slices_section(slices_df))
    
    # Clusters
    if profiles is not None:
        sections.append(generate_clusters_section(profiles))
    
    # Attribution
    if global_importance is not None:
        sections.append(generate_attribution_section(
            global_importance,
            delta if delta is not None else pd.DataFrame(),
        ))
    
    # Recommendations
    sections.append(generate_recommendations(
        slices_df if slices_df is not None else pd.DataFrame(),
        divergence_df if divergence_df is not None else pd.DataFrame(),
        profiles if profiles is not None else [],
    ))
    
    # Footer
    sections.append("---\n")
    sections.append("*Generated by Model Autopsy*")
    
    return "\n".join(sections)


def generate_results_index(
    run_id: str,
    output_dir: Path,
    artifacts: Dict[str, Path],
) -> Dict[str, Any]:
    """
    Generate JSON index of all artifacts.
    
    Args:
        run_id: Run identifier
        output_dir: Output directory
        artifacts: Dictionary of artifact paths
        
    Returns:
        Results index dictionary
    """
    index = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "artifacts": {},
    }
    
    for name, path in artifacts.items():
        if path and Path(path).exists():
            index["artifacts"][name] = {
                "path": str(path),
                "filename": Path(path).name,
                "size_bytes": Path(path).stat().st_size,
            }
    
    return index


def save_report(
    report: str,
    results_index: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save report and results index.
    
    Args:
        report: Markdown report content
        results_index: Results index dictionary
        output_dir: Output directory
        
    Returns:
        Dictionary of saved paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    paths["report"] = report_path
    
    # Save results index
    index_path = output_dir / "results_index.json"
    with open(index_path, "w") as f:
        json.dump(results_index, f, indent=2, default=str)
    paths["index"] = index_path
    
    return paths
