"""
Comparison Module for Model Autopsy

Functionality for comparing two models (A/B testing) to identify
performance differences and disagreement patterns.
"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from engine.metrics import compute_classification_metrics
from engine.slicing import generate_all_slices


def validate_models(
    y_true: pd.Series,
    y_pred_a: pd.Series,
    y_pred_b: pd.Series,
) -> bool:
    """
    Validate that inputs are compatible for comparison.
    
    Args:
        y_true: True labels
        y_pred_a: Predictions from Model A
        y_pred_b: Predictions from Model B
        
    Returns:
        True if valid, raises ValueError if not
    """
    n = len(y_true)
    if len(y_pred_a) != n:
        raise ValueError(f"Model A length ({len(y_pred_a)}) does not match dataset ({n})")
    if len(y_pred_b) != n:
        raise ValueError(f"Model B length ({len(y_pred_b)}) does not match dataset ({n})")
        
    return True


def compute_comparison_metrics(
    y_true: pd.Series,
    y_pred_a: pd.Series,
    y_pred_b: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for both models and their deltas.
    
    Args:
        y_true: True labels
        y_pred_a: Predictions from Model A
        y_pred_b: Predictions from Model B
        
    Returns:
        Dictionary with 'model_a', 'model_b', and 'delta' metrics
    """
    validate_models(y_true, y_pred_a, y_pred_b)
    
    metrics_a = compute_classification_metrics(y_true, y_pred_a)
    metrics_b = compute_classification_metrics(y_true, y_pred_b)
    
    deltas = {}
    for metric in metrics_a:
        if isinstance(metrics_a[metric], (int, float)):
            deltas[metric] = metrics_b.get(metric, 0) - metrics_a[metric]
            
    return {
        "model_a": metrics_a,
        "model_b": metrics_b,
        "delta": deltas
    }


def find_disagreements(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_a_col: str,
    y_pred_b_col: str,
) -> Dict[str, Any]:
    """
    Find samples where models disagree and categorize them.
    
    Args:
        df: Dataframe containing all columns
        y_true_col: Column name for true labels
        y_pred_a_col: Column name for Model A predictions
        y_pred_b_col: Column name for Model B predictions
        
    Returns:
        Dictionary with disagreement stats and dataframe of disagreeing samples
    """
    validate_models(df[y_true_col], df[y_pred_a_col], df[y_pred_b_col])
    
    # Identify disagreements
    disagree_mask = df[y_pred_a_col] != df[y_pred_b_col]
    disagreements = df[disagree_mask].copy()
    
    if disagreements.empty:
        return {
            "count": 0,
            "rate": 0.0,
            "df": pd.DataFrame(),
            "a_correct_count": 0,
            "b_correct_count": 0
        }
    
    # Analyze who was right
    a_correct = disagreements[y_pred_a_col] == disagreements[y_true_col]
    b_correct = disagreements[y_pred_b_col] == disagreements[y_true_col]
    
    disagreements["winner"] = np.where(a_correct, "Model A", "Model B")
    
    return {
        "count": int(disagree_mask.sum()),
        "rate": float(disagree_mask.mean()),
        "df": disagreements,
        "a_correct_count": int(a_correct.sum()),
        "b_correct_count": int(b_correct.sum())
    }


def compare_slices(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_a_col: str,
    y_pred_b_col: str,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare error rates between Model A and Model B for all slices.
    
    Args:
        df: DataFrame with features and targets
        y_true_col: True label column
        y_pred_a_col: Model A predictions
        y_pred_b_col: Model B predictions
        feature_cols: List of features to slice on
        
    Returns:
        DataFrame with slice rules and error improvement
    """
    # 1. Generate slices for Model A
    # Construct DF for slicing logic
    df_a = df.copy()
    df_a["is_error"] = (df[y_true_col] != df[y_pred_a_col])
    df_a["error_type"] = np.where(df_a["is_error"], "Error", "Correct") # specific type not critical here
    
    slices_a = generate_all_slices(
        df_a, 
        feature_cols=feature_cols, 
        exclude_cols=[y_true_col, y_pred_a_col, y_pred_b_col, "is_error", "error_type", "id"],
        is_error_col="is_error",
        error_type_col="error_type"
    )
    
    # 2. Generate slices for Model B
    df_b = df.copy()
    df_b["is_error"] = (df[y_true_col] != df[y_pred_b_col])
    df_b["error_type"] = np.where(df_b["is_error"], "Error", "Correct")
    
    slices_b = generate_all_slices(
        df_b, 
        feature_cols=feature_cols, 
        exclude_cols=[y_true_col, y_pred_a_col, y_pred_b_col, "is_error", "error_type", "id"],
        is_error_col="is_error",
        error_type_col="error_type"
    )
    
    # 3. Merge by rule
    dict_a = {s["rule"]: s for s in slices_a}
    dict_b = {s["rule"]: s for s in slices_b}
    
    comparison = []
    
    for rule, s_a in dict_a.items():
        if rule in dict_b:
            s_b = dict_b[rule]
            
            # Improvement: +ve means Model B has LOWER error rate (good)
            # or technically improvement is (Error A - Error B)
            error_diff = s_a["error_rate"] - s_b["error_rate"] 
            
            comparison.append({
                "rule": rule,
                "feature": s_a["feature"],
                "value": s_a["value"],
                "support": s_a["support"],
                "error_rate_a": s_a["error_rate"],
                "error_rate_b": s_b["error_rate"],
                "improvement": error_diff,
                "relative_improvement": error_diff / s_a["error_rate"] if s_a["error_rate"] > 0 else 0
            })
            
    # Convert to DF and sort by biggest improvement
    comp_df = pd.DataFrame(comparison)
    if not comp_df.empty:
        comp_df = comp_df.sort_values("improvement", ascending=False)
        
    return comp_df
