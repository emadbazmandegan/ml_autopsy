"""
Explanation Module for Model Autopsy

Generates natural language explanations for model predictions
based on feature deviations and similar samples.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def generate_explanation(
    sample_details: Dict[str, Any],
    deviations: pd.DataFrame,
    neighbors: List[Dict[str, Any]],
    target_col: str = "y_true",
    pred_col: str = "y_pred",
) -> str:
    """
    Generate a natural language explanation for a sample's classification.
    
    Args:
        sample_details: Dictionary with sample data
        deviations: DataFrame with feature z-scores
        neighbors: List of nearest neighbors
        target_col: Name of true label column
        pred_col: Name of prediction column
        
    Returns:
        String explanation
    """
    data = sample_details["data"]
    is_error = data.get(target_col) != data.get(pred_col)
    prediction = data.get(pred_col)
    
    explanation = []
    
    # 1. Basic Status
    status = "an ERROR" if is_error else "CORRECT"
    explanation.append(f"Model predicted **{prediction}**, which was **{status}**.")
    
    # 2. Significant Deviations
    if not deviations.empty:
        high_devs = deviations[deviations["abs_z_score"] > 1.5]
        if not high_devs.empty:
            top_devs = high_devs.head(3)
            feats = [f"**{row['feature']}** ({row['z_score']:+.1f}σ)" for _, row in top_devs.iterrows()]
            explanation.append(f"Distinctive features: {', '.join(feats)}.")
        else:
            explanation.append("This sample is very typical (no features > 1.5σ from mean).")
    
    # 3. Neighbor Context
    if neighbors:
        neighbor_preds = [n["data"].get(pred_col) for n in neighbors]
        avg_pred = sum(neighbor_preds) / len(neighbor_preds)
        
        if is_error:
            # If 4/5 neighbors were also predicted wrong (same as this one), it's a region failure
            matching_preds = sum(1 for p in neighbor_preds if p == prediction)
            if matching_preds >= len(neighbors) * 0.8:
                explanation.append(f"Consistent confusion: {matching_preds}/{len(neighbors)} similar samples were also predicted as {prediction}.")
            else:
                 explanation.append(f"Inconsistent boundary: Similar samples have mixed predictions (avg: {avg_pred:.2f}).")
    
    return " ".join(explanation)
