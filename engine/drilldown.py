"""
Drilldown Module for Model Autopsy

Functionality for inspecting individual samples, finding neighbors,
and analyzing feature deviations.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def get_sample_details(
    df: pd.DataFrame,
    index: int,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get detailed information for a specific sample.
    
    Args:
        df: The dataframe containing samples
        index: Index of the sample to retrieve
        feature_cols: List of feature columns (optional)
        
    Returns:
        Dictionary with sample details
    """
    if index not in df.index:
        raise ValueError(f"Index {index} not found in dataframe")
    
    row = df.loc[index]
    
    # Basic info
    details = {
        "index": int(index),
        "data": row.to_dict(),
    }
    
    # Filter to specific features if requested
    if feature_cols:
        details["features"] = {k: row[k] for k in feature_cols if k in row}
    
    return details


def compute_feature_deviation(
    df: pd.DataFrame,
    index: int,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute z-score deviation for numeric features of a sample.
    
    Args:
        df: Dataframe with all samples
        index: Index of sample to analyze
        feature_cols: List of numeric feature columns
        
    Returns:
        DataFrame with feature values, means, and z-scores
    """
    if index not in df.index:
        raise ValueError(f"Index {index} not found in dataframe")
    
    sample = df.loc[index]
    
    # Filter to numeric columns only
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if not numeric_cols:
        return pd.DataFrame()
    
    # Compute stats
    stats = []
    for col in numeric_cols:
        val = sample[col]
        mean = df[col].mean()
        std = df[col].std()
        
        z_score = (val - mean) / std if std > 0 else 0
        
        stats.append({
            "feature": col,
            "value": val,
            "mean": mean,
            "std": std,
            "z_score": z_score,
            "abs_z_score": abs(z_score)
        })
        
    return pd.DataFrame(stats).sort_values("abs_z_score", ascending=False)


def find_nearest_neighbors(
    df: pd.DataFrame,
    index: int,
    feature_cols: List[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find nearest neighbors in feature space.
    
    Args:
        df: Dataframe with all samples
        index: Target sample index
        feature_cols: List of numeric feature columns to use for distance
        k: Number of neighbors to find
        
    Returns:
        List of neighbor dictionaries with distance and index
    """
    if index not in df.index:
        raise ValueError(f"Index {index} not found in dataframe")
    
    # Filter numeric columns and handle missing data
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return []
        
    X = df[numeric_cols].fillna(0).values
    
    # Fit KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_scaled)
    
    # Find neighbors for target
    target_loc = df.index.get_loc(index)
    distances, indices = nbrs.kneighbors(X_scaled[target_loc].reshape(1, -1))
    
    results = []
    # Skip first one (it's the sample itself)
    for dist, idx in zip(distances[0][1:], indices[0][1:]):
        original_idx = df.index[idx]
        results.append({
            "index": int(original_idx),
            "distance": float(dist),
            "data": df.loc[original_idx].to_dict()
        })
        
    return results
