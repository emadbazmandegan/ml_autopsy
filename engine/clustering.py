"""
Clustering Module for Model Autopsy

Groups errors into distinct failure modes using KMeans clustering.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def filter_errors(
    df: pd.DataFrame,
    is_error_col: str = "is_error",
) -> pd.DataFrame:
    """
    Filter to misclassified samples only.
    
    Args:
        df: Canonical DataFrame
        is_error_col: Column indicating error
        
    Returns:
        DataFrame with only error rows
    """
    return df[df[is_error_col] == True].copy()


def preprocess_for_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, LabelEncoder]]:
    """
    Preprocess features for clustering (scale numeric, encode categorical).
    
    Args:
        df: DataFrame with features
        feature_cols: Feature columns to use
        
    Returns:
        Tuple of (preprocessed array, feature names, encoders)
    """
    X = df[feature_cols].copy()
    encoders = {}
    
    # Encode categorical columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("MISSING"))
            encoders[col] = le
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, list(X.columns), encoders


def cluster_errors(
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster errors using KMeans.
    
    Args:
        X: Preprocessed feature matrix
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (cluster labels, fitted KMeans model)
    """
    kmeans = KMeans(
        n_clusters=min(n_clusters, len(X)),
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def project_to_2d(
    X: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project features to 2D using PCA.
    
    Args:
        X: Preprocessed feature matrix
        random_state: Random seed
        
    Returns:
        2D projection array
    """
    n_components = min(2, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_2d = pca.fit_transform(X)
    
    # Pad with zeros if only 1 component
    if X_2d.shape[1] == 1:
        X_2d = np.hstack([X_2d, np.zeros((X_2d.shape[0], 1))])
    
    return X_2d


def generate_cluster_profiles(
    errors_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    error_type_col: str = "error_type",
) -> List[Dict[str, Any]]:
    """
    Generate interpretable profiles for each cluster.
    
    Args:
        errors_df: Error-only DataFrame
        labels: Cluster labels
        feature_cols: Feature columns used
        error_type_col: Error type column
        
    Returns:
        List of cluster profile dictionaries
    """
    errors_df = errors_df.copy()
    errors_df["cluster"] = labels
    
    profiles = []
    
    for cluster_id in sorted(errors_df["cluster"].unique()):
        cluster_df = errors_df[errors_df["cluster"] == cluster_id]
        
        # Size
        size = len(cluster_df)
        size_pct = size / len(errors_df)
        
        # Dominant error type
        error_counts = cluster_df[error_type_col].value_counts()
        dominant_error = error_counts.index[0] if len(error_counts) > 0 else "unknown"
        dominant_error_pct = error_counts.iloc[0] / size if len(error_counts) > 0 else 0
        
        # Top differentiating features (compare cluster mean to overall mean)
        top_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(errors_df[col]):
                cluster_mean = cluster_df[col].mean()
                overall_mean = errors_df[col].mean()
                overall_std = errors_df[col].std()
                if overall_std > 0:
                    z_diff = (cluster_mean - overall_mean) / overall_std
                    top_features.append({
                        "feature": col,
                        "cluster_mean": float(cluster_mean),
                        "overall_mean": float(overall_mean),
                        "z_diff": float(z_diff),
                    })
        
        # Sort by absolute z-score
        top_features.sort(key=lambda x: abs(x["z_diff"]), reverse=True)
        
        # Representative examples (first 3 indices)
        example_indices = cluster_df.index[:3].tolist()
        
        profiles.append({
            "cluster_id": int(cluster_id),
            "size": int(size),
            "size_pct": float(size_pct),
            "dominant_error": str(dominant_error),
            "dominant_error_pct": float(dominant_error_pct),
            "top_features": top_features[:5],
            "example_indices": example_indices,
        })
    
    return profiles


def plot_cluster_map(
    X_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Error Cluster Map",
    figsize: tuple = (10, 8),
) -> Path:
    """
    Generate and save a 2D cluster visualization.
    
    Args:
        X_2d: 2D projection of features
        labels: Cluster labels
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[i]],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            s=50,
        )
    
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def run_error_clustering(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    n_clusters: int = 5,
    random_state: int = 42,
    is_error_col: str = "is_error",
    error_type_col: str = "error_type",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], np.ndarray]:
    """
    Run the full error clustering pipeline.
    
    Args:
        df: Canonical DataFrame
        feature_cols: Feature columns (auto-detect if None)
        exclude_cols: Columns to exclude
        n_clusters: Number of clusters
        random_state: Random seed
        is_error_col: Error indicator column
        error_type_col: Error type column
        
    Returns:
        Tuple of (errors_df with labels, cluster profiles, 2D projection)
    """
    # Default exclusions
    default_exclude = {
        "y_true", "y_pred", "y_score", is_error_col, error_type_col,
        "id", "index", "_id", "cluster"
    }
    exclude_set = default_exclude | set(exclude_cols or [])
    
    # Filter to errors
    errors_df = filter_errors(df, is_error_col)
    
    if len(errors_df) == 0:
        # No errors to cluster
        return errors_df, [], np.array([])
    
    # Auto-detect feature columns
    if feature_cols is None:
        feature_cols = [c for c in errors_df.columns if c not in exclude_set]
    
    if not feature_cols:
        return errors_df, [], np.array([])
    
    # Preprocess
    X, encoded_features, encoders = preprocess_for_clustering(errors_df, feature_cols)
    
    # Cluster
    labels, kmeans = cluster_errors(X, n_clusters=n_clusters, random_state=random_state)
    
    # Add labels to dataframe
    errors_df["cluster"] = labels
    
    # Project to 2D
    X_2d = project_to_2d(X, random_state=random_state)
    
    # Generate profiles
    profiles = generate_cluster_profiles(errors_df, labels, feature_cols, error_type_col)
    
    return errors_df, profiles, X_2d


def save_clustering_results(
    errors_df: pd.DataFrame,
    profiles: List[Dict[str, Any]],
    X_2d: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save all clustering artifacts.
    
    Args:
        errors_df: Errors with cluster labels
        profiles: Cluster profiles
        X_2d: 2D projection
        labels: Cluster labels
        output_dir: Output directory
        
    Returns:
        Dictionary of artifact paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save clusters CSV
    clusters_path = output_dir / "clusters.csv"
    errors_df.to_csv(clusters_path, index=False)
    paths["clusters"] = clusters_path
    
    # Save profiles JSON
    profiles_path = output_dir / "cluster_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2, default=str)
    paths["profiles"] = profiles_path
    
    # Save cluster map
    if len(X_2d) > 0:
        map_path = output_dir / "cluster_map.png"
        plot_cluster_map(X_2d, labels, map_path)
        paths["map"] = map_path
    
    return paths
