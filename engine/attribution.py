"""
Attribution Module for Model Autopsy

Explains what features drive errors using permutation importance and delta attribution.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def preprocess_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, LabelEncoder]]:
    """
    Preprocess features for modeling.
    
    Args:
        df: DataFrame with features
        feature_cols: Feature columns to use
        
    Returns:
        Tuple of (preprocessed array, feature names, encoders)
    """
    X = df[feature_cols].copy()
    encoders = {}
    
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("MISSING"))
            encoders[col] = le
    
    X = X.fillna(X.median(numeric_only=True))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, list(X.columns), encoders


def train_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a surrogate model to predict is_error.
    
    Args:
        X: Feature matrix
        y: Binary target (is_error)
        random_state: Random seed
        
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
    )
    model.fit(X, y)
    return model


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance for features.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target
        feature_names: Feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with feature importance scores
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='accuracy',
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    })
    
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


def compute_subset_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    subset_mask: pd.Series,
    target_col: str = "is_error",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute importance on a subset of data.
    
    Args:
        df: Full DataFrame
        feature_cols: Feature columns
        subset_mask: Boolean mask for subset
        target_col: Target column
        n_repeats: Permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with importance scores
    """
    subset_df = df[subset_mask].copy()
    
    if len(subset_df) < 10:
        # Too few samples
        return pd.DataFrame({
            'feature': feature_cols,
            'importance_mean': [0.0] * len(feature_cols),
            'importance_std': [0.0] * len(feature_cols),
            'rank': range(1, len(feature_cols) + 1),
        })
    
    X, features, encoders = preprocess_features(subset_df, feature_cols)
    y = subset_df[target_col].astype(int).values
    
    # Check for class balance
    if len(np.unique(y)) < 2:
        return pd.DataFrame({
            'feature': features,
            'importance_mean': [0.0] * len(features),
            'importance_std': [0.0] * len(features),
            'rank': range(1, len(features) + 1),
        })
    
    model = train_surrogate(X, y, random_state=random_state)
    importance = compute_permutation_importance(
        model, X, y, features,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    
    return importance


def compute_delta_attribution(
    global_importance: pd.DataFrame,
    error_importance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute delta attribution (error importance - global importance).
    
    Args:
        global_importance: Global importance DataFrame
        error_importance: Error-only importance DataFrame
        
    Returns:
        DataFrame with delta scores
    """
    # Merge on feature
    merged = global_importance[['feature', 'importance_mean']].merge(
        error_importance[['feature', 'importance_mean']],
        on='feature',
        suffixes=('_global', '_error'),
    )
    
    # Compute delta
    merged['delta'] = merged['importance_mean_error'] - merged['importance_mean_global']
    merged['abs_delta'] = merged['delta'].abs()
    
    # Sort by absolute delta
    merged = merged.sort_values('abs_delta', ascending=False)
    merged['rank'] = range(1, len(merged) + 1)
    
    return merged


def run_attribution_analysis(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    is_error_col: str = "is_error",
    n_repeats: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run full attribution analysis.
    
    Args:
        df: Canonical DataFrame
        feature_cols: Feature columns (auto-detect if None)
        exclude_cols: Columns to exclude
        is_error_col: Error indicator column
        n_repeats: Permutation repeats
        random_state: Random seed
        
    Returns:
        Tuple of (global_importance, error_importance, delta_attribution)
    """
    # Default exclusions
    default_exclude = {
        "y_true", "y_pred", "y_score", is_error_col, "error_type",
        "id", "index", "_id", "cluster"
    }
    exclude_set = default_exclude | set(exclude_cols or [])
    
    # Auto-detect feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude_set]
    
    if not feature_cols:
        empty_df = pd.DataFrame(columns=['feature', 'importance_mean', 'importance_std', 'rank'])
        return empty_df, empty_df, pd.DataFrame()
    
    # Preprocess all data
    X, features, encoders = preprocess_features(df, feature_cols)
    y = df[is_error_col].astype(int).values
    
    # Check class balance
    if len(np.unique(y)) < 2:
        empty_df = pd.DataFrame({
            'feature': features,
            'importance_mean': [0.0] * len(features),
            'importance_std': [0.0] * len(features),
            'rank': range(1, len(features) + 1),
        })
        return empty_df, empty_df, pd.DataFrame()
    
    # Global importance
    global_model = train_surrogate(X, y, random_state=random_state)
    global_importance = compute_permutation_importance(
        global_model, X, y, features,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    
    # Error-only importance
    error_mask = df[is_error_col] == True
    error_importance = compute_subset_importance(
        df, feature_cols, error_mask,
        target_col=is_error_col,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    
    # Delta attribution
    delta = compute_delta_attribution(global_importance, error_importance)
    
    return global_importance, error_importance, delta


def plot_attribution(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str = "Feature Importance",
    top_n: int = 15,
    figsize: tuple = (10, 8),
) -> Path:
    """
    Generate bar chart of top features.
    
    Args:
        importance_df: DataFrame with importance scores
        output_path: Path to save plot
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    # Get top N
    plot_df = importance_df.head(top_n).copy()
    
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data to display", ha='center', va='center')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return Path(output_path)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color column
    if 'delta' in plot_df.columns:
        colors = ['#e74c3c' if d > 0 else '#3498db' for d in plot_df['delta']]
        values = plot_df['delta'].values
        xlabel = 'Delta Importance (Error - Global)'
    else:
        colors = '#3498db'
        values = plot_df['importance_mean'].values
        xlabel = 'Importance'
    
    y_pos = np.arange(len(plot_df))
    
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def save_attribution_results(
    global_importance: pd.DataFrame,
    error_importance: pd.DataFrame,
    delta: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save all attribution artifacts.
    
    Args:
        global_importance: Global importance DataFrame
        error_importance: Error importance DataFrame
        delta: Delta attribution DataFrame
        output_dir: Output directory
        
    Returns:
        Dictionary of artifact paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save CSVs
    global_path = output_dir / "attribution_global.csv"
    global_importance.to_csv(global_path, index=False)
    paths["global"] = global_path
    
    error_path = output_dir / "attribution_errors.csv"
    error_importance.to_csv(error_path, index=False)
    paths["errors"] = error_path
    
    # Save plots
    if len(global_importance) > 0:
        global_plot = output_dir / "attribution_global.png"
        plot_attribution(global_importance, global_plot, title="Global Feature Importance")
        paths["global_plot"] = global_plot
    
    if len(delta) > 0:
        delta_plot = output_dir / "attribution_delta.png"
        plot_attribution(delta, delta_plot, title="Delta Attribution (Error - Global)")
        paths["delta_plot"] = delta_plot
    
    return paths
