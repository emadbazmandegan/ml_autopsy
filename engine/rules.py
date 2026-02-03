"""
Rules Module for Model Autopsy

Discovers hidden failure subgroups using decision tree rule mining.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def train_error_tree(
    df: pd.DataFrame,
    feature_cols: List[str],
    is_error_col: str = "is_error",
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> Tuple[DecisionTreeClassifier, Dict[str, LabelEncoder], List[str]]:
    """
    Train a shallow decision tree to predict is_error.
    
    Args:
        df: Canonical DataFrame
        feature_cols: Feature columns to use
        is_error_col: Target column
        max_depth: Maximum tree depth for interpretability
        min_samples_leaf: Minimum samples per leaf
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained tree, label encoders dict, encoded feature names)
    """
    # Prepare features
    X = df[feature_cols].copy()
    y = df[is_error_col].astype(int)
    
    # Encode categorical columns
    encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Train tree
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(X, y)
    
    return tree, encoders, list(X.columns)


def extract_rules(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    encoders: Dict[str, LabelEncoder],
    min_samples: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extract decision rules from a trained tree as human-readable strings.
    
    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: Names of features used
        encoders: Label encoders for categorical features
        min_samples: Minimum samples for a rule to be included
        
    Returns:
        List of rule dictionaries
    """
    tree_ = tree.tree_
    feature_name = feature_names
    
    rules = []
    
    def recurse(node, conditions):
        if tree_.feature[node] == -2:  # Leaf node
            samples = tree_.n_node_samples[node]
            if samples >= min_samples:
                # Error rate in this leaf
                values = tree_.value[node][0]
                error_count = int(values[1]) if len(values) > 1 else 0
                total = int(values.sum())
                error_rate = error_count / total if total > 0 else 0
                
                # Only include leaves with above-average error rate
                if error_rate > 0:
                    rule_str = " AND ".join(conditions) if conditions else "ALL"
                    rules.append({
                        "rule": rule_str,
                        "conditions": conditions.copy(),
                        "samples": samples,
                        "error_count": error_count,
                        "error_rate": error_rate,
                    })
            return
        
        # Get feature info
        feat_idx = tree_.feature[node]
        feat_name = feature_name[feat_idx]
        threshold = tree_.threshold[node]
        
        # Format threshold for readability
        if feat_name in encoders:
            # Categorical: decode threshold
            le = encoders[feat_name]
            try:
                thresh_int = int(threshold)
                if thresh_int < len(le.classes_):
                    left_cond = f"{feat_name} = {le.classes_[thresh_int]}"
                    right_cond = f"{feat_name} != {le.classes_[thresh_int]}"
                else:
                    left_cond = f"{feat_name} <= {threshold:.0f}"
                    right_cond = f"{feat_name} > {threshold:.0f}"
            except:
                left_cond = f"{feat_name} <= {threshold:.2f}"
                right_cond = f"{feat_name} > {threshold:.2f}"
        else:
            # Numeric: use threshold directly
            left_cond = f"{feat_name} <= {threshold:.2f}"
            right_cond = f"{feat_name} > {threshold:.2f}"
        
        # Recurse left
        recurse(tree_.children_left[node], conditions + [left_cond])
        # Recurse right
        recurse(tree_.children_right[node], conditions + [right_cond])
    
    recurse(0, [])
    return rules


def apply_rule(
    df: pd.DataFrame,
    rule: Dict[str, Any],
    feature_cols: List[str],
) -> pd.Series:
    """
    Get boolean mask of rows matching a rule.
    
    Args:
        df: Canonical DataFrame
        rule: Rule dictionary with 'conditions'
        feature_cols: Feature columns
        
    Returns:
        Boolean mask
    """
    conditions = rule.get("conditions", [])
    
    if not conditions or rule.get("rule") == "ALL":
        return pd.Series([True] * len(df), index=df.index)
    
    mask = pd.Series([True] * len(df), index=df.index)
    
    for cond in conditions:
        try:
            # Parse condition
            if " <= " in cond:
                parts = cond.split(" <= ")
                feat, val = parts[0].strip(), float(parts[1].strip())
                mask &= df[feat] <= val
            elif " > " in cond:
                parts = cond.split(" > ")
                feat, val = parts[0].strip(), float(parts[1].strip())
                mask &= df[feat] > val
            elif " = " in cond:
                parts = cond.split(" = ")
                feat, val = parts[0].strip(), parts[1].strip()
                mask &= df[feat].astype(str) == val
            elif " != " in cond:
                parts = cond.split(" != ")
                feat, val = parts[0].strip(), parts[1].strip()
                mask &= df[feat].astype(str) != val
        except Exception:
            # Invalid condition, skip
            continue
    
    return mask


def compute_rule_metrics(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]],
    feature_cols: List[str],
    is_error_col: str = "is_error",
    error_type_col: str = "error_type",
) -> List[Dict[str, Any]]:
    """
    Compute full metrics for each rule.
    
    Args:
        df: Canonical DataFrame
        rules: List of rule dictionaries
        feature_cols: Feature columns
        is_error_col: Error indicator column
        error_type_col: Error type column
        
    Returns:
        Rules with full metrics
    """
    total_samples = len(df)
    global_error_rate = df[is_error_col].sum() / total_samples if total_samples > 0 else 0
    
    results = []
    
    for rule in rules:
        mask = apply_rule(df, rule, feature_cols)
        subset = df[mask]
        
        support = len(subset)
        support_pct = support / total_samples if total_samples > 0 else 0
        
        if support > 0:
            error_count = subset[is_error_col].sum()
            error_rate = error_count / support
            lift = error_rate / global_error_rate if global_error_rate > 0 else 0
            
            # Error type breakdown
            error_types = subset[error_type_col].value_counts().to_dict()
            
            results.append({
                "rule": rule["rule"],
                "conditions": rule.get("conditions", []),
                "support": int(support),
                "support_pct": float(support_pct),
                "error_count": int(error_count),
                "error_rate": float(error_rate),
                "lift": float(lift),
                "fp_count": int(error_types.get("FP", 0)),
                "fn_count": int(error_types.get("FN", 0)),
                "source": "rule",
            })
    
    return results


def discover_rule_slices(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    min_support: int = 10,
    random_state: int = 42,
    is_error_col: str = "is_error",
    error_type_col: str = "error_type",
) -> List[Dict[str, Any]]:
    """
    End-to-end rule slice discovery.
    
    Args:
        df: Canonical DataFrame
        feature_cols: Feature columns (auto-detect if None)
        exclude_cols: Columns to exclude
        max_depth: Tree max depth
        min_samples_leaf: Min samples per leaf
        min_support: Min support for rules
        random_state: Random seed
        is_error_col: Error indicator column
        error_type_col: Error type column
        
    Returns:
        List of rule slice dictionaries with metrics
    """
    # Default exclusions
    default_exclude = {
        "y_true", "y_pred", "y_score", is_error_col, error_type_col,
        "id", "index", "_id"
    }
    exclude_set = default_exclude | set(exclude_cols or [])
    
    # Auto-detect feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude_set]
    
    if not feature_cols:
        return []
    
    # Train tree
    tree, encoders, encoded_names = train_error_tree(
        df, feature_cols,
        is_error_col=is_error_col,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    
    # Extract rules
    rules = extract_rules(tree, encoded_names, encoders, min_samples=min_support)
    
    # Compute metrics
    rule_slices = compute_rule_metrics(
        df, rules, feature_cols,
        is_error_col=is_error_col,
        error_type_col=error_type_col,
    )
    
    return rule_slices


def merge_and_rank_slices(
    simple_slices: List[Dict[str, Any]],
    rule_slices: List[Dict[str, Any]],
    min_support: int = 10,
    min_support_pct: float = 0.01,
    sort_by: str = "lift",
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Merge simple slices with rule slices and rank.
    
    Args:
        simple_slices: Slices from Phase 3
        rule_slices: Slices from rule mining
        min_support: Minimum support filter
        min_support_pct: Minimum support percentage
        sort_by: Sort field
        top_n: Limit results
        
    Returns:
        Merged and ranked slices
    """
    # Add source to simple slices if not present
    for s in simple_slices:
        if "source" not in s:
            s["source"] = "simple"
    
    # Combine
    all_slices = simple_slices + rule_slices
    
    # Filter
    filtered = [
        s for s in all_slices
        if s.get("support", 0) >= min_support and s.get("support_pct", 0) >= min_support_pct
    ]
    
    # Sort
    sorted_slices = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=True)
    
    # Limit
    if top_n is not None:
        sorted_slices = sorted_slices[:top_n]
    
    # Add rank
    for i, s in enumerate(sorted_slices):
        s["rank"] = i + 1
    
    return sorted_slices


def save_rule_slices(
    rule_slices: List[Dict[str, Any]],
    merged_slices: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save rule slices and merged slices to CSV.
    
    Args:
        rule_slices: Rule-only slices
        merged_slices: Merged and ranked slices
        output_dir: Output directory
        
    Returns:
        Tuple of (rules_path, top_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rules_path = output_dir / "slices_rules.csv"
    top_path = output_dir / "slices_top.csv"
    
    # Save rule slices
    if rule_slices:
        rules_df = pd.DataFrame(rule_slices)
        # Drop conditions column for CSV (it's a list)
        if "conditions" in rules_df.columns:
            rules_df = rules_df.drop(columns=["conditions"])
        rules_df.to_csv(rules_path, index=False)
    else:
        pd.DataFrame().to_csv(rules_path, index=False)
    
    # Save merged slices
    if merged_slices:
        top_df = pd.DataFrame(merged_slices)
        if "conditions" in top_df.columns:
            top_df = top_df.drop(columns=["conditions"])
        top_df.to_csv(top_path, index=False)
    else:
        pd.DataFrame().to_csv(top_path, index=False)
    
    return rules_path, top_path
