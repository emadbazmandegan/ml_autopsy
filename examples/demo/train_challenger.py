"""
Train RandomForest challenger model on UCI Adult Income dataset.

This script trains a RandomForestClassifier to compare against the
linear baseline (LogisticRegression) used in the original demo.

Run: python examples/demo/train_challenger.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# Configuration
DEMO_DIR = Path(__file__).parent
RANDOM_STATE = 42


def load_and_prepare_data():
    """Load demo dataset and prepare features."""
    df = pd.read_csv(DEMO_DIR / "demo_dataset.csv")
    
    # Target (income) - Create y_true if it exists or use predictions
    if "income" in df.columns:
        y_true = (df["income"].str.strip() == ">50K").astype(int)
    else:
        # Fallback
        preds = pd.read_csv(DEMO_DIR / "demo_predictions.csv")
        y_true = preds["y_true"].values

    # Features for model (Matches Adult Dataset)
    feature_cols = ["age", "workclass", "education", "occupation", 
                    "sex", "hours_per_week", "race", "marital_status"]
    
    # Filter only available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()
    
    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna("Unknown"))
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_true, df


def train_challenger():
    """Train RandomForest challenger and generate predictions."""
    print("Loading UCI Adult Income dataset...")
    X, y_true, df = load_and_prepare_data()
    
    print(f"Dataset: {len(df)} samples")
    
    print("Training RandomForest challenger (realistic)...")
    # Using confined parameters to prevent 100% overfitting on small data
    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=3,
        min_samples_split=5,
        random_state=RANDOM_STATE
    )
    model.fit(X, y_true)
    
    # Generate predictions
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"Challenger accuracy: {accuracy:.1%}")
    
    # Save predictions
    predictions = pd.DataFrame({
        "y_pred": y_pred,
        "y_score": np.round(y_score, 4),
        "y_true": y_true
    })
    
    output_path = DEMO_DIR / "demo_predictions_rf.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Saved challenger predictions to {output_path}")
    
    return predictions


if __name__ == "__main__":
    train_challenger()
