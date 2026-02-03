"""
Train baseline model on UCI Adult Income dataset and save predictions for demo.

This script trains a LogisticRegression model on the real UCI Adult dataset
and generates predictions for the demo sample.

Run: python examples/demo/train_baseline.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuration
DEMO_DIR = Path(__file__).parent
RANDOM_STATE = 42


def load_and_prepare_data():
    """Load demo dataset and prepare features."""
    df = pd.read_csv(DEMO_DIR / "demo_dataset.csv")
    
    # Target: income column (>50K = 1, <=50K = 0)
    df["y_true"] = (df["income"].str.strip() == ">50K").astype(int)
    
    # Features for model
    feature_cols = ["age", "workclass", "education", "occupation", 
                    "sex", "hours_per_week", "race", "marital_status"]
    
    # Encode categorical features
    X = df[feature_cols].copy()
    encoders = {}
    
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna("Unknown"))
            encoders[col] = le
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df["y_true"].values, df


def train_baseline():
    """Train LogisticRegression baseline and generate predictions."""
    print("Loading UCI Adult Income dataset...")
    X, y_true, df = load_and_prepare_data()
    
    print(f"Dataset: {len(df)} samples, {y_true.sum()} high-income (>50K)")
    
    # Train logistic regression
    print("Training LogisticRegression baseline...")
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X, y_true)
    
    # Generate predictions
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"Training accuracy: {accuracy:.1%}")
    
    # Save predictions
    predictions = pd.DataFrame({
        "y_pred": y_pred,
        "y_score": np.round(y_score, 4),
        "y_true": y_true
    })
    
    output_path = DEMO_DIR / "demo_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    return predictions


if __name__ == "__main__":
    train_baseline()
