"""
Financial Fraud Detection Prototype
Run from the repository root:
    python src/main.py
Or provide the dataset path:
    python src/main.py path/to/fraud_detection_kaggle_style_1000_rows.csv

This script mirrors the main preprocessing and model-comparison workflow
in notebooks/Experiment_6_Financial_Fraud_Detection.ipynb.
It writes results/experiment6_model_comparison.csv.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE


DEFAULT_DATASET = "fraud_detection_kaggle_style_1000_rows.csv"
TARGET = "FraudIndicator"
FEATURES = [
    "Category",
    "TransactionAmount",
    "AnomalyScore",
    "Amount",
    "AccountBalance",
    "SuspiciousFlag",
    "Hour",
    "gap",
]


def prepare_data(csv_path: Path):
    """Load the notebook's dataset and reproduce its feature preparation."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"Place {DEFAULT_DATASET} in the repository root, or run:\n"
            f"python src/main.py path/to/{DEFAULT_DATASET}"
        )

    df = pd.read_csv(csv_path)
    required = [TARGET, "Timestamp", "LastLogin", *FEATURES[:6]]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    model_df = df.copy()
    model_df["Timestamp"] = pd.to_datetime(model_df["Timestamp"], errors="coerce")
    model_df["Hour"] = model_df["Timestamp"].dt.hour
    model_df["LastLogin"] = pd.to_datetime(model_df["LastLogin"], errors="coerce")
    model_df["gap"] = (
        model_df["Timestamp"] - model_df["LastLogin"]
    ).dt.days.abs()

    X = model_df[FEATURES].copy()
    y = model_df[TARGET].astype(int)

    # Match the notebook: encode Category, replace infinities, fill missing
    # feature values with each column's median.
    encoder = LabelEncoder()
    X["Category"] = encoder.fit_transform(X["Category"].astype(str))
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X, y, df.shape


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_DATASET)
    X, y, original_shape = prepare_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Match the notebook: apply SMOTE to the training partition only.
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        })

    results_df = pd.DataFrame(rows).set_index("Model")
    final_results = results_df.reset_index().round(4)

    print(f"Dataset shape: {original_shape}")
    print(f"Training records before SMOTE: {len(X_train)}")
    print(f"Training records after SMOTE:  {len(X_train_balanced)}")
    print(f"Testing records: {len(X_test)}")
    print("\nFINAL MODEL COMPARISON")
    print(final_results.to_string(index=False))
    print("\nModel with the highest F1-score:", results_df["F1-score"].idxmax())
    print("F1-score:", round(results_df["F1-score"].max(), 4))

    output_path = Path("results") / "experiment6_model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
