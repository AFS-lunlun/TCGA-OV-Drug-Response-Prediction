#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-OV Ovarian Cancer DNA Methylation Best Model Prediction Script (CatBoost)
Project: TCGA-OV Drug Treatment Response Prediction
Subgroup: pharma_yes (patients who received drug treatment)
Omics: DNA Methylation
Best Model: CatBoost (pre-trained and saved)

Usage:
    git clone https://github.com/yourname/OV-Methylation-Response-Prediction.git
    cd OV-Methylation-Response-Prediction
    pip install pandas numpy catboost scikit-learn joblib
    python predict.py --test your_test_data.csv

Author: Alice
Date: 2025-04-05
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import logging
import warnings

warnings.filterwarnings("ignore")


# ============================= Configuration =============================
MODEL_DIR = "model/methylation/"                      # Directory containing trained model files
PREDICTION_DIR = "predictions/methylation/"           # Directory to save prediction results
# ========================================================================


def setup_logging():
    """Configure logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(PREDICTION_DIR, "predict.log"),
                encoding='utf-8'
            )
        ]
    )


def load_model_components():
    """Load the trained model, scaler, and feature list."""
    logging.info("Loading model components...")

    model_path    = os.path.join(MODEL_DIR, "best_model_CatBoost_pharma_yes_methylation.pkl")
    scaler_path   = os.path.join(MODEL_DIR, "best_scaler_CatBoost_pharma_yes_methylation.pkl")
    features_path = os.path.join(MODEL_DIR, "best_features_CatBoost_pharma_yes_methylation.csv")

    for path, name in [
        (model_path,    "model file"),
        (scaler_path,   "scaler file"),
        (features_path, "feature list file")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name}: {path}")

    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    features = pd.read_csv(features_path)['feature_name'].tolist()

    logging.info(f"Model loaded successfully | Number of features: {len(features)}")
    return model, scaler, features


def preprocess_test_data(df: pd.DataFrame, features: list, scaler):
    """Apply exactly the same preprocessing as during training."""
    logging.info("Preprocessing test data...")

    has_label = 'label' in df.columns
    if has_label:
        y_true = df['label'].values.astype(int)
        X = df.drop(columns=['label'])
    else:
        y_true = None
        X = df.copy()

    # Ensure all required training features are present
    missing = set(features) - set(X.columns)
    if missing:
        raise KeyError(
            f"Test data is missing {len(missing)} required features. "
            f"First 10 missing: {list(missing)[:10]}"
        )

    # Keep only the features used during training (in correct order)
    X = X[features]

    # Same preprocessing as training: median imputation → log2(x+1) → RobustScaler
    X = X.fillna(X.median(numeric_only=True))
    X = np.log2(X + 1)
    X_scaled = scaler.transform(X)

    logging.info(f"Preprocessing complete. Final shape: {X_scaled.shape}")
    return X_scaled, y_true


def predict(model, X_scaled):
    """Generate predictions and probabilities."""
    pred = model.predict(X_scaled).flatten()
    proba = model.predict_proba(X_scaled)[:, 1]
    return pred, proba


def main(test_csv_path: str):
    # Ensure output directories exist
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    setup_logging()

    # 1. Load model components
    model, scaler, features = load_model_components()

    # 2. Load test data
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test file not found: {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    logging.info(f"Test data loaded: {test_csv_path} | Samples: {len(test_df)}")

    # 3. Preprocess + predict
    X_test, y_true = preprocess_test_data(test_df, features, scaler)
    y_pred, y_proba = predict(model, X_test)

    # 4. Save results
    result_df = pd.DataFrame({
        'predicted_label': y_pred.astype(int),
        'predicted_probability': y_proba.round(6)
    })
    if y_true is not None:
        result_df['true_label'] = y_true

    output_path = os.path.join(PREDICTION_DIR, "prediction_results.csv")
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"Prediction completed! Results saved to: {output_path}")

    # 5. Print performance metrics if ground truth is available
    if y_true is not None:
        logging.info("\n=== Test Set Performance ===")
        logging.info(f"Accuracy   : {accuracy_score(y_true, y_pred):.4f}")
        logging.info(f"Precision  : {precision_score(y_true, y_pred):.4f}")
        logging.info(f"Recall     : {recall_score(y_true, y_pred):.4f}")
        logging.info(f"F1 Score   : {f1_score(y_true, y_pred):.4f}")
        logging.info(f"AUC        : {roc_auc_score(y_true, y_proba):.4f}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TCGA-OV Methylation-based Drug Response Prediction (Best CatBoost Model)"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test CSV file (must contain the same methylation features as training; label column optional)"
    )
    args = parser.parse_args()

    main(args.test)