#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-OV Ovarian Cancer Proteomics Best Model Independent Prediction Script
Subgroup: pharma_yes (patients who received drug treatment)
Omics: Proteomics (RPPA Protein Expression)
Best Model: XGBoost (automatically detected from current files)

Model directory:
    model/proteomics/

Prediction results saved to:
    predictions/proteomics/

Usage:
    python predict_proteomics.py --test your_test_proteomics.csv

Input file requirements:
    - CSV file
    - Column names must exactly match training protein features (e.g., SHP2_pY542, p38-a, YAP, etc.)
    - Optional: include 'label' column for offline evaluation
    - Optional: include 'submitter_id' column for sample identification
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import logging
import warnings

warnings.filterwarnings("ignore")

# ============================= Configuration =============================
MODEL_DIR       = "model/proteomics"           # Model components directory
PREDICTION_DIR  = "predictions/proteomics"     # Prediction output directory
MODEL_NAME      = "XGBoost"                    # Current best model
SUBGROUP        = "pharma_yes"
OMICS_TYPE      = "proteomics"
# =======================================================================

def setup_logging():
    """Initialize logging to console and file."""
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    log_file = os.path.join(PREDICTION_DIR, "predict_proteomics.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info("Proteomics prediction logging initialized")
    logging.info(f"Prediction results will be saved to: {PREDICTION_DIR}")


def load_model_components():
    """Load best model, scaler, and feature list."""
    logging.info("Loading proteomics best model components...")

    model_path    = os.path.join(MODEL_DIR, f"best_model_{MODEL_NAME}_{SUBGROUP}_{OMICS_TYPE}.pkl")
    scaler_path   = os.path.join(MODEL_DIR, f"best_scaler_{MODEL_NAME}_{SUBGROUP}_{OMICS_TYPE}.pkl")
    features_path = os.path.join(MODEL_DIR, f"best_features_{MODEL_NAME}_{SUBGROUP}_{OMICS_TYPE}.csv")

    for path, name in [
        (model_path,    "model file"),
        (scaler_path,   "scaler file"),
        (features_path, "feature list file")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Cannot find {name}\n    → {path}")

    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    features = pd.read_csv(features_path)['feature_name'].tolist()

    logging.info(f"Best model loaded successfully: {MODEL_NAME}")
    logging.info(f"Number of protein features used: {len(features)}")
    return model, scaler, features


def preprocess_test_data(df: pd.DataFrame, required_features: list, scaler):
    """Exactly reproduce the preprocessing pipeline used during training."""
    logging.info("Preprocessing test proteomics data...")

    has_label = 'label' in df.columns
    has_id    = 'submitter_id' in df.columns

    if has_label:
        y_true = df['label'].astype(int).values
        X = df.drop(columns=['label'])
        logging.info(f"True labels detected, {len(y_true)} samples")
    else:
        y_true = None
        X = df.copy()
        logging.info(f"No true labels provided, performing prediction only on {len(X)} samples")

    # Check for missing required features
    missing = set(required_features) - set(X.columns)
    if missing:
        raise KeyError(
            f"Test data missing {len(missing)} protein features used in training!\n"
            f"First 10 missing: {list(missing)[:10]}"
        )

    extra = set(X.columns) - set(required_features)
    if extra:
        logging.warning(f"Test data contains {len(extra)} extra columns (will be ignored)")

    # Keep only training features in exact order
    X = X[required_features]

    # Fill missing values with training median (already encoded in scaler)
    X = X.fillna(X.median(numeric_only=True))

    # Apply the same RobustScaler fitted during training
    X_scaled = scaler.transform(X)

    logging.info(f"Preprocessing complete → Final feature matrix shape: {X_scaled.shape}")
    return X_scaled, y_true, df['submitter_id'] if has_id else None


def predict(model, X_scaled):
    """Perform prediction using XGBoost."""
    logging.info("Predicting with XGBoost model (proteomics)...")
    y_pred = model.predict(X_scaled).flatten()
    y_proba = model.predict_proba(X_scaled)[:, 1]
    return y_pred.astype(int), y_proba


def main():
    parser = argparse.ArgumentParser(
        description="TCGA-OV Proteomics-based Drug Response Prediction (pharma_yes subgroup)"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test CSV file (must contain all protein features used in training)"
    )
    args = parser.parse_args()

    setup_logging()

    # 1. Load model components
    model, scaler, features = load_model_components()

    # 2. Load test data
    if not os.path.exists(args.test):
        raise FileNotFoundError(f"Test file not found: {args.test}")
    test_df = pd.read_csv(args.test)
    logging.info(f"Test data loaded successfully: {args.test}, samples: {len(test_df)}")

    # 3. Preprocess + predict
    X_test_scaled, y_true, submitter_ids = preprocess_test_data(test_df, features, scaler)
    y_pred, y_proba = predict(model, X_test_scaled)

    # 4. Assemble and save results
    result_df = pd.DataFrame({
        'submitter_id': submitter_ids if submitter_ids is not None else test_df.index,
        'predicted_label': y_pred,
        'predicted_probability': y_proba.round(6)
    })
    if y_true is not None:
        result_df['true_label'] = y_true

    output_csv = os.path.join(PREDICTION_DIR, "proteomics_prediction_results.csv")
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"Prediction completed! Results saved to: {output_csv}")

    # 5. Print performance metrics if true labels are available
    if y_true is not None:
        logging.info("\n=== Test Set Performance Metrics ===")
        logging.info(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
        logging.info(f"Precision    : {precision_score(y_true, y_pred):.4f}")
        logging.info(f"Recall       : {recall_score(y_true, y_pred):.4f}")
        logging.info(f"F1 Score     : {f1_score(y_true, y_pred):.4f}")
        logging.info(f"AUC          : {roc_auc_score(y_true, y_proba):.4f}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == "__main__":
    main()