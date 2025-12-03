#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-OV Ovarian Cancer Transcriptome Best Model Prediction Script (RandomForest)
Project: TCGA-OV Drug Treatment Response Prediction
Subgroup: pharma_yes (patients who received drug treatment)
Omics: Transcriptome (mRNA Expression)
Best Model: RandomForest (pre-trained and saved)

Model directory:
    model/transcriptome/

Usage:
    python predict_transcriptome.py --test test_transcriptome_raw_selected.csv

Input file requirements:
    - CSV file
    - Column names must exactly match training features (e.g., transcript_ENSG00000123456)
    - Optional: include a column named "label" for performance evaluation
    - Optional: include "submitter_id" for sample identification

Author: Alice
Date: 2025-04-06
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
MODEL_DIR       = "model/transcriptome"           # Model components directory
PREDICTION_DIR  = "predictions/transcriptome"     # Prediction results directory
MODEL_NAME      = "RandomForest"                  # Current best model (can be changed to CatBoost/XGBoost)
SUBGROUP        = "pharma_yes"
OMICS_TYPE      = "transcriptome"
# ========================================================================

def setup_logging():
    """Initialize logging to console and file."""
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    log_file = os.path.join(PREDICTION_DIR, "predict_transcriptome.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info("Logging system initialized")
    logging.info(f"Prediction results will be saved to: {PREDICTION_DIR}")


def load_model_components():
    """Load trained model, scaler, and feature list."""
    logging.info("Loading best model components...")

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
    logging.info(f"Number of features used: {len(features)}")
    return model, scaler, features


def preprocess_test_data(df: pd.DataFrame, required_features: list, scaler):
    """Exactly reproduce the preprocessing pipeline used during training."""
    logging.info("Preprocessing test data...")

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
            f"Test data missing {len(missing)} features used during training!\n"
            f"First 10 missing: {list(missing)[:10]}"
        )

    extra = set(X.columns) - set(required_features)
    if extra:
        logging.warning(f"Test data contains {len(extra)} unused columns (will be ignored)")

    # Keep only training features in correct order
    X = X[required_features]

    # Same preprocessing as training: median imputation → log2(x+1) → RobustScaler
    X = X.fillna(X.median(numeric_only=True))
    X_log = np.log2(X + 1)
    X_scaled = scaler.transform(X_log)

    logging.info(f"Preprocessing complete → Final feature matrix shape: {X_scaled.shape}")
    return X_scaled, y_true, df['submitter_id'] if has_id else None


def predict(model, X_scaled):
    """Generate predictions and probabilities."""
    logging.info("Generating predictions using the best model...")
    y_pred = model.predict(X_scaled).flatten()
    y_proba = model.predict_proba(X_scaled)[:, 1]
    return y_pred.astype(int), y_proba


def main():
    parser = argparse.ArgumentParser(
        description="TCGA-OV Transcriptome-based Drug Response Prediction (pharma_yes subgroup)"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test CSV file (must contain all features used during training)"
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

    # 4. Save results
    result_df = pd.DataFrame({
        'submitter_id': submitter_ids if submitter_ids is not None else [f"sample_{i}" for i in range(len(y_pred))],
        'predicted_label': y_pred,
        'predicted_probability': y_proba.round(6)
    })
    if y_true is not None:
        result_df['true_label'] = y_true

    output_csv = os.path.join(PREDICTION_DIR, "transcriptome_prediction_results.csv")
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"Prediction completed! Results saved to: {output_csv}")

    # 5. Print performance metrics if ground truth is available
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