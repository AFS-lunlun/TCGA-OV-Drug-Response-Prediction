#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-OV Multi-omics Early Fusion Best Model Independent Prediction Script
Subgroup: pharma_yes (patients who received drug treatment)
Fusion type: Early Fusion (Protein + Methylation + Transcriptome concatenated)
Best model: RandomForest (auto-detected from files)
Model directory:
    ./models/early_fusion/
Prediction output directory:
    ./predictions/early_fusion/
Usage:
    python predict_early_fusion.py --test your_early_fusion_test.csv
Input file requirements (must be strictly followed):
    1. CSV file
    2. Must contain ALL feature columns used during training (exact column names)
       - protein_ prefixed protein features
       - methyl_ prefixed methylation features (gene symbols extracted)
       - transcript_ prefixed transcriptome features
    3. Optional: submitter_id column (for result alignment)
    4. Optional: label column (for offline performance evaluation)
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
# Adjust these paths according to your project structure
MODEL_DIR = "./model/early_fusion"
PREDICTION_DIR = "./predictions/early_fusion"

MODEL_NAME = "RandomForest"          # Current best model
SUBGROUP = "pharma_yes"
FUSION_TYPE = "early_fusion"
# =========================================================================

def setup_logging() -> None:
    """Create prediction directory and configure logging (console + file)."""
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    log_file = os.path.join(PREDICTION_DIR, "predict_early_fusion.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info("=== Early Fusion Independent Prediction Started ===")
    logging.info(f"Model directory: {MODEL_DIR}")
    logging.info(f"Prediction output directory: {PREDICTION_DIR}")


def load_model_components():
    """Load the best model, scaler, and feature list."""
    logging.info("Loading Early Fusion best model components...")
    model_path   = os.path.join(MODEL_DIR, f"best_model_{MODEL_NAME}_{SUBGROUP}_{FUSION_TYPE}.pkl")
    scaler_path  = os.path.join(MODEL_DIR, f"best_scaler_{MODEL_NAME}_{SUBGROUP}_{FUSION_TYPE}.pkl")
    features_path = os.path.join(MODEL_DIR, f"best_features_{MODEL_NAME}_{SUBGROUP}_{FUSION_TYPE}.csv")

    for path, name in [
        (model_path,   "model file"),
        (scaler_path,  "scaler file"),
        (features_path, "feature list file")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Cannot find {name}\n → {path}")

    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    features = pd.read_csv(features_path)['feature_name'].tolist()

    logging.info(f"Best model loaded successfully: {MODEL_NAME}")
    logging.info(f"Number of fused features: {len(features)} (protein + methylation + transcriptome)")
    return model, scaler, features


def preprocess_test_data(df: pd.DataFrame, required_features: list, scaler):
    """Exactly reproduce the preprocessing pipeline used during training."""
    logging.info("Preprocessing Early Fusion test data...")

    has_label = 'label' in df.columns
    has_id    = 'submitter_id' in df.columns

    if has_label:
        y_true = df['label'].astype(int).values
        X = df.drop(columns=['label'] + (['submitter_id'] if has_id else []))
        logging.info(f"True labels detected, {len(y_true)} samples")
    else:
        y_true = None
        X = df.drop(columns=['submitter_id'] if has_id else [])
        logging.info(f"No true labels, performing prediction only on {len(X)} samples")

    # Check feature completeness
    missing = set(required_features) - set(X.columns)
    if missing:
        raise KeyError(
            f"Test data missing {len(missing)} features used in training!\n"
            f"First 10 missing features: {list(missing)[:10]}"
        )

    extra = set(X.columns) - set(required_features)
    if extra:
        logging.warning(
            f"Test data contains {len(extra)} extra columns (ignored). "
            f"First 10: {list(extra)[:10]}"
        )

    # Keep only training features in exact order
    X = X[required_features]

    # Impute missing values with median (same as training)
    X = X.fillna(X.median(numeric_only=True))

    # log2(x+1) transformation for methylation and transcriptome features
    for col in X.columns:
        if col.startswith('methyl_') or col.startswith('transcript_'):
            X[col] = np.log2(X[col] + 1)

    # Apply the same RobustScaler fitted on training data
    X_scaled = scaler.transform(X)

    logging.info(f"Preprocessing finished → Final feature matrix shape: {X_scaled.shape}")
    return X_scaled, y_true, df['submitter_id'] if has_id else None


def predict(model, X_scaled):
    """Perform prediction with RandomForest."""
    logging.info("Predicting with RandomForest (Early Fusion)...")
    y_pred  = model.predict(X_scaled).astype(int)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    return y_pred, y_proba


def main():
    parser = argparse.ArgumentParser(
        description="TCGA-OV Multi-omics Early Fusion Drug Response Prediction"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test CSV file (must contain all fused feature columns)"
    )
    args = parser.parse_args()

    setup_logging()

    # 1. Load model components
    model, scaler, features = load_model_components()

    # 2. Read test data
    if not os.path.exists(args.test):
        raise FileNotFoundError(f"Test file not found: {args.test}")

    test_df = pd.read_csv(args.test)
    logging.info(f"Test data loaded successfully: {args.test}, samples: {len(test_df)}")

    # 3. Preprocess & predict
    X_scaled, y_true, submitter_ids = preprocess_test_data(test_df, features, scaler)
    y_pred, y_proba = predict(model, X_scaled)

    # 4. Assemble results
    result_df = pd.DataFrame({
        'submitter_id': submitter_ids if submitter_ids is not None
                        else [f"sample_{i}" for i in range(len(y_pred))],
        'predicted_label': y_pred,
        'predicted_probability': y_proba.round(6)
    })
    if y_true is not None:
        result_df['true_label'] = y_true

    # 5. Save results
    output_csv = os.path.join(PREDICTION_DIR, "early_fusion_prediction_results.csv")
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"Prediction completed! Results saved to:\n{output_csv}")

    # 6. Performance metrics if ground truth is available
    if y_true is not None:
        logging.info("\n=== Test Set Performance Metrics ===")
        logging.info(f"Accuracy   : {accuracy_score(y_true, y_pred):.4f}")
        logging.info(f"Precision  : {precision_score(y_true, y_pred):.4f}")
        logging.info(f"Recall     : {recall_score(y_true, y_pred):.4f}")
        logging.info(f"F1 Score   : {f1_score(y_true, y_pred):.4f}")
        logging.info(f"AUC        : {roc_auc_score(y_true, y_proba):.4f}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == "__main__":
    main()