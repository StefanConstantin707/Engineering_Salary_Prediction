#!/usr/bin/env python3
"""
Train the best performing model with optimized hyperparameters.
This script trains an XGBoost model with ordinal classification,
feature selection, and outlier removal.
"""

import os
import sys
import argparse
import json
import pickle
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.data.data_handler import DataHandler
from src.models.ordinal_classifier import OrdinalClassifier
from src.preprocessing.outlier_detection import OutlierDetector
from src.preprocessing.feature_selection import FeatureSelector
from xgboost import XGBClassifier


def load_best_params():
    """Load best hyperparameters from previous experiments."""
    # These are the best parameters found through extensive optimization
    best_params = {
        'n_job_clusters': 3,
        'outlier_method': 'ensemble',
        'outlier_contamination': 0.05,
        'feature_selection_method': 'tree_importance',
        'feature_selection_ratio': 0.65,
        'xgb_params': {
            'n_estimators': 800,
            'max_depth': 10,
            'learning_rate': 0.005,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'gamma': 0.0,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
    }
    return best_params


def train_best_model(data_dir=None, output_dir=None):
    """Train the best model configuration."""

    print("=" * 70)
    print("TRAINING BEST MODEL FOR ENGINEERING SALARY PREDICTION")
    print("=" * 70)

    # Load parameters
    params = load_best_params()

    # Set directories
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../data/raw')
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '../experiments/results')

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print("\n1. Loading data...")
    dh = DataHandler(
        data_dir=data_dir,
        n_job_clusters=params['n_job_clusters'],
        use_cluster_probabilities=True
    )

    X_raw, y_df, train_idx = dh.get_train_data_raw()
    X_test_raw, test_idx = dh.get_test_data_raw()
    y = y_df["salary_category"]

    print(f"   Training samples: {len(X_raw)}")
    print(f"   Test samples: {len(X_test_raw)}")
    print(f"   Features: {X_raw.shape[1]}")

    # 2. Outlier detection
    print("\n2. Detecting and removing outliers...")
    X_preprocessed = dh.pipeline.fit_transform(X_raw, y)

    outlier_detector = OutlierDetector(
        method=params['outlier_method'],
        contamination=params['outlier_contamination'],
        random_state=42,
        verbose=True
    )

    outlier_detector.fit(X_preprocessed, y)
    inlier_mask = ~outlier_detector.outlier_mask_

    X_clean = X_raw.loc[inlier_mask].copy()
    y_clean = y.loc[inlier_mask].copy()

    print(f"   Outliers removed: {np.sum(~inlier_mask)} ({np.sum(~inlier_mask) / len(X_raw) * 100:.2f}%)")

    # 3. Create final pipeline
    print("\n3. Creating model pipeline...")

    # Recreate data handler for clean data
    dh_clean = DataHandler(
        data_dir=data_dir,
        n_job_clusters=params['n_job_clusters'],
        use_cluster_probabilities=True
    )

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessing", dh_clean.pipeline),
        ("feature_selection", FeatureSelector(
            method=params['feature_selection_method'],
            n_features=params['feature_selection_ratio'],
            random_state=42,
            verbose=True
        )),
        ("classifier", OrdinalClassifier(
            estimator=XGBClassifier(**params['xgb_params']),
            n_jobs=-1
        ))
    ])

    # 4. Cross-validation
    print("\n4. Running cross-validation...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_clean, y_clean, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"   CV Scores: {cv_scores}")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 5. Train final model
    print("\n5. Training final model on all clean data...")
    pipeline.fit(X_clean, y_clean)

    # Get training accuracy
    y_train_pred = pipeline.predict(X_clean)
    train_accuracy = accuracy_score(y_clean, y_train_pred)
    print(f"   Training Accuracy: {train_accuracy:.4f}")

    # 6. Generate predictions
    print("\n6. Generating test predictions...")
    y_test_pred = pipeline.predict(X_test_raw)

    # Convert to labels
    class_labels = ['Low', 'Medium', 'High']
    test_predictions = [class_labels[pred] for pred in y_test_pred]

    # 7. Save results
    print("\n7. Saving results...")

    # Save model
    model_path = os.path.join(output_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"   Model saved to: {model_path}")

    # Save predictions
    submission_df = pd.DataFrame({
        'id': test_idx,
        'salary_category': test_predictions
    })
    submission_path = os.path.join(output_dir, 'submission_best_model.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"   Predictions saved to: {submission_path}")

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'train_accuracy': float(train_accuracy),
        'n_samples_original': len(X_raw),
        'n_samples_clean': len(X_clean),
        'n_outliers': int(np.sum(~inlier_mask)),
        'test_prediction_distribution': pd.Series(test_predictions).value_counts().to_dict()
    }

    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Submission saved to: {submission_path}")

    # Print classification report on training data
    print("\nClassification Report (Training Data):")
    print(classification_report(y_clean, y_train_pred, target_names=class_labels))

    return pipeline


def main():
    parser = argparse.ArgumentParser(description='Train the best model for salary prediction')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, help='Path to output directory')

    args = parser.parse_args()

    train_best_model(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()