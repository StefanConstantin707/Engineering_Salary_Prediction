#!/usr/bin/env python3
"""
Generate competition submission from a trained model.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.data.data_handler import DataHandler
from src.utils.submission_generator import SubmissionGenerator
from src.config.paths import (
    get_latest_model, MODELS_DIR, SUBMISSIONS_DIR,
    RAW_DATA_DIR, TEST_DATA_PATH
)


def load_model(model_path):
    """Load a saved model from pickle file."""
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def generate_submission(model_path=None, output_dir=None, cv_score=None):
    """
    Generate submission file from a trained model.

    Parameters
    ----------
    model_path : str, optional
        Path to saved model. If None, uses latest model.
    output_dir : str, optional
        Directory to save submission. If None, uses default.
    cv_score : float, optional
        CV score to include in filename.
    """
    print("=" * 70)
    print("GENERATING COMPETITION SUBMISSION")
    print("=" * 70)

    # Set paths
    if model_path is None:
        model_path = get_latest_model()
        if model_path is None:
            raise ValueError("No model found. Please train a model first.")

    if output_dir is None:
        output_dir = SUBMISSIONS_DIR

    # Load model
    model = load_model(model_path)
    model_name = Path(model_path).stem

    # Load test data
    print(f"\nLoading test data from: {TEST_DATA_PATH}")

    # Check if we need to load raw data or if the model handles it
    if hasattr(model, 'named_steps') and 'preprocessing' in model.named_steps:
        # Pipeline includes preprocessing
        test_df = pd.read_csv(TEST_DATA_PATH)
        test_ids = test_df.index + 1281  # Adjust based on training data size
        X_test = test_df
    else:
        # Need to preprocess data
        data_handler = DataHandler(data_dir=RAW_DATA_DIR)
        X_test, test_ids = data_handler.get_test_data_raw()

    print(f"Test data shape: {X_test.shape}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {pd.Series(predictions).value_counts().sort_index().to_dict()}")

    # Create submission
    submission_gen = SubmissionGenerator(output_dir=output_dir)

    # Extract model type from pipeline if possible
    if hasattr(model, 'named_steps'):
        if 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'estimator'):
                model_type = classifier.estimator.__class__.__name__
            else:
                model_type = classifier.__class__.__name__
        else:
            model_type = "pipeline"
    else:
        model_type = model.__class__.__name__

    submission_df, filename = submission_gen.generate(
        predictions=predictions,
        test_ids=test_ids,
        model_name=model_type,
        score=cv_score,
        save=True
    )

    # Validate submission
    print("\nValidating submission...")
    is_valid, errors = submission_gen.validate_submission(
        os.path.join(output_dir, filename),
        reference_path=TEST_DATA_PATH
    )

    if not is_valid:
        print("WARNING: Submission validation failed!")
        for error in errors:
            print(f"  - {error}")

    print("\n" + "=" * 70)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 70)

    return submission_df, filename


def main():
    parser = argparse.ArgumentParser(
        description='Generate competition submission from trained model'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to saved model (uses latest if not specified)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save submission'
    )
    parser.add_argument(
        '--cv-score',
        type=float,
        help='CV score to include in filename'
    )
    parser.add_argument(
        '--compare-with',
        type=str,
        nargs='+',
        help='Compare with other submission files'
    )

    args = parser.parse_args()

    # Generate submission
    submission_df, filename = generate_submission(
        model_path=args.model_path,
        output_dir=args.output_dir,
        cv_score=args.cv_score
    )

    # Compare with other submissions if requested
    if args.compare_with:
        print("\n" + "=" * 70)
        print("COMPARING SUBMISSIONS")
        print("=" * 70)

        submission_gen = SubmissionGenerator()
        submission_paths = [os.path.join(args.output_dir or SUBMISSIONS_DIR, filename)]
        submission_paths.extend(args.compare_with)

        names = [f"New_{filename.split('_')[1]}"]
        for path in args.compare_with:
            names.append(Path(path).stem.split('_')[1])

        comparison_df = submission_gen.compare_submissions(
            submission_paths,
            names=names
        )


if __name__ == '__main__':
    main()