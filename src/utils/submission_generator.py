"""
Utility for generating competition submission files.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


class SubmissionGenerator:
    """
    Generate and validate competition submission files.

    Parameters
    ----------
    output_dir : str
        Directory to save submission files
    competition_name : str
        Name of the competition
    """

    def __init__(self, output_dir="experiments/results/submissions",
                 competition_name="engineering_salary"):
        self.output_dir = output_dir
        self.competition_name = competition_name
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, predictions, test_ids, model_name="model",
                 score=None, save=True):
        """
        Generate submission file from predictions.

        Parameters
        ----------
        predictions : array-like
            Predicted labels (can be numeric or string)
        test_ids : array-like
            Test sample IDs
        model_name : str
            Name of the model for filename
        score : float, optional
            CV score to include in filename
        save : bool
            Whether to save the file

        Returns
        -------
        submission_df : pd.DataFrame
            Submission dataframe
        filename : str
            Generated filename
        """
        # Validate inputs
        if len(predictions) != len(test_ids):
            raise ValueError(
                f"Length mismatch: predictions ({len(predictions)}) "
                f"!= test_ids ({len(test_ids)})"
            )

        # Convert numeric predictions to labels if needed
        if isinstance(predictions[0], (int, np.integer)):
            label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            predictions = [label_map[p] for p in predictions]

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': test_ids,
            'salary_category': predictions
        })

        # Validate predictions
        valid_categories = {'Low', 'Medium', 'High'}
        invalid_preds = set(predictions) - valid_categories
        if invalid_preds:
            raise ValueError(f"Invalid predictions found: {invalid_preds}")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if score is not None:
            filename = f"{self.competition_name}_{model_name}_cv{score:.4f}_{timestamp}.csv"
        else:
            filename = f"{self.competition_name}_{model_name}_{timestamp}.csv"

        # Save if requested
        if save:
            filepath = os.path.join(self.output_dir, filename)
            submission_df.to_csv(filepath, index=False)
            print(f"Submission saved to: {filepath}")

            # Print summary
            self._print_summary(submission_df)

        return submission_df, filename

    def _print_summary(self, submission_df):
        """Print submission summary statistics."""
        print("\nSubmission Summary:")
        print("-" * 40)
        print(f"Total predictions: {len(submission_df)}")
        print("\nClass distribution:")

        value_counts = submission_df['salary_category'].value_counts()
        for category in ['Low', 'Medium', 'High']:
            count = value_counts.get(category, 0)
            percentage = count / len(submission_df) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

    def validate_submission(self, submission_path, reference_path=None):
        """
        Validate a submission file format.

        Parameters
        ----------
        submission_path : str
            Path to submission file to validate
        reference_path : str, optional
            Path to reference test file for ID validation

        Returns
        -------
        is_valid : bool
            Whether the submission is valid
        errors : list
            List of validation errors
        """
        errors = []

        try:
            # Load submission
            submission_df = pd.read_csv(submission_path)

            # Check columns
            required_columns = {'id', 'salary_category'}
            missing_columns = required_columns - set(submission_df.columns)
            if missing_columns:
                errors.append(f"Missing columns: {missing_columns}")

            # Check for null values
            if submission_df.isnull().any().any():
                null_counts = submission_df.isnull().sum()
                errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

            # Check categories
            if 'salary_category' in submission_df.columns:
                valid_categories = {'Low', 'Medium', 'High'}
                unique_categories = set(submission_df['salary_category'].unique())
                invalid_categories = unique_categories - valid_categories
                if invalid_categories:
                    errors.append(f"Invalid categories: {invalid_categories}")

            # Check IDs if reference provided
            if reference_path and os.path.exists(reference_path):
                reference_df = pd.read_csv(reference_path)
                if 'id' in reference_df.columns:
                    expected_ids = set(reference_df['id'])
                elif reference_df.index.name == 'id':
                    expected_ids = set(reference_df.index)
                else:
                    # Assume sequential IDs starting from length of training data
                    expected_ids = set(range(1281, 1281 + len(reference_df)))

                submission_ids = set(submission_df['id'])
                missing_ids = expected_ids - submission_ids
                extra_ids = submission_ids - expected_ids

                if missing_ids:
                    errors.append(f"Missing IDs: {len(missing_ids)} IDs")
                if extra_ids:
                    errors.append(f"Extra IDs: {len(extra_ids)} IDs")

        except Exception as e:
            errors.append(f"Error reading file: {str(e)}")

        is_valid = len(errors) == 0

        if is_valid:
            print("✓ Submission is valid!")
        else:
            print("✗ Submission validation failed:")
            for error in errors:
                print(f"  - {error}")

        return is_valid, errors

    def compare_submissions(self, submission_paths, names=None):
        """
        Compare multiple submission files.

        Parameters
        ----------
        submission_paths : list
            List of paths to submission files
        names : list, optional
            Names for each submission

        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame showing prediction differences
        """
        if names is None:
            names = [f"Submission_{i + 1}" for i in range(len(submission_paths))]

        # Load all submissions
        submissions = []
        for path in submission_paths:
            df = pd.read_csv(path)
            submissions.append(df)

        # Create comparison DataFrame
        comparison_df = submissions[0][['id']].copy()

        for name, df in zip(names, submissions):
            comparison_df[name] = df['salary_category']

        # Add agreement column
        comparison_df['all_agree'] = comparison_df[names].apply(
            lambda row: len(set(row)) == 1, axis=1
        )

        # Calculate statistics
        n_samples = len(comparison_df)
        n_agree = comparison_df['all_agree'].sum()
        agreement_rate = n_agree / n_samples * 100

        print(f"\nSubmission Comparison:")
        print(f"Total samples: {n_samples}")
        print(f"Full agreement: {n_agree} ({agreement_rate:.1f}%)")

        # Show disagreement examples
        disagreements = comparison_df[~comparison_df['all_agree']]
        if len(disagreements) > 0:
            print(f"\nExample disagreements (showing first 10):")
            print(disagreements.head(10))

        # Pairwise agreement
        if len(names) > 2:
            print("\nPairwise agreement rates:")
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    agree = (comparison_df[names[i]] == comparison_df[names[j]]).sum()
                    rate = agree / n_samples * 100
                    print(f"  {names[i]} vs {names[j]}: {rate:.1f}%")

        return comparison_df