"""
Feature extraction components for data preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract and prepare features from raw data.

    This transformer:
    - Selects relevant columns
    - Replaces zeros with NaN in job description columns
    - Ensures proper data types
    """

    def __init__(self):
        self.date_columns = ["job_posted_date"]
        self.categorical_columns = ["job_title", "feature_1", "job_state"]
        self.bool_columns = [f"feature_{i}" for i in range(3, 10)] + ["feature_11", "feature_12"]
        self.quantitative_columns = ["feature_2", "feature_10"]
        self.job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]

        self.all_features = (
                self.date_columns +
                self.categorical_columns +
                self.bool_columns +
                self.quantitative_columns +
                self.job_desc_cols
        )

    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self

    def transform(self, X):
        """
        Transform the input data.

        Args:
            X: pandas DataFrame with raw features

        Returns:
            pandas DataFrame with selected and cleaned features
        """
        # Validate columns
        missing = [col for col in self.all_features if col not in X.columns]
        if missing:
            raise ValueError(f"Missing columns in input DataFrame: {missing}")

        # Select features
        df = X[self.all_features].copy()

        # Replace zeros with NaN in job description columns
        for col in self.job_desc_cols:
            df[col] = df[col].replace(0, np.nan)

        # Ensure job_posted_date is datetime
        if not np.issubdtype(df["job_posted_date"].dtype, np.datetime64):
            df["job_posted_date"] = pd.to_datetime(
                df["job_posted_date"],
                format="%Y/%m",
                errors="coerce"
            )

        return df

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.all_features

    def set_output(self, transform=None):
        """Set output format (for sklearn compatibility)."""
        return self