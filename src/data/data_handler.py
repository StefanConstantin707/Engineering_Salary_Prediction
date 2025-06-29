"""
Main data handler for engineering salary prediction.
Manages data loading, preprocessing pipeline creation, and data access.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector

from .feature_extractor import FeatureExtractor
from .transformers import CustomJobStateFeature1Transformer, DateFeaturesTransformer
from .job_clustering import JobDescriptionClusterTransformer


class DataHandler:
    """
    Main data handler class that manages data loading and preprocessing pipeline.

    Args:
        data_dir (str): Path to data directory
        n_job_clusters (int): Number of clusters for job descriptions (0 to disable)
        use_cluster_probabilities (bool): Use soft clustering (probabilities) vs hard clustering
    """

    def __init__(self, data_dir=None, n_job_clusters=0, use_cluster_probabilities=True):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "../../data/raw")
        self.n_job_clusters = n_job_clusters
        self.use_cluster_probabilities = use_cluster_probabilities

        # Load data
        self._load_data()

        # Build preprocessing pipeline
        if self.n_job_clusters == 0:
            self.pipeline = self._create_pipeline()
        else:
            self.pipeline = self._create_cluster_pipeline()

    def _load_data(self):
        """Load train and test data from CSV files."""
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        # Capture original indices
        self.train_index = self.train_df.index.copy()
        self.test_index = self.test_df.index.copy() + len(self.train_df)

        # Prepare target variable
        mapping = {"Low": 0, "Medium": 1, "High": 2}
        self.train_df["salary_category"] = self.train_df["salary_category"].map(mapping)
        self.y_cols = ["salary_category"]

        # Split features and target
        self.X_train_raw = self.train_df.drop(columns=["salary_category"])
        self.y_train = self.train_df[self.y_cols].copy()
        self.X_test_raw = self.test_df.copy()

    def _create_pipeline(self):
        """Create preprocessing pipeline without clustering."""
        fe = FeatureExtractor()

        preprocessor = ColumnTransformer(
            transformers=[
                ("js_f1", Pipeline([("custom", CustomJobStateFeature1Transformer())]),
                 ["job_state", "feature_1"]),
                ("title", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                 ["job_title"]),
                ("date", DateFeaturesTransformer(), ["job_posted_date"]),
                ("bools", "passthrough", fe.bool_columns),
                ("quants", "passthrough", fe.quantitative_columns),
                ("jobdesc", "passthrough", fe.job_desc_cols),
            ],
            remainder="drop"
        ).set_output(transform="pandas")

        imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")

        # Define columns to scale
        scale_cols = (
                ["js_f1__job_state", "date__months_since_first"] +
                [f"quants__{col}" for col in fe.quantitative_columns] +
                [f"jobdesc__{col}" for col in fe.job_desc_cols]
        )

        scaler = ColumnTransformer(
            transformers=[("scale", StandardScaler(), scale_cols)],
            remainder="passthrough"
        ).set_output(transform="pandas")

        return Pipeline([
            ("feature_extractor", fe),
            ("preprocessor", preprocessor),
            ("imputer", imputer),
            ("scaler", scaler)
        ])

    def _create_cluster_pipeline(self):
        """Create preprocessing pipeline with job description clustering."""
        fe = FeatureExtractor()

        preprocessor = ColumnTransformer(
            transformers=[
                ("js_f1", Pipeline([("custom", CustomJobStateFeature1Transformer())]),
                 ["job_state", "feature_1"]),
                ("title", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                 ["job_title"]),
                ("date", DateFeaturesTransformer(), ["job_posted_date"]),
                ("bools", "passthrough", fe.bool_columns),
                ("quants", "passthrough", fe.quantitative_columns),
                ("jobdesc_cluster", JobDescriptionClusterTransformer(
                    n_clusters=self.n_job_clusters,
                    use_probabilities=self.use_cluster_probabilities,
                    data_dir=self.data_dir,
                    random_state=42
                ), fe.job_desc_cols),
            ],
            remainder="drop"
        ).set_output(transform="pandas")

        imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")

        # Dynamic scaler selector for varying number of cluster features
        scaler_selector = make_column_selector(
            pattern=(r"js_f1__job_state|date__|quants__|jobdesc_cluster__"),
            dtype_include=np.number
        )

        scaler = ColumnTransformer(
            transformers=[("scale", StandardScaler(), scaler_selector)],
            remainder="passthrough"
        ).set_output(transform="pandas")

        return Pipeline([
            ("feature_extractor", fe),
            ("preprocessor", preprocessor),
            ("imputer", imputer),
            ("scaler", scaler)
        ])

    def get_train_data_raw(self):
        """Get raw training data before preprocessing."""
        return self.X_train_raw.copy(), self.y_train.copy(), self.train_index

    def get_test_data_raw(self):
        """Get raw test data before preprocessing."""
        return self.X_test_raw.copy(), self.test_index

    def get_train_data_processed(self):
        """Get preprocessed training data."""
        X_processed = self.pipeline.fit_transform(self.X_train_raw, self.y_train)
        return X_processed, self.y_train.copy()

    def analyze_clusters(self):
        """Analyze job description clusters if clustering is enabled."""
        if self.n_job_clusters == 0:
            print("Clustering is not enabled.")
            return

        preprocessor = self.pipeline.named_steps['preprocessor']
        cluster_transformer = None

        for name, transformer, _ in preprocessor.transformers_:
            if name == 'jobdesc_cluster':
                cluster_transformer = transformer
                break

        if cluster_transformer is None or not hasattr(cluster_transformer, 'analyze_clusters'):
            print("Cluster analysis not available.")
            return

        # Perform cluster analysis
        cluster_transformer.analyze_clusters(self.X_train_raw, self.y_train)