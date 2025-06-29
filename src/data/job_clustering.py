"""
Job description clustering transformer.
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class JobDescriptionClusterTransformer(BaseEstimator, TransformerMixin):
    """
    Transform 300 job description columns into cluster-based features.

    This transformer:
    - Handles missing values in job descriptions
    - Scales features before clustering
    - Creates cluster probability features or one-hot encoded assignments
    - Adds cluster distance features
    """

    def __init__(self, n_clusters=5, use_probabilities=True,
                 data_dir=None, random_state=42):
        self.n_clusters = n_clusters
        self.use_probabilities = use_probabilities
        self.random_state = random_state
        self.data_dir = data_dir

        # Components to be fitted
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=30
        )
        self.imputer = SimpleImputer(strategy='mean')
        self.means_ = None

        # Fit on all available data if data_dir is provided
        if data_dir is not None:
            self._fit_on_all_data()

    def _fit_on_all_data(self):
        """Fit clustering on combined train and test data for better clusters."""
        job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]

        # Load train and test data
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        train_df = pd.read_csv(train_path)[job_desc_cols]
        test_df = pd.read_csv(test_path)[job_desc_cols]

        # Convert to arrays and handle zeros
        train_array = train_df.values
        test_array = test_df.values

        train_array = np.where(train_array == 0, np.nan, train_array)
        test_array = np.where(test_array == 0, np.nan, test_array)

        # Combine and remove rows with all NaN
        combined_array = np.concatenate((train_array, test_array), axis=0)
        combined_array = combined_array[~np.isnan(combined_array[:, 0])]

        # Fit imputer and scaler
        combined_imputed = self.imputer.fit_transform(combined_array)
        scaled_array = self.scaler.fit_transform(combined_imputed)

        # Fit clustering and compute mean probabilities
        distances = self.kmeans.fit_transform(scaled_array)

        # Convert distances to probabilities
        neg_distances = -distances
        exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

        # Store mean probabilities for missing data imputation
        prob_with_dist = np.concatenate(
            (probabilities, np.min(distances, axis=1).reshape(-1, 1)),
            axis=1
        )
        self.means_ = np.mean(prob_with_dist, axis=0)

    def fit(self, X, y=None):
        """Fit the clustering model if not already fitted."""
        if self.means_ is not None:
            # Already fitted on all data
            return self

        # Standard fitting on provided data
        X_array = X.values if hasattr(X, 'values') else X
        X_array = np.where(X_array == 0, np.nan, X_array)

        # Remove rows with all NaN
        valid_mask = ~np.isnan(X_array[:, 0])
        X_valid = X_array[valid_mask]

        if len(X_valid) == 0:
            raise ValueError("No valid samples for clustering")

        # Fit preprocessing
        X_imputed = self.imputer.fit_transform(X_valid)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Fit clustering
        self.kmeans.fit(X_scaled)

        # Compute default means for missing data
        distances = self.kmeans.transform(X_scaled)
        neg_distances = -distances
        exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

        prob_with_dist = np.concatenate(
            (probabilities, np.min(distances, axis=1).reshape(-1, 1)),
            axis=1
        )
        self.means_ = np.mean(prob_with_dist, axis=0)

        return self

    def transform(self, X):
        """Transform job descriptions into cluster features."""
        X_array = X.values if hasattr(X, 'values') else X
        X_array = np.where(X_array == 0, np.nan, X_array)

        # Separate valid and invalid rows
        nan_mask = np.isnan(X_array[:, 0])
        X_valid = X_array[~nan_mask]

        # Get index if available
        index = X.index if hasattr(X, 'index') else None

        if self.use_probabilities:
            # Initialize output array
            n_features = self.n_clusters + 1  # probabilities + min distance
            output = np.empty((X_array.shape[0], n_features))

            # Fill invalid rows with means
            output[nan_mask] = self.means_

            # Process valid rows if any
            if len(X_valid) > 0:
                X_imputed = self.imputer.transform(X_valid)
                X_scaled = self.scaler.transform(X_imputed)

                # Calculate distances and probabilities
                distances = self.kmeans.transform(X_scaled)
                neg_distances = -distances
                exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
                probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

                # Combine probabilities and min distance
                prob_with_dist = np.concatenate(
                    (probabilities, np.min(distances, axis=1, keepdims=True)),
                    axis=1
                )
                output[~nan_mask] = prob_with_dist

            # Create feature names
            feature_names = [f'job_cluster_prob_{i}' for i in range(self.n_clusters)]
            feature_names.append('job_cluster_min_dist')

        else:
            # One-hot encoding
            output = np.zeros((X_array.shape[0], self.n_clusters))

            # Process valid rows
            if len(X_valid) > 0:
                X_imputed = self.imputer.transform(X_valid)
                X_scaled = self.scaler.transform(X_imputed)
                cluster_labels = self.kmeans.predict(X_scaled)

                # Create one-hot encoding
                valid_output = np.zeros((len(X_valid), self.n_clusters))
                valid_output[np.arange(len(cluster_labels)), cluster_labels] = 1
                output[~nan_mask] = valid_output

            # For invalid rows, assign to cluster 0 by default
            output[nan_mask, 0] = 1

            feature_names = [f'job_cluster_{i}' for i in range(self.n_clusters)]

        return pd.DataFrame(output, columns=feature_names, index=index)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        if self.use_probabilities:
            names = [f'job_cluster_prob_{i}' for i in range(self.n_clusters)]
            names.append('job_cluster_min_dist')
        else:
            names = [f'job_cluster_{i}' for i in range(self.n_clusters)]
        return np.array(names)

    def analyze_clusters(self, X, y):
        """Analyze cluster composition with respect to target variable."""
        if not hasattr(self.kmeans, 'cluster_centers_'):
            print("Model must be fitted before analysis")
            return

        # Get job description columns
        job_desc_cols = [f"job_desc_{i:03d}" for i in range(1, 301)]
        X_job_desc = X[job_desc_cols] if hasattr(X, '__getitem__') else X

        # Preprocess
        X_array = X_job_desc.values if hasattr(X_job_desc, 'values') else X_job_desc
        X_array = np.where(X_array == 0, np.nan, X_array)

        # Get valid rows
        valid_mask = ~np.isnan(X_array[:, 0])
        X_valid = X_array[valid_mask]

        if len(X_valid) == 0:
            print("No valid data for cluster analysis")
            return

        # Transform and predict clusters
        X_imputed = self.imputer.transform(X_valid)
        X_scaled = self.scaler.transform(X_imputed)
        cluster_labels = self.kmeans.predict(X_scaled)

        # Analyze salary distribution per cluster
        print(f"\nCluster Analysis ({self.n_clusters} clusters):")
        print("-" * 50)

        salary_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        y_values = y.values.ravel() if hasattr(y, 'values') else y.ravel()
        y_valid = y_values[valid_mask]

        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_size = np.sum(mask)

            if cluster_size == 0:
                print(f"\nCluster {cluster_id}: Empty")
                continue

            print(f"\nCluster {cluster_id} (n={cluster_size}):")
            cluster_salaries = y_valid[mask]

            for salary_val, salary_name in salary_map.items():
                count = np.sum(cluster_salaries == salary_val)
                pct = 100 * count / cluster_size
                print(f"  {salary_name}: {count} ({pct:.1f}%)")

        # Print summary statistics
        print("\n" + "-" * 50)
        print("Cluster Summary:")

        # Calculate average distance to cluster centers
        distances = self.kmeans.transform(X_scaled)
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                avg_dist = np.mean(distances[cluster_mask, cluster_id])
                print(f"Cluster {cluster_id}: Avg distance = {avg_dist:.3f}")