"""
Outlier detection methods for preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from scipy import stats


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detects outliers in the training data using various methods.

    Parameters
    ----------
    method : str, default='isolation_forest'
        Outlier detection method. Options:
        - 'isolation_forest': Isolation Forest algorithm
        - 'lof': Local Outlier Factor
        - 'elliptic': Elliptic Envelope (assumes Gaussian distribution)
        - 'zscore': Statistical Z-score method
        - 'iqr': Interquartile Range method
        - 'ensemble': Combination of multiple methods
    contamination : float or 'auto', default='auto'
        Expected proportion of outliers in the dataset
    n_neighbors : int, default=20
        Number of neighbors for LOF method
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print information about outliers
    """

    def __init__(self, method='isolation_forest', contamination='auto',
                 n_neighbors=20, random_state=42, verbose=True):
        self.method = method
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.verbose = verbose
        self.outlier_mask_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """Fit the outlier detector on the training data."""
        X_array = X.values if hasattr(X, 'values') else X

        if self.method == 'isolation_forest':
            self._fit_isolation_forest(X_array, y)
        elif self.method == 'lof':
            self._fit_lof(X_array, y)
        elif self.method == 'elliptic':
            self._fit_elliptic(X_array, y)
        elif self.method == 'zscore':
            self._fit_zscore(X_array, y)
        elif self.method == 'iqr':
            self._fit_iqr(X_array, y)
        elif self.method == 'ensemble':
            self._fit_ensemble(X_array, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted_ = True

        if self.verbose:
            n_outliers = np.sum(self.outlier_mask_)
            print(f"\nOutlier Detection ({self.method}):")
            print(f"  Total samples: {len(X)}")
            print(f"  Outliers detected: {n_outliers} ({n_outliers / len(X) * 100:.2f}%)")

            if y is not None:
                # Analyze outliers by class
                y_array = y.values.ravel() if hasattr(y, 'values') else y.ravel()
                for class_val in np.unique(y_array):
                    class_mask = y_array == class_val
                    class_outliers = np.sum(self.outlier_mask_ & class_mask)
                    class_total = np.sum(class_mask)
                    print(f"  Class {class_val}: {class_outliers}/{class_total} " +
                          f"({class_outliers / class_total * 100:.2f}% outliers)")

        return self

    def transform(self, X, y=None):
        """
        Remove outliers from X (and y if provided).
        Only removes outliers during training (when y is provided).
        """
        if not self.is_fitted_:
            raise ValueError("OutlierDetector must be fitted before transform")

        # During prediction (y is None), return all data
        if y is None:
            return X

        # During training, remove outliers
        inlier_mask = ~self.outlier_mask_

        if hasattr(X, 'loc'):
            X_clean = X.loc[inlier_mask].copy()
        else:
            X_clean = X[inlier_mask]

        return X_clean

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)

        if y is None:
            return X

        # Remove outliers from X
        inlier_mask = ~self.outlier_mask_

        if hasattr(X, 'loc'):
            X_clean = X.loc[inlier_mask].copy()
        else:
            X_clean = X[inlier_mask]

        return X_clean

    def _fit_isolation_forest(self, X, y):
        """Fit Isolation Forest."""
        contamination = self.contamination
        if contamination == 'auto':
            contamination = 0.05  # Default 5% contamination

        clf = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        predictions = clf.fit_predict(X)
        self.outlier_mask_ = predictions == -1

    def _fit_lof(self, X, y):
        """Fit Local Outlier Factor."""
        contamination = self.contamination
        if contamination == 'auto':
            contamination = 0.05

        clf = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )
        predictions = clf.fit_predict(X)
        self.outlier_mask_ = predictions == -1

    def _fit_elliptic(self, X, y):
        """Fit Elliptic Envelope."""
        contamination = self.contamination
        if contamination == 'auto':
            contamination = 0.05

        clf = EllipticEnvelope(
            contamination=contamination,
            random_state=self.random_state
        )
        predictions = clf.fit_predict(X)
        self.outlier_mask_ = predictions == -1

    def _fit_zscore(self, X, y, threshold=3):
        """Fit Z-score based outlier detection."""
        z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
        self.outlier_mask_ = np.any(z_scores > threshold, axis=1)

    def _fit_iqr(self, X, y, multiplier=1.5):
        """Fit IQR based outlier detection."""
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outlier_mask = np.zeros(X.shape[0], dtype=bool)
        for i in range(X.shape[1]):
            col_outliers = (X[:, i] < lower_bound[i]) | (X[:, i] > upper_bound[i])
            outlier_mask |= col_outliers

        self.outlier_mask_ = outlier_mask

    def _fit_ensemble(self, X, y):
        """Fit ensemble of multiple outlier detection methods."""
        methods = ['isolation_forest', 'lof', 'zscore']
        votes = np.zeros(X.shape[0])

        for method in methods:
            detector = OutlierDetector(
                method=method,
                contamination=self.contamination,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                verbose=False
            )
            detector.fit(X, y)
            votes += detector.outlier_mask_.astype(int)

        # Consider outlier if majority of methods agree
        self.outlier_mask_ = votes >= len(methods) // 2 + 1

    def get_outlier_indices(self):
        """Return indices of detected outliers."""
        if not self.is_fitted_:
            raise ValueError("OutlierDetector must be fitted first")
        return np.where(self.outlier_mask_)[0]