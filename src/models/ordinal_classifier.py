"""
Ordinal Classification implementation based on Frank & Hall's approach.

This module implements ordinal classification by transforming a k-class ordinal
problem into k-1 binary classification problems, leveraging the natural ordering
of classes (e.g., Low < Medium < High).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.base import MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.multiclass import type_of_target
from typing import Iterable


def _fit_binary(estimator, X, y, classes):
    """Fit a single binary classifier."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if unique_y[0] == -1:
            c = 0
        else:
            c = len(classes) - 1
        estimator.classes_ = classes
        return estimator

    # Fit the binary classifier
    estimator.fit(X, y)
    return estimator


class OrdinalClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """
    Ordinal multiclass classifier.

    Based on "A Simple Approach to Ordinal Classification" by Frank and Hall.
    This classifier transforms an ordinal classification problem with k classes
    into k-1 binary classification problems.

    For classes V1, V2, ..., Vk, it creates k-1 binary problems:
    - Problem 1: Is class > V1? (i.e., is it V2, V3, ..., or Vk?)
    - Problem 2: Is class > V2? (i.e., is it V3, ..., or Vk?)
    - ...
    - Problem k-1: Is class > Vk-1? (i.e., is it Vk?)

    Parameters
    ----------
    estimator : estimator object
        An estimator implementing fit and predict_proba.
    n_jobs : int, default=None
        Number of jobs for parallel computation.
    reverse_classes : bool, default=False
        Whether to reverse the class order.

    Attributes
    ----------
    estimators_ : list of estimators
        The collection of fitted binary estimators.
    classes_ : array-like of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        Number of classes.
    """

    def __init__(self, estimator, n_jobs=None, reverse_classes=False):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.reverse_classes = reverse_classes

        # Validate that estimator has predict_proba
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                f"Estimator {self.estimator.__class__.__name__} "
                "does not have predict_proba method which is required "
                "for ordinal classification."
            )

    def fit(self, X, y):
        """
        Fit the ordinal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (ordinal classes).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Determine class order
        self.classes_ = np.sort(np.unique(y))

        if self.reverse_classes:
            self.classes_ = self.classes_[::-1]

        self.n_classes_ = len(self.classes_)

        # Check target type
        self.y_type_ = type_of_target(y)
        if self.y_type_ not in ["multiclass", "binary"]:
            raise ValueError(
                f"This classifier expects multiclass or binary targets. "
                f"Got type: {self.y_type_}"
            )

        if self.n_classes_ == 2:
            # Binary case - fit single estimator
            self.estimators_ = [clone(self.estimator)]
            self.estimators_[0].fit(X, y)
        else:
            # Multiclass case - create k-1 binary problems
            y_derived = self._create_binary_problems(y)

            # Fit k-1 binary classifiers in parallel
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_binary)(
                    clone(self.estimator),
                    X,
                    y_d,
                    classes=[0, 1]  # Binary labels
                )
                for y_d in y_derived.T
            )

        # Set feature names if available
        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        """
        Predict ordinal class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)

        if self.n_classes_ == 2:
            # Binary case
            return self.estimators_[0].predict(X)
        else:
            # Multiclass case - use probabilities
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        The class probabilities are derived from the binary classifiers:
        - P(y = V1) = 1 - P(y > V1)
        - P(y = Vi) = P(y > Vi-1) - P(y > Vi) for 1 < i < k
        - P(y = Vk) = P(y > Vk-1)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)

        if self.n_classes_ == 2:
            # Binary case
            proba = self.estimators_[0].predict_proba(X)
            return proba

        # Get probabilities for each binary problem
        # Y[i, j] = P(y > Vj) for sample i
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

        # Convert to class probabilities
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))

        # P(y = V1) = 1 - P(y > V1)
        probas[:, 0] = 1 - Y[:, 0]

        # P(y = Vi) = P(y > Vi-1) - P(y > Vi) for middle classes
        for i in range(1, self.n_classes_ - 1):
            probas[:, i] = Y[:, i - 1] - Y[:, i]

        # P(y = Vk) = P(y > Vk-1)
        probas[:, -1] = Y[:, -1]

        # Ensure probabilities are non-negative and sum to 1
        probas = np.maximum(probas, 0)
        probas /= probas.sum(axis=1)[:, np.newaxis]

        return probas

    def _create_binary_problems(self, y):
        """
        Create k-1 binary target arrays for ordinal classification.

        For each position i (0 to k-2):
        - y_binary[i] = 1 if original class > classes_[i], else 0

        Parameters
        ----------
        y : array-like
            Original ordinal targets.

        Returns
        -------
        y_binary : array of shape (n_samples, n_classes - 1)
            Binary targets for each threshold.
        """
        n_samples = len(y)
        n_thresholds = self.n_classes_ - 1
        y_binary = np.zeros((n_samples, n_thresholds))

        for i in range(n_thresholds):
            threshold = self.classes_[i]
            # Create binary target: 1 if y > threshold, 0 otherwise
            y_binary[:, i] = (y > threshold).astype(int)

        return y_binary

    @property
    def feature_importances_(self):
        """
        Feature importances from the first binary classifier.

        Only available if the base estimator has feature_importances_.
        """
        check_is_fitted(self)
        if hasattr(self.estimators_[0], "feature_importances_"):
            return self.estimators_[0].feature_importances_
        else:
            raise AttributeError(
                f"Base estimator {self.estimator.__class__.__name__} "
                "does not have feature_importances_ attribute."
            )