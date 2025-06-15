import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, RFECV, VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature selection transformer that supports multiple methods.
    """

    def __init__(self, method='ensemble', n_features='auto', threshold=0.5,
                 random_state=42, verbose=True):
        """
        Parameters:
        -----------
        method : str, default='ensemble'
            Feature selection method. Options:
            - 'variance': Remove low-variance features
            - 'correlation': Remove highly correlated features
            - 'chi2': Chi-squared test (for non-negative features)
            - 'anova': ANOVA F-test
            - 'mutual_info': Mutual information
            - 'tree_importance': Tree-based feature importance
            - 'lasso': L1-based selection
            - 'rfe': Recursive Feature Elimination
            - 'ensemble': Combination of multiple methods
        n_features : int, float, or 'auto', default='auto'
            Number of features to select:
            - If int: select exactly n_features
            - If float (0-1): select this proportion of features
            - If 'auto': automatically determine based on method
        threshold : float, default=0.5
            Threshold for some methods (e.g., correlation threshold)
        random_state : int, default=42
            Random state for reproducibility
        verbose : bool, default=True
            Whether to print information about selection
        """
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose
        self.selected_features_ = None
        self.feature_names_ = None
        self.selector_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """Fit the feature selector."""
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]

        # Determine number of features to select
        n_features_total = X.shape[1]
        if self.n_features == 'auto':
            # Auto selection based on method
            if self.method in ['variance', 'correlation']:
                n_features_to_select = None  # These methods use thresholds
            else:
                # Select 50% of features by default
                n_features_to_select = max(10, n_features_total // 2)
        elif isinstance(self.n_features, float) and 0 < self.n_features < 1:
            n_features_to_select = max(1, int(n_features_total * self.n_features))
        else:
            n_features_to_select = min(int(self.n_features), n_features_total)

        # Apply the selected method
        if self.method == 'variance':
            self._fit_variance(X, y)
        elif self.method == 'correlation':
            self._fit_correlation(X, y)
        elif self.method == 'chi2':
            self._fit_chi2(X, y, n_features_to_select)
        elif self.method == 'anova':
            self._fit_anova(X, y, n_features_to_select)
        elif self.method == 'mutual_info':
            self._fit_mutual_info(X, y, n_features_to_select)
        elif self.method == 'tree_importance':
            self._fit_tree_importance(X, y, n_features_to_select)
        elif self.method == 'lasso':
            self._fit_lasso(X, y, n_features_to_select)
        elif self.method == 'rfe':
            self._fit_rfe(X, y, n_features_to_select)
        elif self.method == 'ensemble':
            self._fit_ensemble(X, y, n_features_to_select)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted_ = True

        if self.verbose:
            n_selected = len(self.selected_features_)
            print(f"\nFeature Selection ({self.method}):")
            print(f"  Total features: {n_features_total}")
            print(f"  Selected features: {n_selected} ({n_selected / n_features_total * 100:.1f}%)")

        return self

    def transform(self, X):
        """Transform X by selecting features."""
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted first")

        if hasattr(X, 'columns'):
            return X[self.selected_features_]
        else:
            # Get indices of selected features
            indices = [self.feature_names_.index(f) for f in self.selected_features_]
            return X[:, indices]

    def _fit_variance(self, X, y):
        """Remove low-variance features."""
        X_array = X.values if hasattr(X, 'values') else X

        # Use a fraction of the maximum variance as threshold
        variances = np.var(X_array, axis=0)
        threshold_value = np.percentile(variances, self.threshold * 100)

        selector = VarianceThreshold(threshold=threshold_value)
        selector.fit(X_array)

        self.selected_features_ = [f for f, s in zip(self.feature_names_, selector.get_support()) if s]
        self.selector_ = selector

    def _fit_correlation(self, X, y):
        """Remove highly correlated features."""
        X_df = pd.DataFrame(X, columns=self.feature_names_) if not hasattr(X, 'columns') else X

        # Calculate correlation matrix
        corr_matrix = X_df.corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Features to drop
        to_drop = set()
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            correlated_features = list(upper_tri.index[upper_tri[column] > self.threshold])
            to_drop.update(correlated_features)

        self.selected_features_ = [f for f in self.feature_names_ if f not in to_drop]

    def _fit_chi2(self, X, y, n_features):
        """Chi-squared test for non-negative features."""
        X_array = X.values if hasattr(X, 'values') else X

        # Ensure non-negative values
        X_non_negative = MinMaxScaler().fit_transform(X_array)

        # Ensure n_features is an integer
        n_features_int = int(n_features)
        selector = SelectKBest(chi2, k=n_features_int)
        selector.fit(X_non_negative, y)

        self.selected_features_ = [f for f, s in zip(self.feature_names_, selector.get_support()) if s]
        self.selector_ = selector

    def _fit_anova(self, X, y, n_features):
        """ANOVA F-test."""
        X_array = X.values if hasattr(X, 'values') else X

        # Ensure n_features is an integer
        n_features_int = int(n_features)
        selector = SelectKBest(f_classif, k=n_features_int)
        selector.fit(X_array, y)

        self.selected_features_ = [f for f, s in zip(self.feature_names_, selector.get_support()) if s]
        self.selector_ = selector

    def _fit_mutual_info(self, X, y, n_features):
        """Mutual information."""
        X_array = X.values if hasattr(X, 'values') else X

        # Ensure n_features is an integer
        n_features_int = int(n_features)
        selector = SelectKBest(
            lambda X, y: mutual_info_classif(X, y, random_state=self.random_state),
            k=n_features_int
        )
        selector.fit(X_array, y)

        self.selected_features_ = [f for f, s in zip(self.feature_names_, selector.get_support()) if s]
        self.selector_ = selector

    def _fit_tree_importance(self, X, y, n_features):
        """Tree-based feature importance using XGBoost."""
        X_array = X.values if hasattr(X, 'values') else X

        # Use XGBoost for feature importance
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        clf.fit(X_array, y)

        # Get feature importances
        importances = clf.feature_importances_
        # Ensure n_features is an integer
        n_features_int = int(n_features)
        indices = np.argsort(importances)[::-1][:n_features_int]

        self.selected_features_ = [self.feature_names_[i] for i in indices]

    def _fit_lasso(self, X, y, n_features):
        """L1-based feature selection."""
        X_array = X.values if hasattr(X, 'values') else X

        # Use logistic regression with L1 penalty
        lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            random_state=self.random_state,
            max_iter=1000
        )

        # Ensure n_features is an integer
        n_features_int = int(n_features)
        selector = SelectFromModel(lasso, max_features=n_features_int)
        selector.fit(X_array, y)

        self.selected_features_ = [f for f, s in zip(self.feature_names_, selector.get_support()) if s]
        self.selector_ = selector

    def _fit_rfe(self, X, y, n_features):
        """Recursive Feature Elimination."""
        X_array = X.values if hasattr(X, 'values') else X

        # Use a fast estimator for RFE
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Ensure n_features is an integer
        n_features_int = int(n_features)
        selector = RFE(estimator, n_features_to_select=n_features_int, step=0.1)
        selector.fit(X_array, y)

        self.selected_features_ = [f for f, s in zip(self.feature_names_, selector.get_support()) if s]
        self.selector_ = selector

    def _fit_ensemble(self, X, y, n_features):
        """Ensemble of multiple feature selection methods."""
        X_array = X.values if hasattr(X, 'values') else X

        # Ensure n_features is an integer
        n_features_int = int(n_features)

        # Methods to include in ensemble
        methods = ['anova', 'mutual_info', 'tree_importance']
        feature_votes = {f: 0 for f in self.feature_names_}

        # Run each method and collect votes
        for method in methods:
            selector = FeatureSelector(
                method=method,
                n_features=n_features_int,
                random_state=self.random_state,
                verbose=False
            )
            selector.fit(X, y)

            for feature in selector.selected_features_:
                feature_votes[feature] += 1

        # Select features that appear in at least half of the methods
        min_votes = len(methods) // 2 + 1
        selected = [f for f, votes in feature_votes.items() if votes >= min_votes]

        # If we have too few features, add the top voted ones
        if len(selected) < n_features_int:
            remaining = sorted(
                [(f, v) for f, v in feature_votes.items() if f not in selected],
                key=lambda x: x[1],
                reverse=True
            )
            selected.extend([f for f, v in remaining[:n_features_int - len(selected)]])

        self.selected_features_ = selected[:n_features_int]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.selected_features_


class FeatureSelectionPipeline:
    """
    Helper class to create pipelines with feature selection.
    """

    @staticmethod
    def add_feature_selection(pipeline, method='ensemble', n_features='auto',
                              position='after_preprocessing'):
        """
        Add feature selection to an existing pipeline.

        Parameters:
        -----------
        pipeline : sklearn.pipeline.Pipeline
            The existing pipeline
        method : str
            Feature selection method
        n_features : int, float, or 'auto'
            Number of features to select
        position : str
            Where to add feature selection:
            - 'after_preprocessing': After all preprocessing steps
            - 'before_classifier': Just before the classifier
        """
        from sklearn.pipeline import Pipeline

        # Get pipeline steps
        steps = list(pipeline.steps)

        # Find where to insert feature selection
        if position == 'after_preprocessing':
            # Insert after the last preprocessing step (before classifier)
            insert_idx = len(steps) - 1
        else:  # before_classifier
            insert_idx = len(steps) - 1

        # Create feature selector
        feature_selector = ('feature_selection', FeatureSelector(
            method=method,
            n_features=n_features,
            random_state=42,
            verbose=True
        ))

        # Insert into pipeline
        steps.insert(insert_idx, feature_selector)

        # Return new pipeline
        return Pipeline(steps)


def analyze_feature_importance(pipeline, X, y, top_n=20):
    """
    Analyze and visualize feature importance after selection.
    """
    import matplotlib.pyplot as plt

    # Get the feature selector from pipeline
    if 'feature_selection' in pipeline.named_steps:
        selector = pipeline.named_steps['feature_selection']
        selected_features = selector.selected_features_
    else:
        print("No feature selection step found in pipeline")
        return None

    # Transform data through preprocessing
    preproc_steps = []
    for name, step in pipeline.steps:
        if name == 'feature_selection':
            break
        preproc_steps.append((name, step))

    from sklearn.pipeline import Pipeline
    preproc_pipeline = Pipeline(preproc_steps)
    X_preprocessed = preproc_pipeline.transform(X)

    # Get feature names after preprocessing
    if hasattr(X_preprocessed, 'columns'):
        all_features = X_preprocessed.columns.tolist()
    else:
        all_features = [f'feature_{i}' for i in range(X_preprocessed.shape[1])]

    # Train a tree model to get importances
    X_selected = selector.transform(X_preprocessed)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss'
    )

    if hasattr(X_selected, 'values'):
        clf.fit(X_selected.values, y)
    else:
        clf.fit(X_selected, y)

    # Get importances
    importances = clf.feature_importances_

    # Ensure we don't have mismatched lengths
    n_features = min(len(selected_features), len(importances))

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': selected_features[:n_features],
        'importance': importances[:n_features]
    }).sort_values('importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(min(top_n, len(importance_df)))

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {len(top_features)} Selected Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return importance_df