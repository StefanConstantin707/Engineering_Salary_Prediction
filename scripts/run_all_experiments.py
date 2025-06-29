#!/usr/bin/env python3
"""
Run all model experiments with different configurations.
This script runs multiple models and preprocessing strategies to find the best approach.
"""

import os
import sys
import time
import warnings
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from src.data.data_handler import DataHandler
from src.models.ordinal_classifier import OrdinalClassifier
from src.preprocessing.outlier_detection import OutlierDetector
from src.preprocessing.feature_selection import FeatureSelector
from src.utils.run_tracker import RunTracker
from src.data.transformers import PolynomialFeaturesDataFrame

# Configuration for different experiments
EXPERIMENTS = {
    'xgboost_basic': {
        'name': 'XGBoost Basic',
        'model_class': XGBClassifier,
        'model_params': {
            'eval_metric': 'mlogloss',
            'random_state': 42
        },
        'search_space': {
            'clf__estimator__n_estimators': Integer(100, 1000),
            'clf__estimator__max_depth': Integer(3, 15),
            'clf__estimator__learning_rate': Real(0.001, 0.1, prior='log-uniform'),
            'clf__estimator__subsample': Real(0.5, 1.0),
            'clf__estimator__colsample_bytree': Real(0.5, 1.0),
        },
        'preprocessing': {
            'n_clusters': 3,
            'outlier_removal': True,
            'feature_selection': True
        }
    },

    'xgboost_advanced': {
        'name': 'XGBoost Advanced',
        'model_class': XGBClassifier,
        'model_params': {
            'eval_metric': 'mlogloss',
            'random_state': 42
        },
        'search_space': {
            'preproc__preprocessor__jobdesc_cluster__n_clusters': Categorical([3, 5, 7]),
            'feature_selection__n_features': Real(0.5, 0.9),
            'clf__estimator__n_estimators': Categorical([500, 800, 1000]),
            'clf__estimator__max_depth': Integer(8, 12),
            'clf__estimator__learning_rate': Real(0.001, 0.01, prior='log-uniform'),
            'clf__estimator__subsample': Real(0.7, 0.9),
            'clf__estimator__colsample_bytree': Real(0.6, 0.8),
            'clf__estimator__gamma': Real(0, 0.1),
            'clf__estimator__reg_alpha': Real(0.01, 1.0, prior='log-uniform'),
            'clf__estimator__reg_lambda': Real(0.01, 1.0, prior='log-uniform'),
        },
        'preprocessing': {
            'n_clusters': 'optimize',  # Will be optimized
            'outlier_removal': True,
            'feature_selection': True
        }
    },

    'lightgbm': {
        'name': 'LightGBM',
        'model_class': LGBMClassifier,
        'model_params': {
            'objective': 'multiclass',
            'random_state': 42,
            'verbosity': -1
        },
        'search_space': {
            'clf__estimator__n_estimators': Integer(100, 1000),
            'clf__estimator__num_leaves': Integer(10, 100),
            'clf__estimator__learning_rate': Real(0.001, 0.1, prior='log-uniform'),
            'clf__estimator__max_depth': Integer(3, 15),
            'clf__estimator__min_child_samples': Integer(5, 50),
            'clf__estimator__subsample': Real(0.5, 1.0),
            'clf__estimator__colsample_bytree': Real(0.5, 1.0),
        },
        'preprocessing': {
            'n_clusters': 3,
            'outlier_removal': True,
            'feature_selection': True
        }
    },

    'logistic_regression': {
        'name': 'Logistic Regression',
        'model_class': LogisticRegression,
        'model_params': {
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 5000,
            'random_state': 42
        },
        'search_space': {
            'poly__degree': Categorical([1, 2]),
            'poly__interaction_only': Categorical([True, False]),
            'feature_selection__n_features': Real(0.3, 0.7),
            'clf__estimator__C': Real(1e-4, 10.0, prior='log-uniform'),
        },
        'preprocessing': {
            'n_clusters': 3,
            'outlier_removal': True,
            'feature_selection': True,
            'polynomial_features': True
        }
    }
}


def run_experiment(experiment_config, X_train, y_train, cv, n_iter=50):
    """
    Run a single experiment with the given configuration.

    Parameters
    ----------
    experiment_config : dict
        Configuration for the experiment
    X_train : DataFrame
        Training features
    y_train : Series
        Training labels
    cv : cross-validator
        Cross-validation strategy
    n_iter : int
        Number of optimization iterations

    Returns
    -------
    best_estimator : Pipeline
        Best model found
    best_score : float
        Best cross-validation score
    run_tracker : RunTracker
        Experiment tracking object
    """
    config = experiment_config
    print(f"\n{'=' * 70}")
    print(f"Running experiment: {config['name']}")
    print(f"{'=' * 70}")

    # Initialize run tracker
    run_tracker = RunTracker(run_name=config['name'].lower().replace(' ', '_'))

    # Preprocessing setup
    prep_config = config['preprocessing']

    # 1. Outlier detection if enabled
    if prep_config.get('outlier_removal', False):
        print("\nDetecting outliers...")
        # Create a temporary pipeline for outlier detection
        temp_dh = DataHandler(n_job_clusters=prep_config.get('n_clusters', 3))
        X_preprocessed = temp_dh.pipeline.fit_transform(X_train, y_train)

        outlier_detector = OutlierDetector(
            method='ensemble',
            contamination='auto',
            random_state=42,
            verbose=True
        )
        outlier_detector.fit(X_preprocessed, y_train)
        inlier_mask = ~outlier_detector.outlier_mask_

        X_train_clean = X_train.loc[inlier_mask].copy()
        y_train_clean = y_train.loc[inlier_mask].copy()
    else:
        X_train_clean = X_train
        y_train_clean = y_train
        outlier_detector = None

    # 2. Build pipeline
    pipeline_steps = []

    # Data preprocessing
    n_clusters = prep_config.get('n_clusters', 3)
    if n_clusters == 'optimize':
        n_clusters = 3  # Default, will be optimized

    dh = DataHandler(n_job_clusters=n_clusters, use_cluster_probabilities=True)
    pipeline_steps.append(('preproc', dh.pipeline))

    # Polynomial features if needed
    if prep_config.get('polynomial_features', False):
        pipeline_steps.append(('poly', PolynomialFeaturesDataFrame()))

    # Feature selection
    if prep_config.get('feature_selection', False):
        pipeline_steps.append(('feature_selection', FeatureSelector(
            method='tree_importance',
            n_features=0.7,
            random_state=42,
            verbose=False
        )))

    # Model
    base_model = config['model_class'](**config['model_params'])
    pipeline_steps.append(('clf', OrdinalClassifier(estimator=base_model)))

    pipeline = Pipeline(pipeline_steps)

    # 3. Bayesian optimization
    print(f"\nStarting Bayesian optimization with {n_iter} iterations...")

    bayes_search = BayesSearchCV(
        pipeline,
        config['search_space'],
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    start_time = time.time()
    bayes_search.fit(X_train_clean, y_train_clean)
    duration = time.time() - start_time

    # 4. Save results
    run_tracker.save_optimization_results(
        bayes_search=bayes_search,
        X_train=X_train_clean,
        y_train=y_train_clean,
        outlier_detector=outlier_detector,
        feature_selector_info={
            'enabled': prep_config.get('feature_selection', False),
            'method': 'tree_importance'
        }
    )

    # Add additional diagnostics
    run_tracker.add_diagnostic('experiment_duration', duration)
    run_tracker.add_diagnostic('n_samples_used', len(X_train_clean))

    # Print results
    print(f"\nBest score: {bayes_search.best_score_:.4f}")
    print(f"Duration: {duration:.1f} seconds")
    print("\nBest parameters:")
    for param, value in bayes_search.best_params_.items():
        print(f"  {param}: {value}")

    # Save run data
    run_tracker.save_to_file(include_model=True, model=bayes_search.best_estimator_)

    return bayes_search.best_estimator_, bayes_search.best_score_, run_tracker


def main():
    """Run all experiments."""
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    dh = DataHandler(n_job_clusters=3)
    X_raw, y_df, _ = dh.get_train_data_raw()
    y = y_df["salary_category"]

    print(f"Data shape: {X_raw.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Results storage
    results = {}

    # Run each experiment
    for exp_name, exp_config in EXPERIMENTS.items():
        try:
            best_model, best_score, run_tracker = run_experiment(
                exp_config,
                X_raw,
                y,
                cv,
                n_iter=50  # Adjust as needed
            )

            results[exp_name] = {
                'model': best_model,
                'score': best_score,
                'tracker': run_tracker
            }

        except Exception as e:
            print(f"\nERROR in experiment {exp_name}: {str(e)}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for exp_name, result in results.items():
        print(f"{exp_name}: {result['score']:.4f}")

    # Find best overall
    if results:
        best_exp = max(results.items(), key=lambda x: x[1]['score'])
        print(f"\nBest experiment: {best_exp[0]} with score {best_exp[1]['score']:.4f}")

    print(f"\nEnd time: {datetime.now()}")
    print("All experiments complete!")


if __name__ == '__main__':
    main()