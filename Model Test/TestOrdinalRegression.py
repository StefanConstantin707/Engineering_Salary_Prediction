import time
import warnings

from sklearn.base import TransformerMixin, BaseEstimator

from Classes.RunAnalyzer import RunTracker

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet.*",
    category=FutureWarning
)

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import seaborn as sns

from Classes.DataHandler import DataHandler, FeatureExtractor
from Classes.OutlierDetection import OutlierDetector
from Classes.FeatureSelection import FeatureSelector, analyze_feature_importance
from xgboostTest import OrdinalClassifier


class PolyToDataFrame(TransformerMixin, BaseEstimator):
    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        # we will create the PolynomialFeatures in fit
    def fit(self, X, y=None):
        # X may be a DataFrame or array: we need column names if DataFrame
        if hasattr(X, "values") and hasattr(X, "columns"):
            X_arr = X.values
            self.input_feature_names_ = list(X.columns)
        else:
            X_arr = X
            # If no column names available, create generic names:
            n_features = X_arr.shape[1]
            self.input_feature_names_ = [f"feature_{i}" for i in range(n_features)]
        # instantiate and fit PolynomialFeatures on X_arr to learn powers_
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self.poly_.fit(X_arr)
        # generate output names once
        self.output_feature_names_ = list(self.poly_.get_feature_names_out(self.input_feature_names_))
        return self

    def transform(self, X):
        if hasattr(X, "values") and hasattr(X, "columns"):
            X_arr = X.values
            index = X.index
        else:
            X_arr = X
            index = None
        arr_poly = self.poly_.transform(X_arr)
        return pd.DataFrame(arr_poly, columns=self.output_feature_names_, index=index)


def plot_confusion_matrix(cm, labels, title):
    """Plot confusion matrix with nice formatting."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def analyze_final_clusters(final_pipeline, X, y):
    """
    Analyzes and prints the salary distribution for each cluster
    from the final, fitted pipeline.
    """
    print("\n" + "=" * 70)
    print("FINAL CLUSTER ANALYSIS")
    print("=" * 70)

    try:
        # Navigate the nested pipeline to find the fitted cluster transformer and its KMeans model
        cluster_transformer = final_pipeline.named_steps['preproc'] \
            .named_steps['preprocessor'] \
            .named_transformers_['jobdesc_cluster']
        kmeans_model = cluster_transformer.kmeans
        n_clusters = kmeans_model.n_clusters
    except (KeyError, AttributeError) as e:
        print(f"Could not find a fitted cluster transformer in the final pipeline: {e}")
        print("Skipping cluster analysis.")
        return

    # To get the cluster labels, we must pass the original data through the
    # same preprocessing steps used by the clustering model.
    feature_extractor = FeatureExtractor()
    job_desc_cols = feature_extractor.job_desc_cols
    X_job_desc = X[job_desc_cols].copy()

    # Convert to numpy array if it's a DataFrame
    X_array = X_job_desc.values if hasattr(X_job_desc, 'values') else X_job_desc

    # Replace zeros with NaN (matching the transformer's behavior)
    X_array = np.where(X_array == 0, np.nan, X_array)

    # Filter out rows where the first column is NaN (matching the transformer's behavior)
    nan_mask = np.isnan(X_array[:, 0])
    X_valid = X_array[~nan_mask]

    if len(X_valid) == 0:
        print("No valid data points for cluster analysis (all have NaN in first job_desc column)")
        return

    # Scale the data using the fitted scaler from the cluster transformer
    try:
        X_scaled = cluster_transformer.scaler.transform(X_valid)
    except Exception as e:
        print(f"Error scaling data: {e}")
        return

    # Predict cluster labels
    cluster_labels_valid = kmeans_model.predict(X_scaled)

    # Create full cluster labels array (with -1 for invalid rows)
    cluster_labels = np.full(len(X_array), -1)
    cluster_labels[~nan_mask] = cluster_labels_valid

    # Analyze salary distribution per cluster
    print(f"Analysis of the {n_clusters} clusters found by the optimization:")
    print(f"Total samples: {len(X_array)} (Valid: {len(X_valid)}, Invalid: {np.sum(nan_mask)})")
    print("-" * 50)

    salary_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    y_values = y.values.ravel() if hasattr(y, 'values') else y.ravel()

    # First show invalid data statistics
    if np.sum(nan_mask) > 0:
        print(f"\nInvalid data (n={np.sum(nan_mask)}):")
        invalid_salaries = y_values[nan_mask]
        for salary_val, salary_name in salary_map.items():
            count = np.sum(invalid_salaries == salary_val)
            pct = 100 * count / len(invalid_salaries) if len(invalid_salaries) > 0 else 0
            print(f"  {salary_name}: {count} ({pct:.1f}%)")

    # Then show cluster statistics
    for cluster_id in range(n_clusters):
        mask = (cluster_labels == cluster_id)
        cluster_size = np.sum(mask)

        if cluster_size == 0:
            print(f"\nCluster {cluster_id} (n=0): No data points assigned.")
            continue

        print(f"\nCluster {cluster_id} (n={cluster_size}):")
        cluster_salaries = y_values[mask]
        for salary_val, salary_name in salary_map.items():
            count = np.sum(cluster_salaries == salary_val)
            pct = 100 * count / cluster_size
            print(f"  {salary_name}: {count} ({pct:.1f}%)")

    # Additional cluster statistics
    print("\n" + "-" * 50)
    print("Cluster Summary Statistics:")
    print("-" * 50)

    if len(X_valid) > 0:
        # Calculate distances to cluster centers for valid data
        distances = kmeans_model.transform(X_scaled)
        min_distances = np.min(distances, axis=1)

        for cluster_id in range(n_clusters):
            cluster_mask_valid = (cluster_labels_valid == cluster_id)
            if np.sum(cluster_mask_valid) > 0:
                cluster_distances = min_distances[cluster_mask_valid]
                print(f"Cluster {cluster_id}: Avg distance to center: {np.mean(cluster_distances):.3f} "
                      f"(±{np.std(cluster_distances):.3f})")

    print("=" * 70)


def on_step(optim_result):
    """Callback function called after each iteration."""
    n_calls = len(optim_result.func_vals)
    current_best = -optim_result.fun  # Negative because skopt minimizes
    current_params = optim_result.x

    print(f"\nIteration {n_calls} | Current best: {current_best:.4f}")
    return False  # Return False to continue optimization


def enhanced_bayes_search_with_tracking(pipe_ordinal, search_space, X_train, y_train,
                                        cv, run_name="logistic_regression", n_iter=50):
    """
    Enhanced Bayesian search with detailed iteration tracking.
    This version uses the correct approach for BayesSearchCV.
    """
    # Initialize run tracker
    run_tracker = RunTracker(run_name=run_name)

    # Method 1: Use verbose output and parse results after fitting
    print(f"\nStarting Bayesian optimization with {n_iter} iterations...")
    print("=" * 70)

    # Create BayesSearchCV
    bayes_search = BayesSearchCV(
        pipe_ordinal,
        search_space,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=0,
        verbose=2,  # This will print progress
        return_train_score=True
    )

    # Fit the model
    import time
    start_time = time.time()
    bayes_search.fit(X_train, y_train)
    total_time = time.time() - start_time

    print(f"\nOptimization completed in {total_time:.2f} seconds")

    # Parse all results after fitting
    results_df = pd.DataFrame(bayes_search.cv_results_)

    # Log each iteration to the tracker
    for i in range(len(results_df)):
        # Extract parameters for this iteration
        params = {}
        for col in results_df.columns:
            if col.startswith('param_'):
                param_name = col.replace('param_', '')
                params[param_name] = results_df.loc[i, col]

        # Calculate duration (use fit time + score time)
        duration = results_df.loc[i, 'mean_fit_time'] + results_df.loc[i, 'mean_score_time']

        # Log to tracker
        run_tracker.log_iteration(
            iteration=i + 1,
            params=params,
            score=results_df.loc[i, 'mean_test_score'],
            std=results_df.loc[i, 'std_test_score'],
            duration=duration
        )

    # Print summary of best iterations
    print("\n" + "=" * 70)
    print("TOP 5 ITERATIONS")
    print("=" * 70)

    top_5 = results_df.nlargest(5, 'mean_test_score')
    for idx, (i, row) in enumerate(top_5.iterrows()):
        print(f"\n{idx + 1}. Iteration {i + 1}: Score = {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print("   Parameters:")
        for col in results_df.columns:
            if col.startswith('param_'):
                param_name = col.replace('param_', '').split('__')[-1]  # Get short name
                print(f"     {param_name}: {row[col]}")

    return bayes_search, run_tracker


def main():
    # 1) Load raw data
    print("Loading data...")
    # Initialize with a base number of clusters. BayesSearchCV will optimize this.
    dh = DataHandler(n_job_clusters=5, use_cluster_probabilities=True)
    X_raw, y_df, train_idx = dh.get_train_data_raw()
    X_test_raw, test_idx = dh.get_test_data_raw()
    y = y_df["salary_category"]

    print(f"Original training data shape: {X_raw.shape}")
    print(f"Test data shape: {X_test_raw.shape}")
    print(f"Original target distribution:\n{y.value_counts().sort_index()}")

    # 2) OUTLIER DETECTION SECTION
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION AND REMOVAL")
    print("=" * 70)

    # Apply preprocessing to get features for outlier detection
    print("\nApplying preprocessing for outlier detection...")
    X_preprocessed = dh.pipeline.fit_transform(X_raw, y)

    # Detect outliers using ensemble method
    outlier_detector = OutlierDetector(
        method='ensemble',  # Options: 'isolation_forest', 'lof', 'elliptic', 'zscore', 'iqr', 'ensemble'
        contamination='auto',  # Will use 5% as default
        random_state=42,
        verbose=True
    )

    # Fit the outlier detector
    outlier_detector.fit(X_preprocessed, y)
    inlier_mask = ~outlier_detector.outlier_mask_

    # Remove outliers from raw data and target
    X_raw_clean = X_raw.loc[inlier_mask].copy()
    y_clean = y.loc[inlier_mask].copy()

    print(f"\nCleaned training data shape: {X_raw_clean.shape}")
    print(f"Removed {np.sum(~inlier_mask)} outliers ({np.sum(~inlier_mask) / len(X_raw) * 100:.2f}%)")
    print(f"Cleaned target distribution:\n{y_clean.value_counts().sort_index()}")

    # Create new DataHandler for clean data (to refit preprocessing on clean data)
    print("\nRefitting preprocessing pipeline on cleaned data...")
    dh_clean = DataHandler(n_job_clusters=5, use_cluster_probabilities=True)

    # 3) Define CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # 4) Define search space for the model, including the number of clusters and feature selection
    search_space = {
        # Search for the ideal number of job description clusters
        'preproc__preprocessor__jobdesc_cluster__n_clusters': Categorical([3]),

        # Feature selection parameters
        'feature_selection__n_features': Real(0.27, 0.33),

        # Polynomial features search space
        'poly__degree': Categorical([2]),
        'poly__interaction_only': [True],

        # Classifier search space
        'clf__estimator__C': Real(1e-4, 1.0, prior='log-uniform')
    }

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION WITH OUTLIER REMOVAL AND FEATURE SELECTION")
    print("=" * 70)


    # 5) Create pipeline with feature selection and ordinal classifier
    pipe_ordinal = Pipeline([
        ("preproc", dh_clean.pipeline),
        ("poly", PolyToDataFrame(
            include_bias=False
        )),
        ("feature_selection", FeatureSelector(
            method='tree_importance', # Set a default method
            n_features=1.0,
            random_state=42,
            verbose=False
        )),
        ("clf", OrdinalClassifier(
            estimator=LogisticRegression(
                penalty="l2",
                solver="saga",
                max_iter=5000,
                random_state=0
            ),
            n_jobs=-1
        ))
    ])


    # 6) Bayesian optimization WITH TRACKING
    print("\nOptimizing hyperparameters with detailed tracking...")

    bayes_search, run_tracker = enhanced_bayes_search_with_tracking(
        pipe_ordinal=pipe_ordinal,
        search_space=search_space,
        X_train=X_raw_clean,
        y_train=y_clean,
        cv=cv,
        run_name="logistic_outlier_featsel",
        n_iter=5
    )

    # Save optimization results to tracker
    run_tracker.save_optimization_results(
        bayes_search=bayes_search,
        X_train=X_raw_clean,
        y_train=y_clean,
        outlier_detector=outlier_detector,
        feature_selector_info={
            "method": 'tree_importance',
            "n_features_ratio": bayes_search.best_params_.get('feature_selection__n_features'),
            "n_features_selected": None
        }
    )

    print(f"\nBest parameters:")
    print(f"  Optimal number of clusters: {bayes_search.best_params_['preproc__preprocessor__jobdesc_cluster__n_clusters']}")
    print(f"  Feature selection ratio: {bayes_search.best_params_['feature_selection__n_features']:.2%}")
    print(f"  Polynomial degree: {bayes_search.best_params_['poly__degree']}")
    print(f"  Interaction only: {bayes_search.best_params_['poly__interaction_only']}")
    print(f"  C parameter: {bayes_search.best_params_['clf__estimator__C']:.6f}")
    print(f"Best CV Score: {bayes_search.best_score_:.4f} (+/- {bayes_search.cv_results_['std_test_score'][bayes_search.best_index_]:.4f})")

    # 7) REFIT WITH BEST PARAMETERS AND VERBOSE FEATURE SELECTION
    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING WITH FEATURE SELECTION")
    print("=" * 70)

    best_params = bayes_search.best_params_
    optimal_clusters = best_params['preproc__preprocessor__jobdesc_cluster__n_clusters']
    dh_clean_optimal = DataHandler(n_job_clusters=optimal_clusters, use_cluster_probabilities=True)

    pipe_final = Pipeline([
        ("preproc", dh_clean_optimal.pipeline),
        ("poly", PolyToDataFrame(
            degree=best_params['poly__degree'],
            interaction_only=best_params['poly__interaction_only'],
            include_bias=False
        )),
        ("feature_selection", FeatureSelector(
            method='tree_importance',
            n_features=best_params['feature_selection__n_features'],
            random_state=42,
            verbose=True
        )),
        ("clf", OrdinalClassifier(
            estimator=LogisticRegression(
                C=best_params['clf__estimator__C'],
                penalty="l2",
                solver="saga",
                max_iter=5000,
                random_state=0
            ),
            n_jobs=-1
        ))
    ])

    # 8) Final cross-validation with optimized parameters
    print("\nRunning final cross-validation with the best pipeline on cleaned data...")
    cv_scores_final = cross_val_score(pipe_final, X_raw_clean, y_clean, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"Cross-validation scores: {cv_scores_final}")
    print(f"Mean CV accuracy: {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std():.4f})")

    if best_params['poly__degree'] > 1:
        print("\n⚠️  Warning: Using polynomial features may increase overfitting risk")
        print("   Monitor the gap between training and CV accuracy carefully")

    # 9) Train final model on cleaned training data
    print("\nTraining final model on cleaned training data...")
    pipe_final.fit(X_raw_clean, y_clean)

    # 10) FEATURE IMPORTANCE ANALYSIS
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    importance_df = analyze_feature_importance(pipe_final, X_raw_clean, y_clean, top_n=20)

    # 11) Generate predictions
    print("\nGenerating predictions...")
    y_train_pred_all = pipe_final.predict(X_raw)
    train_accuracy_all = accuracy_score(y, y_train_pred_all)
    y_train_pred_clean = pipe_final.predict(X_raw_clean)
    train_accuracy_clean = accuracy_score(y_clean, y_train_pred_clean)
    y_test_pred = pipe_final.predict(X_test_raw)

    # 12) Display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # CORRECTED FEATURE COUNTING
    # We need to pass y to fit_transform because a transformer in the preproc pipeline requires it.
    temp_preproc_pipe = Pipeline(pipe_final.steps[:1])
    # Pass both X and y to the fit_transform call
    X_preprocessed_sample = temp_preproc_pipe.fit_transform(X_raw_clean.head(1), y_clean.head(1))
    n_features_before_poly = X_preprocessed_sample.shape[1]

    temp_poly_pipe = Pipeline(pipe_final.steps[:2])
    # Pass both X and y here as well
    X_poly_sample = temp_poly_pipe.fit_transform(X_raw_clean.head(1), y_clean.head(1))
    n_features_after_poly = X_poly_sample.shape[1]

    n_features_selected = len(pipe_final.named_steps['feature_selection'].selected_features_)

    results_df = pd.DataFrame({
        "Metric": [
            "Outliers removed",
            "Optimal number of clusters",
            "Feature selection method",
            "Features before polynomial",
            "Features after polynomial",
            "Features selected",
            "Polynomial degree",
            "Interaction terms only",
            "Best C parameter",
            "Cross-validation accuracy (clean data)",
            "Training accuracy (clean data)",
            "Training accuracy (all data)"
        ],
        "Value": [
            f"{np.sum(~inlier_mask)} ({np.sum(~inlier_mask) / len(X_raw) * 100:.2f}%)",
            f"{optimal_clusters}",
            'tree_importance',
            f"{n_features_before_poly}",
            f"{n_features_after_poly}",
            f"{n_features_selected}/{n_features_after_poly} ({n_features_selected / n_features_after_poly * 100:.1f}%)",
            f"{best_params['poly__degree']}",
            f"{best_params['poly__interaction_only']}",
            f"{best_params['clf__estimator__C']:.6f}",
            f"{cv_scores_final.mean():.4f} ± {cv_scores_final.std():.4f}",
            f"{train_accuracy_clean:.4f}",
            f"{train_accuracy_all:.4f}"
        ]
    })


    print(results_df.to_string(index=False))

    # 13) Confusion Matrix and Classification Report
    print("\n" + "=" * 70)
    print("TRAINING PERFORMANCE ANALYSIS")
    print("=" * 70)

    class_labels = ['Low', 'Medium', 'High']

    print("\nPerformance on ALL training data (including outliers):")
    cm_train_all = confusion_matrix(y, y_train_pred_all)
    print("\nConfusion Matrix (All Training Data):")
    print("Predicted:", "   ".join([f"{label:>6}" for label in class_labels]))
    for i, (actual_label, row) in enumerate(zip(class_labels, cm_train_all)):
        print(f"Actual {actual_label:>6}: {' '.join([f'{val:6d}' for val in row])}")
    print("\nClassification Report:")
    print(classification_report(y, y_train_pred_all, target_names=class_labels))
    plot_confusion_matrix(cm_train_all, class_labels, "Training Data Confusion Matrix (All Data)")

    print("\n" + "-" * 50)
    print("Performance on CLEAN training data (outliers removed):")
    cm_train_clean = confusion_matrix(y_clean, y_train_pred_clean)
    print("\nConfusion Matrix (Clean Training Data):")
    print("Predicted:", "   ".join([f"{label:>6}" for label in class_labels]))
    for i, (actual_label, row) in enumerate(zip(class_labels, cm_train_clean)):
        print(f"Actual {actual_label:>6}: {' '.join([f'{val:6d}' for val in row])}")

    # 14) Test predictions summary
    print("\n" + "=" * 70)
    print("TEST DATA PREDICTIONS")
    print("=" * 70)

    test_pred_counts = pd.Series(y_test_pred).value_counts().sort_index()
    test_summary = pd.DataFrame({
        "Class": class_labels,
        "Predictions": [test_pred_counts.get(i, 0) for i in range(3)],
        "Percentage": [f"{test_pred_counts.get(i, 0) / len(y_test_pred) * 100:.1f}%" for i in range(3)]
    })
    print(test_summary.to_string(index=False))

    # 15) Save test predictions
    submission_df = pd.DataFrame(
        {'salary_category': [class_labels[pred] for pred in y_test_pred]},
        index=test_idx
    )
    submission_df.index.name = 'id'
    submission_df.to_csv('logistic_with_outlier_removal_and_feature_selection.csv', index=True)
    print("\nSubmission file saved to 'logistic_with_outlier_removal_and_feature_selection.csv'")

    # 16) Analyze final clusters on clean data
    analyze_final_clusters(pipe_final, X_raw_clean, y_clean)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Print top selected features
    print("\nTop 10 Selected Features:")
    if importance_df is not None and len(importance_df) > 0:
        for i, row in importance_df.head(10).iterrows():
            print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")

        if importance_df is not None:
            run_tracker.save_feature_importance(importance_df)

        run_tracker.add_diagnostic("train_accuracy_clean", train_accuracy_clean)
        run_tracker.add_diagnostic("train_accuracy_all", train_accuracy_all)
        run_tracker.add_diagnostic("cv_mean", cv_scores_final.mean())
        run_tracker.add_diagnostic("cv_std", cv_scores_final.std())
        run_tracker.add_diagnostic("n_features_selected", n_features_selected)
        run_tracker.add_diagnostic("n_features_after_poly", n_features_after_poly)

        run_tracker.print_run_summary()

        run_tracker.save_to_file(include_model=True, model=pipe_final)

        return pipe_final, outlier_detector, run_tracker


if __name__ == "__main__":
    main()