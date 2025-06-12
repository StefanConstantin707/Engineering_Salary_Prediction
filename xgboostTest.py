import warnings

from Classes.OutlierDetection import OutlierDetector

from Classes.OrdinalClassifier import OrdinalClassifier

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet.*",
    category=FutureWarning
)

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import seaborn as sns

from Classes.DataHandler import DataHandler, FeatureExtractor
from xgboost import XGBClassifier


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
        cluster_transformer = final_pipeline.named_steps['preproc'] \
            .named_steps['preprocessor'] \
            .named_transformers_['jobdesc_cluster']
        kmeans_model = cluster_transformer.kmeans
        n_clusters = kmeans_model.n_clusters
    except (KeyError, AttributeError):
        print("Could not find a fitted cluster transformer in the final pipeline.")
        print("Skipping cluster analysis.")
        return

    feature_extractor = FeatureExtractor()
    job_desc_cols = feature_extractor.job_desc_cols
    X_job_desc = X[job_desc_cols].copy()
    X_job_desc.replace(0, np.nan, inplace=True)

    X_imputed = cluster_transformer.imputer.transform(X_job_desc)
    X_scaled = cluster_transformer.scaler.transform(X_imputed)
    cluster_labels = kmeans_model.predict(X_scaled)

    print(f"Analysis of the {n_clusters} clusters:")
    print("-" * 50)

    salary_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    y_values = y.values.ravel() if hasattr(y, 'values') else y.ravel()

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


def main():
    # 1) Load raw data with fixed clusters
    print("Loading data with 9 job description clusters...")
    dh = DataHandler(n_job_clusters=3, use_cluster_probabilities=True)
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
    dh_clean = DataHandler(n_job_clusters=3, use_cluster_probabilities=True)

    # 3) Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # 4) Define search space for XGBoost hyperparameters
    search_space = {
        'clf__estimator__n_estimators': Integer(100, 1000),
        'clf__estimator__max_depth': Integer(3, 10),
        'clf__estimator__learning_rate': Real(1e-3, 0.3, prior='log-uniform'),
        'clf__estimator__subsample': Real(0.5, 1.0),
        'clf__estimator__colsample_bytree': Real(0.5, 1.0),
        'clf__estimator__gamma': Real(0, 5),
        'clf__estimator__reg_alpha': Real(1e-8, 1.0, prior='log-uniform'),
        'clf__estimator__reg_lambda': Real(1e-8, 1.0, prior='log-uniform')
    }

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION FOR XGBOOST ORDINAL CLASSIFIER")
    print("(Training on cleaned data without outliers)")
    print("=" * 70)

    xgb_base = XGBClassifier(eval_metric='mlogloss', random_state=0)
    pipe_ordinal = Pipeline([
        ("preproc", dh_clean.pipeline),
        ("clf", OrdinalClassifier(
            estimator=xgb_base,
            n_jobs=-1
        ))
    ])

    print("\nOptimizing hyperparameters on cleaned data...")
    bayes_search = BayesSearchCV(
        pipe_ordinal,
        search_space,
        n_iter=50,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=0,
        verbose=2
    )

    # Fit on cleaned data
    bayes_search.fit(X_raw_clean, y_clean)

    print(f"\nBest parameters:")
    for param in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']:
        key = f"clf__estimator__{param}"
        print(f"  {param}: {bayes_search.best_params_[key]}")
    print(f"Best CV Score: {bayes_search.best_score_:.4f} (+/- {bayes_search.cv_results_['std_test_score'][bayes_search.best_index_]:.4f})")

    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING")
    print("=" * 70)
    pipe_final = bayes_search.best_estimator_

    print("Running final cross-validation with the best pipeline on cleaned data...")
    cv_scores_final = cross_val_score(pipe_final, X_raw_clean, y_clean, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"Cross-validation scores: {cv_scores_final}")
    print(f"Mean CV accuracy: {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std():.4f})")

    print("\nTraining final model on cleaned training data...")
    pipe_final.fit(X_raw_clean, y_clean)

    print("\nGenerating predictions...")
    # Evaluate on original training data (including outliers) to see full performance
    y_train_pred_all = pipe_final.predict(X_raw)
    train_accuracy_all = accuracy_score(y, y_train_pred_all)

    # Also evaluate on clean training data
    y_train_pred_clean = pipe_final.predict(X_raw_clean)
    train_accuracy_clean = accuracy_score(y_clean, y_train_pred_clean)

    # Test predictions
    y_test_pred = pipe_final.predict(X_test_raw)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    results_df = pd.DataFrame({
        "Metric": [
            "Outliers removed",
            "Fixed number of clusters",
            "Best XGBoost hyperparameters summary",
            "Cross-validation accuracy (clean data)",
            "Training accuracy (clean data)",
            "Training accuracy (all data)"
        ],
        "Value": [
            f"{np.sum(~inlier_mask)} ({np.sum(~inlier_mask) / len(X_raw) * 100:.2f}%)",
            "9",
            "See printed params above",
            f"{cv_scores_final.mean():.4f} ± {cv_scores_final.std():.4f}",
            f"{train_accuracy_clean:.4f}",
            f"{train_accuracy_all:.4f}"
        ]
    })
    print(results_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TRAINING PERFORMANCE ANALYSIS")
    print("=" * 70)
    class_labels = ['Low', 'Medium', 'High']

    # Confusion matrix on all training data
    print("\nPerformance on ALL training data (including outliers):")
    cm_train_all = confusion_matrix(y, y_train_pred_all)
    print("\nConfusion Matrix (All Training Data):")
    print("Predicted:", "   ".join([f"{label:>6}" for label in class_labels]))
    for i, (actual_label, row) in enumerate(zip(class_labels, cm_train_all)):
        print(f"Actual {actual_label:>6}: {' '.join([f'{val:6d}' for val in row])}")
    print("\nClassification Report:")
    print(classification_report(y, y_train_pred_all, target_names=class_labels))
    plot_confusion_matrix(cm_train_all, class_labels, "Training Data Confusion Matrix (All Data)")

    # Confusion matrix on clean training data
    print("\n" + "-" * 50)
    print("Performance on CLEAN training data (outliers removed):")
    cm_train_clean = confusion_matrix(y_clean, y_train_pred_clean)
    print("\nConfusion Matrix (Clean Training Data):")
    print("Predicted:", "   ".join([f"{label:>6}" for label in class_labels]))
    for i, (actual_label, row) in enumerate(zip(class_labels, cm_train_clean)):
        print(f"Actual {actual_label:>6}: {' '.join([f'{val:6d}' for val in row])}")

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

    # Save test predictions
    submission_df = pd.DataFrame(
        {'salary_category': [class_labels[pred] for pred in y_test_pred]},
        index=test_idx
    )
    submission_df.index.name = 'id'
    submission_df.to_csv('submission_with_outlier_removal.csv', index=True)
    print("\nSubmission file saved to 'submission_with_outlier_removal.csv'")

    # Analyze final clusters on clean data
    analyze_final_clusters(pipe_final, X_raw_clean, y_clean)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return pipe_final, outlier_detector


if __name__ == "__main__":
    main()