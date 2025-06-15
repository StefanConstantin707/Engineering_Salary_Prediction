import warnings
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
from xgboostTest import OrdinalClassifier

# Import LightGBM
from lightgbm import LGBMClassifier


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
    print("\n" + "="*70)
    print("FINAL CLUSTER ANALYSIS")
    print("="*70)

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


def run_lightgbm_pipeline(X_raw, y, X_test_raw, test_idx):
    print("\nRunning LightGBM pipeline with 3 clusters...")
    dh = DataHandler(n_job_clusters=3, use_cluster_probabilities=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search_space = {
        'clf__estimator__n_estimators': Integer(100, 1000),
        'clf__estimator__num_leaves': Integer(7, 255),
        'clf__estimator__learning_rate': Real(1e-3, 0.3, prior='log-uniform'),
        'clf__estimator__max_depth': Integer(3, 15),
        'clf__estimator__min_child_samples': Integer(5, 100),
        'clf__estimator__subsample': Real(0.5, 1.0),
        'clf__estimator__colsample_bytree': Real(0.5, 1.0),
        'clf__estimator__reg_alpha': Real(1e-8, 10, prior='log-uniform'),
        'clf__estimator__reg_lambda': Real(1e-8, 10, prior='log-uniform')
    }
    lgb_base = LGBMClassifier(objective='binary', random_state=0)
    pipe = Pipeline([
        ('preproc', dh.pipeline),
        ('clf', OrdinalClassifier(estimator=lgb_base, n_jobs=-1))
    ])
    bayes = BayesSearchCV(
        pipe,
        search_space,
        n_iter=50,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=0,
        verbose=1
    )
    bayes.fit(X_raw, y)
    print("\nBest LightGBM parameters:")
    for param in ['n_estimators', 'num_leaves', 'learning_rate', 'max_depth', 'min_child_samples', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
        key = f"clf__estimator__{param}"
        print(f"  {param}: {bayes.best_params_[key]}")
    print(f"Best CV Score: {bayes.best_score_:.4f} (+/- {bayes.cv_results_['std_test_score'][bayes.best_index_]:.4f})")
    pipe_final = bayes.best_estimator_
    cv_scores = cross_val_score(pipe_final, X_raw, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    pipe_final.fit(X_raw, y)
    y_train_pred = pipe_final.predict(X_raw)
    y_test_pred = pipe_final.predict(X_test_raw)
    train_acc = accuracy_score(y, y_train_pred)
    print(f"Training accuracy: {train_acc:.4f}")
    class_labels = ['Low', 'Medium', 'High']
    submission = pd.DataFrame({'salary_category': [class_labels[p] for p in y_test_pred]}, index=test_idx)
    submission.index.name = 'id'
    submission.to_csv('submission_lightgbm.csv', index=True)
    print("Saved LightGBM submission to 'submission_lightgbm.csv'")
    print("\nLightGBM Training Performance:")
    cm = confusion_matrix(y, y_train_pred)
    print(classification_report(y, y_train_pred, target_names=class_labels))
    plot_confusion_matrix(cm, class_labels, "LightGBM Training CM")
    analyze_final_clusters(pipe_final, X_raw, y)


def main():
    print("Loading data with 3 job description clusters...")
    dh = DataHandler(n_job_clusters=3, use_cluster_probabilities=True)
    X_raw, y_df, train_idx = dh.get_train_data_raw()
    X_test_raw, test_idx = dh.get_test_data_raw()
    y = y_df["salary_category"]
    print(f"Training data shape: {X_raw.shape}")
    print(f"Test data shape: {X_test_raw.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    # Run LightGBM pipeline
    run_lightgbm_pipeline(X_raw, y, X_test_raw, test_idx)
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()