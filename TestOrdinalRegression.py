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
from xgboostTest import OrdinalClassifier


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
        # Navigate the nested pipeline to find the fitted cluster transformer and its KMeans model
        cluster_transformer = final_pipeline.named_steps['preproc'] \
                                            .named_steps['preprocessor'] \
                                            .named_transformers_['jobdesc_cluster']
        kmeans_model = cluster_transformer.kmeans
        n_clusters = kmeans_model.n_clusters
    except (KeyError, AttributeError):
        print("Could not find a fitted cluster transformer in the final pipeline.")
        print("Skipping cluster analysis.")
        return

    # To get the cluster labels, we must pass the original data through the
    # same preprocessing steps used by the clustering model (imputation and scaling).
    feature_extractor = FeatureExtractor() # Assuming this is available or imported
    job_desc_cols = feature_extractor.job_desc_cols
    X_job_desc = X[job_desc_cols].copy()
    X_job_desc.replace(0, np.nan, inplace=True)

    # Use the fitted imputer and scaler from the cluster transformer
    X_imputed = cluster_transformer.imputer.transform(X_job_desc)
    X_scaled = cluster_transformer.scaler.transform(X_imputed)
    cluster_labels = kmeans_model.predict(X_scaled)

    # Analyze salary distribution per cluster
    print(f"Analysis of the {n_clusters} clusters found by the optimization:")
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


# 1) Load raw data
print("Loading data...")
# Initialize with a base number of clusters. BayesSearchCV will optimize this.
dh = DataHandler(n_job_clusters=5, use_cluster_probabilities=True)
X_raw, y_df, train_idx = dh.get_train_data_raw()
X_test_raw, test_idx = dh.get_test_data_raw()
y = y_df["salary_category"]

print(f"Training data shape: {X_raw.shape}")
print(f"Test data shape: {X_test_raw.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# 2) Define CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# 3) Define search space for the model, including the number of clusters
search_space = {
    # NEW: Search for the ideal number of job description clusters
    'preproc__preprocessor__jobdesc_cluster__n_clusters': Integer(3, 50),

    # Original search space for the classifier
    'poly__degree': Categorical([1, 2]),
    'poly__interaction_only': [True, False],
    'clf__estimator__C': Real(1e-4, 1e2, prior='log-uniform')
}


print("\n" + "="*70)
print("BAYESIAN OPTIMIZATION FOR CLUSTERS AND ORDINAL CLASSIFIER")
print("="*70)

# 4) Create pipeline with ordinal classifier
pipe_ordinal = Pipeline([
    ("preproc", dh.pipeline),
    ("poly", PolynomialFeatures(include_bias=False)),
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

# 5) Bayesian optimization
print("\nOptimizing hyperparameters (including n_clusters)...")

bayes_search = BayesSearchCV(
    pipe_ordinal,
    search_space, # Use the new combined search space
    n_iter=30,    # Increased iterations for the larger search space
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    random_state=0,
    verbose=2
)

# Fit and find best parameters
bayes_search.fit(X_raw, y)

print(f"\nBest parameters:")
# MODIFIED: Print the best number of clusters found
print(f"  Optimal number of clusters: {bayes_search.best_params_['preproc__preprocessor__jobdesc_cluster__n_clusters']}")
print(f"  Polynomial degree: {bayes_search.best_params_['poly__degree']}")
print(f"  Interaction only: {bayes_search.best_params_['poly__interaction_only']}")
print(f"  C parameter: {bayes_search.best_params_['clf__estimator__C']:.6f}")
print(f"Best CV Score: {bayes_search.best_score_:.4f} (+/- {bayes_search.cv_results_['std_test_score'][bayes_search.best_index_]:.4f})")

# 6) Get final optimized pipeline from search results
print("\n" + "="*70)
print("FINAL MODEL TRAINING")
print("="*70)

# The best_estimator_ is the pipeline already configured with the best params
pipe_final = bayes_search.best_estimator_
print("Final pipeline configured using the best parameters found by Bayesian Search.")

# 7) Final cross-validation with optimized parameters
print("\nRunning final cross-validation with the best pipeline...")
cv_scores_final = cross_val_score(pipe_final, X_raw, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"Cross-validation scores: {cv_scores_final}")
print(f"Mean CV accuracy: {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std():.4f})")

# Check for overfitting warning
if bayes_search.best_params_['poly__degree'] > 1:
    print("\n⚠️  Warning: Using polynomial features may increase overfitting risk")
    print("   Monitor the gap between training and CV accuracy carefully")

# 8) Train final model on full training data
# Note: The pipe_final from bayes_search is already fitted on the full data.
# Re-fitting is redundant but harmless.
print("\nTraining final model on full training data...")
pipe_final.fit(X_raw, y)

# 9) Generate predictions
print("\nGenerating predictions...")
y_train_pred = pipe_final.predict(X_raw)
train_accuracy = accuracy_score(y, y_train_pred)
y_test_pred = pipe_final.predict(X_test_raw)

# 10) Display results
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results_df = pd.DataFrame({
    "Metric": [
        "Optimal number of clusters",
        "Polynomial degree",
        "Interaction terms only",
        "Best C parameter",
        "Cross-validation accuracy",
        "Training accuracy"
    ],
    "Value": [
        f"{bayes_search.best_params_['preproc__preprocessor__jobdesc_cluster__n_clusters']}",
        f"{bayes_search.best_params_['poly__degree']}",
        f"{bayes_search.best_params_['poly__interaction_only']}",
        f"{bayes_search.best_params_['clf__estimator__C']:.6f}",
        f"{cv_scores_final.mean():.4f} ± {cv_scores_final.std():.4f}",
        f"{train_accuracy:.4f}"
    ]
})

print(results_df.to_string(index=False))

# Show feature expansion info
if bayes_search.best_params_['poly__degree'] > 1:
    n_features_after_clustering = pipe_final.named_steps['preproc'].transform(X_raw.head(1)).shape[1]
    n_features_after_poly = pipe_final.named_steps['poly'].transform(
        pipe_final.named_steps['preproc'].transform(X_raw.head(1))
    ).shape[1]
    print(f"\nFeature expansion:")
    print(f"  After clustering: {n_features_after_clustering} features")
    print(f"  After polynomial: {n_features_after_poly} features")
    print(f"  Expansion factor: {n_features_after_poly/n_features_after_clustering:.1f}x")

# 11) Confusion Matrix and Classification Report
print("\n" + "="*70)
print("TRAINING PERFORMANCE ANALYSIS")
print("="*70)

class_labels = ['Low', 'Medium', 'High']
cm_train = confusion_matrix(y, y_train_pred)

print("\nConfusion Matrix (Training Data):")
print("Predicted:", "   ".join([f"{label:>6}" for label in class_labels]))
for i, (actual_label, row) in enumerate(zip(class_labels, cm_train)):
    print(f"Actual {actual_label:>6}: {' '.join([f'{val:6d}' for val in row])}")

print("\nClassification Report:")
print(classification_report(y, y_train_pred, target_names=class_labels))
plot_confusion_matrix(cm_train, class_labels, "Training Data Confusion Matrix")

# 12) Test predictions summary
print("\n" + "="*70)
print("TEST DATA PREDICTIONS")
print("="*70)

test_pred_counts = pd.Series(y_test_pred).value_counts().sort_index()
test_summary = pd.DataFrame({
    "Class": class_labels,
    "Predictions": [test_pred_counts.get(i, 0) for i in range(3)],
    "Percentage": [f"{test_pred_counts.get(i, 0)/len(y_test_pred)*100:.1f}%" for i in range(3)]
})
print(test_summary.to_string(index=False))

# 13) Save test predictions
test_predictions_df = pd.DataFrame({
    'id': test_idx,
    'salary_prediction': [class_labels[pred] for pred in y_test_pred],
    'prediction_numeric': y_test_pred
})
test_predictions_df.to_csv('ordinal_predictions_optimized_clusters.csv', index=False)
print(f"\nTest predictions saved to 'ordinal_predictions_optimized_clusters.csv'")

# 14) Feature importance analysis using the final fitted model
# NEW: Call the standalone analysis function with the final pipeline
analyze_final_clusters(pipe_final, X_raw, y)

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)