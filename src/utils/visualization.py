import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix",
                          figsize=(8, 6), cmap='Blues', save_path=None):
    """
    Plot a confusion matrix with nice formatting.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Class labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    save_path : str, optional
        Path to save the figure
    """
    if labels is None:
        labels = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print metrics
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Accuracy: {accuracy:.4f}")

    # Per-class metrics
    for i, label in enumerate(labels):
        if np.sum(cm[i, :]) > 0:
            precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
            recall = cm[i, i] / np.sum(cm[i, :])
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"{label}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")


def plot_feature_importance(feature_names, importances, top_n=20,
                            figsize=(10, 8), save_path=None):
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    feature_names : list
        Feature names
    importances : array-like
        Feature importances
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Select top features
    top_features = importance_df.head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {len(top_features)} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return importance_df


def plot_learning_curves(history, metrics=['loss', 'accuracy'],
                         figsize=(12, 4), save_path=None):
    """
    Plot training history for neural networks.

    Parameters
    ----------
    history : dict
        Training history with keys like 'train_loss', 'val_loss', etc.
    metrics : list
        Metrics to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot training metric
        train_key = f'train_{metric}'
        if train_key in history:
            ax.plot(history[train_key], label='Train', linewidth=2)

        # Plot validation metric
        val_key = f'val_{metric}'
        if val_key in history:
            ax.plot(history[val_key], label='Validation', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cv_scores(cv_scores, model_names=None, figsize=(10, 6), save_path=None):
    """
    Plot cross-validation scores for multiple models.

    Parameters
    ----------
    cv_scores : dict or array-like
        CV scores for each model
    model_names : list, optional
        Model names
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if isinstance(cv_scores, dict):
        data = []
        names = []
        for name, scores in cv_scores.items():
            data.extend(scores)
            names.extend([name] * len(scores))
        df = pd.DataFrame({'Model': names, 'Score': data})
    else:
        # Single model
        df = pd.DataFrame({'Model': 'Model', 'Score': cv_scores})

    plt.figure(figsize=figsize)

    # Box plot
    sns.boxplot(data=df, x='Model', y='Score')

    # Add mean markers
    means = df.groupby('Model')['Score'].mean()
    positions = range(len(means))
    plt.scatter(positions, means, color='red', s=100, zorder=5, label='Mean')

    plt.ylabel('CV Score')
    plt.title('Cross-Validation Scores by Model')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics
    for i, model in enumerate(df['Model'].unique()):
        scores = df[df['Model'] == model]['Score']
        mean_score = scores.mean()
        std_score = scores.std()
        plt.text(i, plt.ylim()[0] + 0.01, f'{mean_score:.3f}±{std_score:.3f}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_hyperparameter_importance(cv_results, param_names=None,
                                   figsize=(12, 8), save_path=None):
    """
    Plot hyperparameter importance from cross-validation results.

    Parameters
    ----------
    cv_results : dict
        CV results from GridSearchCV or BayesSearchCV
    param_names : list, optional
        Parameter names to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    results_df = pd.DataFrame(cv_results)

    if param_names is None:
        param_names = [col for col in results_df.columns if col.startswith('param_')]

    n_params = len(param_names)
    fig, axes = plt.subplots((n_params + 1) // 2, 2, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]

    for idx, param in enumerate(param_names):
        if idx < len(axes):
            ax = axes[idx]

            # Get parameter values and scores
            param_values = results_df[param]
            scores = results_df['mean_test_score']

            # Handle different parameter types
            try:
                # Numeric parameters
                param_numeric = pd.to_numeric(param_values)
                ax.scatter(param_numeric, scores, alpha=0.6)
                ax.set_xlabel(param.replace('param_', ''))
                ax.set_ylabel('CV Score')

                # Add trend line
                z = np.polyfit(param_numeric, scores, 1)
                p = np.poly1d(z)
                ax.plot(param_numeric, p(param_numeric), "r--", alpha=0.8)
            except:
                # Categorical parameters
                unique_values = param_values.unique()
                value_scores = [scores[param_values == val].mean() for val in unique_values]
                ax.bar(range(len(unique_values)), value_scores)
                ax.set_xticks(range(len(unique_values)))
                ax.set_xticklabels(unique_values, rotation=45)
                ax.set_xlabel(param.replace('param_', ''))
                ax.set_ylabel('Mean CV Score')

            ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Hyperparameter Impact on CV Score')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cluster_analysis(cluster_labels, y_true, cluster_names=None,
                          figsize=(10, 6), save_path=None):
    """
    Visualize cluster composition with respect to target variable.

    Parameters
    ----------
    cluster_labels : array-like
        Cluster assignments
    y_true : array-like
        True target values
    cluster_names : list, optional
        Names for clusters
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Create DataFrame
    df = pd.DataFrame({
        'cluster': cluster_labels,
        'target': y_true
    })

    # Count occurrences
    cluster_counts = df.groupby(['cluster', 'target']).size().unstack(fill_value=0)

    # Convert to percentages
    cluster_percentages = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100

    # Plot
    ax = cluster_percentages.plot(kind='bar', stacked=True, figsize=figsize,
                                  colormap='viridis')

    plt.xlabel('Cluster')
    plt.ylabel('Percentage')
    plt.title('Target Distribution by Cluster')
    plt.legend(title='Target Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add cluster sizes
    cluster_sizes = df['cluster'].value_counts().sort_index()
    for i, size in enumerate(cluster_sizes):
        plt.text(i, 101, f'n={size}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("Cluster Statistics:")
    print("-" * 50)
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]['target']
        print(f"\nCluster {cluster} (n={len(cluster_data)}):")
        print(cluster_data.value_counts().sort_index())


def plot_optimization_history(run_data, figsize=(12, 8), save_path=None):
    """
    Plot optimization history from RunTracker data.

    Parameters
    ----------
    run_data : dict
        Run data from RunTracker
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if 'iteration_history' not in run_data or not run_data['iteration_history']:
        print("No iteration history found")
        return

    # Extract data
    iterations = []
    scores = []
    stds = []

    for item in run_data['iteration_history']:
        iterations.append(item['iteration'])
        scores.append(item['score'])
        stds.append(item.get('std', 0))

    # Convert to arrays
    iterations = np.array(iterations)
    scores = np.array(scores)
    stds = np.array(stds)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Plot scores with error bars
    ax1.errorbar(iterations, scores, yerr=stds, fmt='o-', capsize=5,
                 alpha=0.7, label='CV Score')

    # Add best score line
    best_idx = np.argmax(scores)
    ax1.axhline(y=scores[best_idx], color='r', linestyle='--',
                label=f'Best: {scores[best_idx]:.4f}')
    ax1.scatter(iterations[best_idx], scores[best_idx], color='r', s=100, zorder=5)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('CV Score')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot running best
    running_best = np.maximum.accumulate(scores)
    ax2.plot(iterations, running_best, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Score')
    ax2.set_title('Running Best Score')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"Best iteration: {iterations[best_idx]}")
    print(f"Best score: {scores[best_idx]:.4f} ± {stds[best_idx]:.4f}")
    print(f"Total iterations: {len(iterations)}")
    print(f"Score improvement: {scores[best_idx] - scores[0]:.4f}")
    """
Visualization utilities for model analysis and results.
"""
