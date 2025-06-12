import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skorch import NeuralNetClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from Classes.NNModel import NNModel
from Classes.OrdinalClassifier import OrdinalClassifier
# Import the DataHandler from the second file
from Classes.DataHandler import DataHandler  # Adjust import path as needed


class DataFrameToTensor(BaseEstimator, TransformerMixin):
    """Convert pandas DataFrame to numpy array and ensure float32."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert DataFrame to numpy array if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        # Ensure float32 for PyTorch
        return X_array.astype(np.float32)


def create_nn_pipeline(preprocessing_pipeline, input_size: int, base_params=None):
    """Create pipeline combining DataHandler preprocessing with neural network."""

    # Default base_params for NeuralNetClassifier
    if base_params is None:
        base_params = {
            'module': NNModel,
            'module__input_size': input_size,
            'module__output_size': 2,  # For ordinal with 3 classes
            'module__hidden_size': 32,
            'module__num_layers': 2,
            'module__dropout': 0.5,
            'max_epochs': 1,
            'lr': 1e-3,
            'optimizer__weight_decay': 1e-4,
            'batch_size': 32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'iterator_train__shuffle': True,
            'criterion': torch.nn.CrossEntropyLoss,
            'verbose': 0,
        }

    base_estimator = NeuralNetClassifier(**base_params)

    # Combine DataHandler preprocessing with DataFrame->Tensor conversion and NN
    pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('to_tensor', DataFrameToTensor()),
        ('classifier', OrdinalClassifier(estimator=base_estimator, n_jobs=1))
    ])

    return pipeline


def train_ordinal_NN_with_hyperopt(data_handler, n_iter=20, cv_folds=5):
    """
    Train neural network with hyperparameter optimization using DataHandler.

    Args:
        data_handler: Instance of DataHandler class
        n_iter: Number of iterations for Bayesian optimization
        cv_folds: Number of cross-validation folds
    """

    # Get raw training data
    X_train_raw, y_train, _ = data_handler.get_train_data_raw()

    # Clone and fit the preprocessing pipeline on ALL training data first
    # This ensures consistent feature dimensions across CV folds
    from sklearn.base import clone
    preprocessing_pipeline = clone(data_handler.pipeline)
    X_preprocessed = preprocessing_pipeline.fit_transform(X_train_raw)
    input_size = X_preprocessed.shape[1]

    print(f"Input size after preprocessing: {input_size}")
    print(f"Feature names: {list(X_preprocessed.columns)[:10]}...")  # Show first 10 features

    # Transform the data once for all CV (this ensures consistent dimensions)
    X_train_transformed = X_preprocessed.values.astype(np.float32)

    # Create a simpler pipeline for CV that doesn't include preprocessing
    # (since we've already preprocessed the data)
    base_params = {
        'module': NNModel,
        'module__input_size': input_size,
        'module__output_size': 2,  # For ordinal with 3 classes
        'module__hidden_size': 32,
        'module__num_layers': 2,
        'module__dropout': 0.5,
        'max_epochs': 1,
        'lr': 1e-3,
        'optimizer__weight_decay': 1e-4,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'iterator_train__shuffle': True,
        'criterion': torch.nn.CrossEntropyLoss,
        'verbose': 0,
    }

    base_estimator = NeuralNetClassifier(**base_params)
    ordinal_classifier = OrdinalClassifier(estimator=base_estimator, n_jobs=1)

    # Define search space for the neural network hyperparameters
    search_space = {
        "estimator__lr": Real(1e-5, 1e-2, prior='log-uniform'),
        "estimator__max_epochs": Integer(3, 300),
        "estimator__module__hidden_size": Categorical([8, 16, 32, 64, 128]),
        "estimator__module__num_layers": Integer(1, 5),
        "estimator__module__dropout": Real(0.1, 0.7),
        "estimator__optimizer__weight_decay": Real(1e-4, 1e-1, prior='log-uniform'),
        "estimator__batch_size": Categorical([16, 32, 64, 128]),
    }

    # Get target as 1D array
    y_train_1d = y_train.values.ravel()

    # StratifiedKFold on the 1D label array
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)

    bayes_search = BayesSearchCV(
        ordinal_classifier,
        search_space,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
        verbose=2
    )

    # Fit using preprocessed data
    bayes_search.fit(X_train_transformed, y_train_1d)

    print(f"\nBest CV accuracy: {bayes_search.best_score_:.4f}")
    print("\nBest parameters:")
    for param, value in bayes_search.best_params_.items():
        print(f"  {param}: {value}")

    # Create final pipeline with best parameters for predictions
    best_nn_params = bayes_search.best_params_
    # Update base_params with best found parameters
    for param, value in best_nn_params.items():
        param_name = param.replace("estimator__", "")
        if param_name in base_params:
            base_params[param_name] = value
        elif param_name.startswith("module__"):
            base_params[param_name] = value
        elif param_name.startswith("optimizer__"):
            base_params[param_name] = value

    # Create final pipeline with preprocessing for predictions
    final_pipeline = create_nn_pipeline(preprocessing_pipeline, input_size, base_params)
    final_pipeline.fit(X_train_raw, y_train_1d)

    return final_pipeline, bayes_search.best_params_, bayes_search.best_score_


def make_predictions(best_pipeline, data_handler):
    """Make predictions on test data using the best pipeline."""

    X_test_raw, test_index = data_handler.get_test_data_raw()

    # Make predictions (returns ordinal encoded values 0, 1, 2)
    predictions = best_pipeline.predict(X_test_raw)

    # Convert back to original labels
    label_mapping = {0: "Low", 1: "Medium", 2: "High"}
    predictions_labeled = [label_mapping[pred] for pred in predictions]

    # Create submission DataFrame
    import pandas as pd
    submission = pd.DataFrame({
        'obs': test_index,
        'salary_category': predictions_labeled
    })

    return submission


if __name__ == "__main__":
    # Initialize DataHandler
    print("Initializing DataHandler...")
    data_handler = DataHandler()

    # Train model with hyperparameter optimization
    print("\nTraining neural network with hyperparameter optimization...")
    best_pipeline, best_params, cv_score = train_ordinal_NN_with_hyperopt(
        data_handler,
        n_iter=20,  # Reduced for testing, increase for better results
        cv_folds=5
    )

    # Make predictions on test set
    print("\nMaking predictions on test set...")
    submission = make_predictions(best_pipeline, data_handler)

    # Save predictions
    submission.to_csv("submission.csv", index=False)
    print(f"\nPredictions saved to submission.csv")
    print(f"First few predictions:\n{submission.head()}")