import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

from DataHandler import DataHandler

class PolarsToPandasTransformer(TransformerMixin, BaseEstimator):
    """Transformer that converts a Polars DataFrame to a pandas DataFrame (numpy array)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X: pl.DataFrame):
        return X.to_pandas().values

def test():
    dh = DataHandler(1, 1, True)
    X_pl, Y = dh.X, dh.Y

    # Build a sklearn pipeline:
    pipe = Pipeline([
        # 1) bring Polars → numpy array
        ("polars_to_numpy", PolarsToPandasTransformer()),
        # 2) impute missing via KNN
        # ("imputer", KNNImputer()),
        # 3) scale
        ("scaler", StandardScaler()),
        # 4) reduce dims
        # ("pca", PCA()),
        # 5) classify
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5_000,
            random_state=0
        ))
    ])

    # Now we can tune the imputer’s n_neighbors & weights too:
    search_space = {
        # "imputer__n_neighbors": (10, 100),  # integer range
        # "imputer__weights": ["uniform", "distance"],
        # "pca__n_components": ([348]),
        # "clf__C": (1e-4, 1e1, "log-uniform"),
        "clf__C": ([0.012]),
    }

    bayes_cv = BayesSearchCV(
        pipe,
        search_space,
        n_iter=1,
        cv=10,
        scoring="accuracy",
        random_state=0,
        n_jobs=-1,
    )

    bayes_cv.fit(X_pl, Y)

    print("Best CV score:", bayes_cv.best_score_)
    print("Best params:", bayes_cv.best_params_)
    print("Total features:", X_pl.shape[1])

def main():
    dataHandler = DataHandler(fill_data=True)

    X = dataHandler.X
    Y = dataHandler.Y

    # 4. Build pipeline (note: PolynomialFeatures is passed as the class)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            random_state=0
        ))
    ])

    bayes_cv = BayesSearchCV(
        pipe,
        {
            'clf__C': (1e-4, 1e1, 'log-uniform'),
            'pca__n_components': (20, X.shape[1])
        },
        n_iter=30,
        cv=5,
        scoring='accuracy',
        random_state=0,
        n_jobs=8,
    )

    # 6. Fit on the NumPy array + encoded target
    bayes_cv.fit(X, Y)

    print("Best CV score:", bayes_cv.best_score_)
    print("Best params:", bayes_cv.best_params_)
    print("Number of features", X.shape[1])

def nn_test():


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
