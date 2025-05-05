import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skopt import BayesSearchCV
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skopt.space import Real, Integer, Categorical
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from DataHandler import DataHandler
from NNModel import NNModel
from SubmissionGenerator import SubmissionGenerator


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
    dataHandler = DataHandler()
    X, Y = dataHandler.get_train_data()

    pipe = Pipeline([
        ("net", NeuralNetClassifier(
            module=NNModel,
            module__input_size=X.shape[1],
            module__hidden_size=64,
            module__output_size=3,
            module__num_layers=2,  # match your manual setting
            module__dropout=0.3,  # as in manual
            max_epochs=200,  # give it enough epochs
            lr=3e-4,
            optimizer=torch.optim.Adam,
            batch_size=32,
            verbose=0,
            criterion=torch.nn.CrossEntropyLoss,
        )),
    ])

    param_search = {
        "net__lr": Real(1e-5, 1e-3, prior='log-uniform'),
        "net__max_epochs": Integer(5, 300),
        "net__module__hidden_size": Categorical([4, 8, 16, 32, 64, 128]),
        "net__module__num_layers": Integer(1, 5),
        "net__module__dropout": Real(0.1, 0.7),
        "net__optimizer__weight_decay": Real(1e-4, 1e-1, prior='log-uniform'),
        "net__batch_size": Categorical([16, 32, 64, 128]),
    }

    opt = BayesSearchCV(
        pipe,
        param_search,
        n_iter=10,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=3,
    )

    # make sure Y is int64 so skorch produces LongTensor targets
    X_np = X.to_numpy().astype(np.float32)
    Y_np = Y.to_numpy().astype(np.int64)

    opt.fit(X_np, Y_np)

    # 1) Best parameters found
    print("Best parameters:")
    for param, val in opt.best_params_.items():
        print(f"  {param}: {val}")

    # 2) Best cross-validated accuracy
    print(f"\nBest CV accuracy: {opt.best_score_:.4f}")

    # 3) Number of parameter settings evaluated
    print(f"\nTotal parameter settings evaluated: {len(opt.cv_results_['params'])}")

    # 4) Show the top 5 candidates by mean test accuracy
    results = pd.DataFrame(opt.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    top5 = results.head(5)[[
        "mean_test_score", "std_test_score", "params"
    ]].reset_index(drop=True)

    print("\nTop 5 parameter settings:")
    for i, row in top5.iterrows():
        print(f"\nRank {i + 1}:")
        print(f"  mean_test_accuracy: {row.mean_test_score:.4f}")
        print(f"  std_test_accuracy:  {row.std_test_score:.4f}")
        print(f"  params:            {row.params}")


def generate_nn_submission():
    # 1) Train your model & get pipeline + label encoder:
    dh = DataHandler(fill_data=True, train_data=True)
    X_train, Y_train = dh.X.to_pandas().values, dh.Y
    le = LabelEncoder().fit(Y_train)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", NeuralNetClassifier(
            module=NNModel,
            module__input_size=X_train.shape[1],
            module__hidden_size=128,
            module__output_size=3,
            module__num_layers=2,
            module__dropout=0.5,
            max_epochs=20,
            lr=6e-4,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=0.001,
            batch_size=64,
            train_split=None,
            verbose=0,
            criterion=torch.nn.CrossEntropyLoss,
        ))
    ])
    pipe.fit(X_train.astype(np.float32), le.transform(Y_train))

    label_encoder = dh.le
    dh = DataHandler(fill_data=True, train_data=False)
    X_test, index = dh.X.to_pandas().values, dh.indexes

    sub = SubmissionGenerator(pipe, label_encoder=label_encoder)
    sub.generate(X_test, index)


def nn_t_t(
        test_size: float = 0.2,
        random_state: int = 0,
        batch_size: int = 32
) -> nn.Module:
    # 1. Load data
    dataHandler = DataHandler(fill_data=True)
    X, Y = dataHandler.X.to_numpy(), dataHandler.Y

    # 2. Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Train/eval split
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state,
        stratify=Y
    )

    # 4. Convert to PyTorch tensors and create DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 5. Initialize model, loss, optimizer
    model = NNModel(
        input_size=X_train.shape[1],
        hidden_size=64,
        output_size=len(np.unique(Y)),
        num_layers=2,
        dropout=0.3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # 6. Training loop with batching
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / len(train_ds)

        if epoch % 10 == 0 or epoch == 1:
            # Evaluate
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    val_loss += loss.item() * xb.size(0)
                    pred = outputs.argmax(dim=1)
                    correct += (pred == yb).sum().item()
            avg_val_loss = val_loss / len(val_ds)
            val_acc = correct / len(val_ds)

            print(
                f"Epoch {epoch:4d}/{num_epochs} "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f} "
                f"Val Acc: {val_acc:.4f}"
            )

    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nn_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
