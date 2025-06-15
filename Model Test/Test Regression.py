import pandas as pd
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from Classes.DataHandler import DataHandler
from Classes.NNModel import NNModel

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class CustomTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        output_size=2,
        num_layers=2,
        dropout=0.3,
        lr=1e-3,
        batch_size=32,
        max_epochs=50,
        n_splits=5,
        random_state=None,
        weight_decay=1e-5,
        device=None,
        ordered_classification=False,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_splits = n_splits
        self.random_state = random_state
        self.weight_decay = weight_decay
        self.device = "cpu"
        self.ordered_classification=ordered_classification


        self.model = NNModel(
            input_size = input_size,
            hidden_size = hidden_size,
            output_size = output_size,
            num_layers = num_layers,
            dropout = dropout,
            ordered_classification = True,
        )

    def _train_one_fold(self, X_train, y_train):
        # prepare data
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.int64).to(self.device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # set up optimizer/loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()

        # training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            for xb, yb in loader:
                # N, 3
                yb = F.one_hot(yb.squeeze(), num_classes=3)
                # N, 3 @ 3, 2 -> N, 2
                yb = yb @ torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int64)
                yb = yb.to(torch.float32)

                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

    def fit(self, X, y):
        # simple KFold training (no stratification)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, _ in kf.split(X):
            self._train_one_fold(X[train_idx], y[train_idx])
        return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            col1 = torch.where(logits[:, 0] > 0.5, 1, 0)
            col2 = torch.where(logits[:, 1] > 0.5, 1, 0)
            col2 = col1 * col2
            preds = col1 + col2
        return preds.cpu().numpy()

def train_nn_model_ordinal_classification(X, Y):
    pipe = Pipeline([
        ("net", CustomTorchClassifier(
        input_size=X.shape[1],
        hidden_size=64,
        output_size=2,
        num_layers=2,
        dropout=0.3,
        lr=3e-4,
        batch_size=32,
        max_epochs=1,
        n_splits=5,
        random_state=42,
    )),
    ])


    param_search = {
        "net__lr": Real(1e-5, 1e-3, prior='log-uniform'),
        "net__max_epochs": Integer(1, 300),
        "net__hidden_size": Categorical([4, 8, 16, 32, 64, 128, 256]),
        "net__num_layers": Integer(1, 5),
        "net__dropout": Categorical([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "net__weight_decay": Real(1e-4, 1e-1, prior='log-uniform'),
        "net__batch_size": Categorical([16, 32, 64, 128, 256]),
    }

    opt = BayesSearchCV(
        pipe,
        param_search,
        n_iter=10,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=3,
    )

    # make sure Y is int64 so skorch produces LongTensor targets
    X_np = X.to_numpy()
    Y_np = Y.to_numpy()

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

def main():
    dh = DataHandler(ordinal_classification=False)
    X, Y = dh.get_train_data()
    train_nn_model_ordinal_classification(X=X, Y=Y)


if __name__ == '__main__':
    main()