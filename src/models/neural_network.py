"""
Neural network models for salary prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetworkModel(nn.Module):
    """
    PyTorch neural network model for classification.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of units in hidden layers
    output_size : int
        Number of output classes
    num_layers : int
        Number of hidden layers
    dropout : float
        Dropout rate
    """

    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=2, dropout=0.5):
        super().__init__()

        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OrdinalNeuralNetwork(nn.Module):
    """
    Neural network for ordinal classification.
    Uses special output layer for ordinal predictions.
    """

    def __init__(self, input_size, hidden_size, n_classes,
                 num_layers=2, dropout=0.5):
        super().__init__()

        # For ordinal classification with k classes, we need k-1 outputs
        output_size = n_classes - 1

        layers = []

        # Hidden layers
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer with sigmoid for ordinal probabilities
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, x):
        # Get ordinal probabilities
        ordinal_probs = self.net(x)

        # Convert to class probabilities
        # P(y=0) = 1 - P(y>0)
        # P(y=k) = P(y>k-1) - P(y>k) for 0 < k < n_classes-1
        # P(y=n_classes-1) = P(y>n_classes-2)

        batch_size = x.size(0)
        class_probs = torch.zeros(batch_size, self.n_classes)

        # First class
        class_probs[:, 0] = 1 - ordinal_probs[:, 0]

        # Middle classes
        for i in range(1, self.n_classes - 1):
            class_probs[:, i] = ordinal_probs[:, i - 1] - ordinal_probs[:, i]

        # Last class
        class_probs[:, -1] = ordinal_probs[:, -1]

        return class_probs


class TorchClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for PyTorch models.

    Parameters
    ----------
    model_class : class
        PyTorch model class to instantiate
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden units
    output_size : int
        Number of output classes
    num_layers : int
        Number of hidden layers
    dropout : float
        Dropout rate
    lr : float
        Learning rate
    batch_size : int
        Batch size for training
    max_epochs : int
        Maximum number of training epochs
    weight_decay : float
        L2 regularization strength
    device : str
        Device to use ('cpu' or 'cuda')
    ordinal : bool
        Whether to use ordinal classification
    """

    def __init__(self, model_class=NeuralNetworkModel, input_size=None,
                 hidden_size=64, output_size=3, num_layers=2,
                 dropout=0.3, lr=1e-3, batch_size=32, max_epochs=50,
                 weight_decay=1e-5, device=None, ordinal=False):
        self.model_class = model_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ordinal = ordinal
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Fit the neural network."""
        # Convert to numpy if needed
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y

        # Set input size if not provided
        if self.input_size is None:
            self.input_size = X_array.shape[1]

        # Get unique classes
        self.classes_ = np.unique(y_array)
        n_classes = len(self.classes_)

        # Initialize model
        if self.ordinal:
            self.model_ = OrdinalNeuralNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_classes=n_classes,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
        else:
            self.model_ = self.model_class(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=n_classes,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

        # Prepare data
        X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_array, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Set up optimizer and loss
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.ordinal:
            # For ordinal, we need a special loss
            criterion = OrdinalLoss(n_classes)
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model_.train()
        for epoch in range(self.max_epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self

    def predict(self, X):
        """Make predictions."""
        self.model_.eval()

        X_array = X.values if hasattr(X, 'values') else X
        X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model_(X_tensor)
            if self.ordinal:
                # For ordinal, outputs are already class probabilities
                predictions = torch.argmax(outputs, dim=1)
            else:
                predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Predict class probabilities."""
        self.model_.eval()

        X_array = X.values if hasattr(X, 'values') else X
        X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model_(X_tensor)
            if self.ordinal:
                # Already probabilities
                probs = outputs
            else:
                probs = F.softmax(outputs, dim=1)

        return probs.cpu().numpy()


class OrdinalLoss(nn.Module):
    """
    Custom loss function for ordinal classification.
    Combines binary cross-entropy losses for each threshold.
    """

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.bce = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        predictions: (batch_size, n_classes-1) ordinal probabilities
        targets: (batch_size,) class labels
        """
        batch_size = targets.size(0)
        device = predictions.device

        # Convert targets to ordinal encoding
        # For each threshold k: 1 if target > k, 0 otherwise
        ordinal_targets = torch.zeros(batch_size, self.n_classes - 1).to(device)

        for i in range(self.n_classes - 1):
            ordinal_targets[:, i] = (targets > i).float()

        # Calculate BCE loss for each threshold
        loss = self.bce(predictions, ordinal_targets)

        return loss


class CustomOrdinalTorchClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom implementation of ordinal neural network classifier.
    Specifically designed for the competition's 3-class ordinal problem.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.3, lr=1e-3, batch_size=32, max_epochs=50,
                 n_splits=5, random_state=None, weight_decay=1e-5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_splits = n_splits
        self.random_state = random_state
        self.weight_decay = weight_decay
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        self.model = OrdinalNeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            n_classes=3,  # Fixed for Low/Medium/High
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

    def _train_one_fold(self, X_train, y_train):
        """Train on one fold of data."""
        # Prepare data
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Set up optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = nn.BCELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            for xb, yb in loader:
                # Convert labels to ordinal encoding
                # For 3 classes, we need 2 binary problems
                yb_ordinal = torch.zeros(yb.size(0), 2).to(self.device)
                yb_ordinal[:, 0] = (yb > 0).float()  # Is Medium or High?
                yb_ordinal[:, 1] = (yb > 1).float()  # Is High?

                optimizer.zero_grad()
                out = self.model.net(xb)  # Get ordinal probabilities directly
                loss = criterion(out, yb_ordinal)
                loss.backward()
                optimizer.step()

    def fit(self, X, y):
        """Fit using k-fold cross-validation for robustness."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, _ in kf.split(X):
            self._train_one_fold(X[train_idx], y[train_idx])

        return self

    def predict(self, X):
        """Make predictions using ordinal decoding."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Get ordinal probabilities
            ordinal_probs = self.model.net(X_t)

            # Decode to classes
            # Class 0 (Low): P(y>0) < 0.5
            # Class 1 (Medium): P(y>0) >= 0.5 and P(y>1) < 0.5
            # Class 2 (High): P(y>1) >= 0.5

            col1 = (ordinal_probs[:, 0] > 0.5).long()
            col2 = (ordinal_probs[:, 1] > 0.5).long()
            predictions = col1 + col2

        return predictions.cpu().numpy()