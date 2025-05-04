import torch
import torch.nn as nn

class NNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        layers = []

        # 1) first hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # 2) additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # 3) output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # pack into Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)