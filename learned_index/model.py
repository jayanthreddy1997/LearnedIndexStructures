import torch.nn as nn
from collections import OrderedDict


class NeuralNetwork(nn.Module):
    def __init__(self, n_layers=10, n_units=2):
        super().__init__()
        stack = OrderedDict()
        stack["h0"] = nn.Linear(1, n_units)
        stack["a0"] = nn.ReLU()
        for i in range(1, n_layers):
            stack[f"h{i}"] = nn.Linear(n_units, n_units)
            stack[f"a{i}"] = nn.ReLU()
        stack[f"h{n_layers}"] = nn.Linear(n_units, 1)
        self.nn_stack = nn.Sequential(stack)

    def forward(self, x):
        return self.nn_stack(x)
