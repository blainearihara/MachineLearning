# -*- coding: utf-8 -*-
"""
mytorch — a compact, faithful re-implementation of the slice of PyTorch needed
to build and train fully-connected networks with autograd.

Usage mirrors PyTorch almost verbatim:

    import mytorch as torch
    import mytorch.nn as nn
    import mytorch.optim as optim

    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )
        def forward(self, x):
            return self.network(x)

    model = MLP(input_dim=100, num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
"""

from .engine import (
    Tensor,
    tensor,
    no_grad,
    set_default_dtype,
    manual_seed,
    is_grad_enabled,
    float32,
    float64,
)

from . import nn
from . import optim

__all__ = [
    'Tensor', 'tensor', 'no_grad', 'set_default_dtype', 'manual_seed',
    'is_grad_enabled', 'float32', 'float64', 'nn', 'optim',
]
