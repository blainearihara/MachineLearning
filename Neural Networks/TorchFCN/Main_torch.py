# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 2026

Main driver for the mytorch (PyTorch-style) MLP.

Two things are demonstrated here:

  1) CONFIDENCE TEST. Build the SAME architecture as the earlier sklearn MLP
     (one hidden layer of 100 units, ReLU, no dropout / no batchnorm), train it
     with Adam + cross-entropy through the autograd engine, and confirm it
     reaches accuracy comparable to scikit-learn's MLPClassifier on the same
     data. This verifies the engine end-to-end on a real training run, with the
     extra operations (dropout / batchnorm) switched off.

  2) MODULAR EXTENSION. Build a second model that DOES use BatchNorm1d and
     Dropout — exactly the operations sklearn's MLPClassifier cannot express —
     by dropping those modules into the same nn.Sequential, and train it the
     same way. Because the framework was verified in step 1, adding these
     layers is just adding modular operations to a trusted engine.

The model class body is identical to PyTorch; only the import lines differ.

@author: blain
"""

#%% Libraries
import numpy as np

import mytorch as torch
import mytorch.nn as nn
import mytorch.optim as optim

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier as SklearnMLP


#%% Model definition (PyTorch-style)
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=(100,),
                 use_batchnorm=False, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#%% Training / evaluation helpers
def train_model(model, X_train, y_train, epochs=300, lr=1e-3,
                batch_size=200, seed=0, verbose=False):
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = X_train.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.tensor(X_train[idx])
            yb = y_train[idx]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1:3d}  loss {epoch_loss / n:.4f}")
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
    preds = logits.data.argmax(axis=1)
    return accuracy_score(y, preds)


#%% Main
if __name__ == "__main__":
    # Load and standardize data (neural nets are scale-sensitive)
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    print(f"wine: n_train={X_train.shape[0]}, n_features={input_dim}, "
          f"n_classes={num_classes}\n")

    # ---- 1) Confidence test: same architecture as the sklearn MLP ----
    print("=" * 60)
    print("1) mytorch MLP (100, ReLU) vs sklearn MLPClassifier")
    print("=" * 60)
    custom = MLP(input_dim, num_classes, hidden=(100,))
    train_model(custom, X_train, y_train, epochs=300, lr=1e-3, seed=0)
    acc_custom = evaluate(custom, X_test, y_test)

    skl = SklearnMLP(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                     alpha=0.0, learning_rate_init=1e-3, max_iter=300,
                     random_state=42)
    skl.fit(X_train, y_train)
    acc_skl = accuracy_score(y_test, skl.predict(X_test))

    print(f"  mytorch  test accuracy : {acc_custom:.4f}")
    print(f"  sklearn  test accuracy : {acc_skl:.4f}\n")

    # ---- 2) Modular extension: add BatchNorm1d + Dropout ----
    print("=" * 60)
    print("2) mytorch MLP (100, BatchNorm1d, ReLU, Dropout(0.3))")
    print("   -- operations sklearn's MLPClassifier cannot express --")
    print("=" * 60)
    extended = MLP(input_dim, num_classes, hidden=(100,),
                   use_batchnorm=True, dropout=0.3)
    train_model(extended, X_train, y_train, epochs=300, lr=1e-3, seed=0)
    acc_extended = evaluate(extended, X_test, y_test)
    print(f"  mytorch+BN+Dropout test accuracy : {acc_extended:.4f}")
