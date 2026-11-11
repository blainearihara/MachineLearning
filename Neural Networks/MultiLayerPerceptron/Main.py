# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Main driver for the MLP (fully-connected neural network) classifier comparison.
Compares the custom MLPClassifier implementation against scikit-learn's
sklearn.neural_network.MLPClassifier on a sklearn toy dataset.

Mirrors the structure of the tree-based Main.py: load a dataset, split it,
train the custom model, score it, and print the result side-by-side with the
sklearn equivalent trained on the same data with the same hyperparameters.

Note on scaling: unlike decision trees, neural networks are sensitive to input
feature scale, so the features are standardized (zero mean, unit variance) with
a StandardScaler fit on the training split and applied to both models. The
scaler is preprocessing external to the classifier, so both implementations are
still used in exactly the same way.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from MLPClassifier import MLPClassifier as MLPClassifier

from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits, load_wine

#%% Main
if __name__ == "__main__":

    #%% MLP Custom Model
    data_name = 'wine'
    # Load data from sklearn library
    if data_name == 'iris':
        loaded_data = load_iris()
    elif data_name == 'digit':
        loaded_data = load_digits()
    elif data_name == 'wine':
        loaded_data = load_wine()

    df = pd.DataFrame(loaded_data.data, columns=loaded_data.feature_names)
    df['target'] = loaded_data.target
    df['target_name'] = df['target'].map(dict(enumerate(loaded_data.target_names)))
    print(df.head())
    print(df['target_name'].value_counts())
    X, Y = loaded_data.data, loaded_data.target.reshape(-1, 1)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)

    # Standardize features (neural nets are sensitive to feature scale).
    # Fit on train only, then apply the same transform to test.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #%% Custom MLP
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42
    )
    # Train model
    mlp_classifier.fit(X_train, Y_train)

    # Test model
    Y_pred_custom = mlp_classifier.predict(X_test)
    score_custom = accuracy_score(Y_test.ravel(), Y_pred_custom)
    print(f'Custom MLP Test Score: {score_custom:.4f}')

    # Show class probabilities for the first 5 test samples
    probs = mlp_classifier.predict_proba(X_test[:5])
    print('\nClass probabilities for first 5 test samples:')
    for i, p in enumerate(probs):
        print(f'  Sample {i}: {[round(x, 4) for x in p]}')

    #%% Sklearn MLP for comparison
    sklearn_mlp = SklearnMLP(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42
    )
    sklearn_mlp.fit(X_train, Y_train.ravel())

    Y_pred_sklearn = sklearn_mlp.predict(X_test)
    score_sklearn = accuracy_score(Y_test.ravel(), Y_pred_sklearn)
    print(f'\nSklearn MLP Test Score: {score_sklearn:.4f}')

    pass
