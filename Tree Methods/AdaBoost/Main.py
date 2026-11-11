# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 2026

Main driver for AdaBoost classifier comparison.
Compares custom AdaBoost implementation against sklearn's AdaBoostClassifier.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from AdaBoost import AdaBoostClassifier as AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits, load_wine

#%% Main
if __name__ == "__main__":
    
    #%% AdaBoost Custom Model
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
    
    #%% Custom AdaBoost
    ada_classifier = AdaBoostClassifier(
        n_estimators=50,
        max_depth=1,
        learning_rate=0.25,
        random_state=42
    )
    # Train model
    ada_classifier.fit(X_train, Y_train)

    # Test model
    Y_pred_custom = ada_classifier.predict(X_test)
    score_custom = accuracy_score(Y_test, Y_pred_custom)
    print(f'Custom AdaBoost Test Score: {score_custom:.4f}')
    
    # Show estimator weights (alpha values)
    alphas = ada_classifier.get_estimator_weights()
    print(f'Number of estimators used: {len(alphas)}')
    print(f'Alpha range: [{min(alphas):.4f}, {max(alphas):.4f}]')
    
    #%% Sklearn AdaBoost for comparison
    sklearn_ada = SklearnAdaBoost(
        estimator=SklearnDecisionTree(max_depth=1),
        n_estimators=50,
        learning_rate=0.25,
        random_state=42
    )
    sklearn_ada.fit(X_train, Y_train.ravel())
    
    Y_pred_sklearn = sklearn_ada.predict(X_test)
    score_sklearn = accuracy_score(Y_test, Y_pred_sklearn)
    print(f'Sklearn AdaBoost Test Score: {score_sklearn:.4f}')
    
    pass