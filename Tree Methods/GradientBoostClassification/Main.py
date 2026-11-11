# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Main driver for Gradient Boosting classifier comparison.
Compares custom Gradient Boosting implementation against sklearn's
GradientBoostingClassifier.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from GradientBoost import GradientBoostingClassifier as GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoosting
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits, load_wine

#%% Main
if __name__ == "__main__":
    
    #%% Gradient Boosting Custom Model
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
    # print(df.head())
    # print(df['target_name'].value_counts())
    X, Y = loaded_data.data, loaded_data.target.reshape(-1, 1)   
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    
    #%% Custom Gradient Boosting
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samp_split=3,
        random_state=42
    )
    # Train model
    gb_classifier.fit(X_train, Y_train)

    # Test model
    Y_pred_custom = gb_classifier.predict(X_test)
    score_custom = accuracy_score(Y_test, Y_pred_custom)
    print(f'Custom Gradient Boosting Test Score: {score_custom:.4f}')
    
    # Show class probabilities for first 5 test samples
    probs = gb_classifier.predict_proba(X_test[:5])
    # print(f'\nClass probabilities for first 5 test samples:')
    # for i, p in enumerate(probs):
    #     print(f'  Sample {i}: {[round(x, 4) for x in p]}')
    
    #%% Sklearn Gradient Boosting for comparison
    sklearn_gb = SklearnGradientBoosting(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_split=3,
        random_state=42
    )
    sklearn_gb.fit(X_train, Y_train.ravel())
    
    Y_pred_sklearn = sklearn_gb.predict(X_test)
    score_sklearn = accuracy_score(Y_test, Y_pred_sklearn)
    print(f'Sklearn Gradient Boosting Test Score: {score_sklearn:.4f}')
    
    pass