# -*- coding: utf-8 -*-
"""
Created on Wed May 07 2026

Main driver for XGBoost classifier comparison.
Compares custom XGBoost implementation against the xgboost library's
XGBClassifier.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from XGBoost import XGBoostClassifier as XGBoostClassifier

from xgboost import XGBClassifier as LibXGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits, load_wine

#%% Main
if __name__ == "__main__":
    
    #%% XGBoost Custom Model
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
    
    #%% Custom XGBoost
    xgb_classifier = XGBoostClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        min_samp_split=2,
        reg_lambda=1.0,
        gamma=0.0,
        subsample=1.0,
        colsample_bylevel=1.0,
        random_state=42
    )
    # Train model
    xgb_classifier.fit(X_train, Y_train)

    # Test model
    Y_pred_custom = xgb_classifier.predict(X_test)
    score_custom = accuracy_score(Y_test, Y_pred_custom)
    print(f'Custom XGBoost Test Score: {score_custom:.4f}')
    
    # Show class probabilities for first 5 test samples
    probs = xgb_classifier.predict_proba(X_test[:5])
    print(f'\nClass probabilities for first 5 test samples:')
    for i, p in enumerate(probs):
        print(f'  Sample {i}: {[round(x, 4) for x in p]}')
    
    #%% XGBoost library for comparison
    # match parameters as closely as possible
    lib_xgb = LibXGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        reg_lambda=1.0,
        gamma=0.0,
        subsample=1.0,
        colsample_bylevel=1.0,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
    )
    lib_xgb.fit(X_train, Y_train.ravel())
    
    Y_pred_lib = lib_xgb.predict(X_test)
    score_lib = accuracy_score(Y_test, Y_pred_lib)
    print(f'\nXGBoost Library Test Score: {score_lib:.4f}')
    
    pass