# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 12:09:42 2025

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from RandomForest import RandomForestClassifier as RandomForestClassifier
from GetKaggleDataset import get_dataset

from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits, load_wine

#%% Main
if __name__ == "__main__":
    
    #%% Random Forest Custom Model
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
    
    #%% Custom Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=10,
        min_samp_split=3,
        max_depth=3,
        max_features='sqrt',
        random_state=42
    )
    # Train model
    rf_classifier.fit(X_train, Y_train)

    # Test model
    Y_pred_custom = rf_classifier.predict(X_test)
    score_custom = accuracy_score(Y_test, Y_pred_custom)
    print(f'Custom Random Forest Test Score: {score_custom:.4f}')
    
    #%% Sklearn Random Forest for comparison
    sklearn_rf = SklearnRandomForest(
        n_estimators=10,
        min_samples_split=3,
        max_depth=3,
        max_features='sqrt',
        random_state=42
    )
    sklearn_rf.fit(X_train, Y_train.ravel())
    
    Y_pred_sklearn = sklearn_rf.predict(X_test)
    score_sklearn = accuracy_score(Y_test, Y_pred_sklearn)
    print(f'Sklearn Random Forest Test Score: {score_sklearn:.4f}')
    
    pass
    
    
    
    