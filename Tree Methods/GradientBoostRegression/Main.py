# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Main driver for Gradient Boosting Regressor comparison.
Compares custom Gradient Boosting Regressor implementation against sklearn's
GradientBoostingRegressor.

Uses regression datasets (continuous targets) instead of the classification
datasets used in the classifier version.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from GradientBoostRegression import GradientBoostingRegressor as GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoosting
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, fetch_california_housing

#%% Main
if __name__ == "__main__":
    
    #%% Gradient Boosting Regression Custom Model
    data_name = 'california'
    # Load data from sklearn library
    if data_name == 'diabetes':
        loaded_data = load_diabetes()
    elif data_name == 'california':
        loaded_data = fetch_california_housing()
        
    df = pd.DataFrame(loaded_data.data, columns=loaded_data.feature_names)
    df['target'] = loaded_data.target
    print(df.head())
    print(f'\nTarget statistics:')
    print(f'  Mean:  {loaded_data.target.mean():.4f}')
    print(f'  Std:   {loaded_data.target.std():.4f}')
    print(f'  Min:   {loaded_data.target.min():.4f}')
    print(f'  Max:   {loaded_data.target.max():.4f}')
    X, Y = loaded_data.data, loaded_data.target.reshape(-1, 1)   
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    
    # Custom Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samp_split=3,
        random_state=42
    )
    # Train model
    gb_regressor.fit(X_train, Y_train)

    # Test model
    Y_pred_custom = gb_regressor.predict(X_test)
    mse_custom = mean_squared_error(Y_test, Y_pred_custom)
    r2_custom = r2_score(Y_test, Y_pred_custom)
    print(f'\nCustom Gradient Boosting Regressor:')
    print(f'  MSE:  {mse_custom:.4f}')
    print(f'  RMSE: {np.sqrt(mse_custom):.4f}')
    print(f'  R2:   {r2_custom:.4f}')
    
    # Show predictions vs actual for first 10 test samples
    print(f'\nFirst 10 predictions vs actual:')
    print(f'  {"Predicted":<12} {"Actual":<12} {"Error":<12}')
    for pred, actual in zip(Y_pred_custom[:10], Y_test.ravel()[:10]):
        print(f'  {pred:<12.2f} {actual:<12.2f} {pred - actual:<12.2f}')
    
    #%% Sklearn Gradient Boosting Regressor for comparison
    sklearn_gb = SklearnGradientBoosting(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_split=3,
        random_state=42
    )
    sklearn_gb.fit(X_train, Y_train.ravel())
    
    Y_pred_sklearn = sklearn_gb.predict(X_test)
    mse_sklearn = mean_squared_error(Y_test, Y_pred_sklearn)
    r2_sklearn = r2_score(Y_test, Y_pred_sklearn)
    print(f'\nSklearn Gradient Boosting Regressor:')
    print(f'  MSE:  {mse_sklearn:.4f}')
    print(f'  RMSE: {np.sqrt(mse_sklearn):.4f}')
    print(f'  R2:   {r2_sklearn:.4f}')
    
    pass