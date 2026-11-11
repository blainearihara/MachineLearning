# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 23:37:57 2025

@author: blain
"""
#%% Libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from RegressionTree import DecisionTreeRegressor as dtr
# Scikit-learn's regression tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree


#%% Main
if __name__ == "__main__":
    
    data = pd.read_csv("airfoil_noise_data.csv")
    data.head(5)
    
    # Train-Test Split
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    
    regressor = dtr(min_samples_split=3, max_depth=3)
    regressor.fit(X_train, Y_train)
    regressor.print_tree()
    
    # Test the model
    Y_pred = regressor.predict(X_test)
    error_pred = np.sqrt(mean_squared_error(Y_test, Y_pred))
    print(f'Custom Root mean squared error: {error_pred}')
    
    #%% SCIKIT LEARN REGRESSION TREE
    # Instantiate regressor model
    sklearn_regressor = DecisionTreeRegressor(
        min_samples_split=3,
        max_depth=3,
        random_state=41
        )
    
    # Train sklearn model
    sklearn_regressor.fit(X_train, Y_train)
    
    # Test sklearn model
    Y_pred_sklearn = sklearn_regressor.predict(X_test)
    error_pred_sklearn = np.sqrt(mean_squared_error(Y_test, Y_pred_sklearn))
    print(f'Sklearn Root mean squared error: {error_pred_sklearn}')
    
    # Visualize the sklearn tree
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    plt.figure(figsize=(15, 10))
    plot_tree(sklearn_regressor,
              feature_names=feature_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Scikit-learn Decision Tree Visualization')
    plt.show()
    
    
    