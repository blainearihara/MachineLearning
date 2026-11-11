# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 17:42:39 2025
4
@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression as LinearRegression


if __name__ == "__main__":
    
    np.random.seed(42)
    num_deg = 3
    num_train = 200
    
    # Generate fake data
    X = np.random.uniform(low=0, high=3, size=(num_train,))
    y = 20*X - 28*X**2 - 7*X**3 + 26*X**4 - 13*X**5 + 2*X**6
    noise = np.random.normal(loc=0, scale=1, size=(num_train))
    y += noise
    
    # Instantiate model
    model = LinearRegression(degree=num_deg, include_bias=True, regularization='ridge')
    model.fit(X, y)
    # predictions = model.predict(X_new)
    coefficients = model.get_coefficients()
    r2_score = model.score(X, y)
    mse = model.calculate_mse(X, y)
    model.plot_fit(X, y)
    
    pass