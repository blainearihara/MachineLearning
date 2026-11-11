# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Gradient Boosting Regressor using MSE (mean squared error) loss.

Key differences from Gradient Boosting Classifier:
- Classifier: maintains one raw score per class per sample, fits K trees
  per round (one per class), uses softmax to convert scores to probabilities,
  uses log-loss and Newton-Raphson leaf values, initializes with log-priors.
- Regressor: maintains a single prediction per sample, fits one tree per
  round, uses MSE loss where the gradient is simply (y - prediction), leaf
  values are the mean of residuals, initializes with mean(Y).

Algorithm (Gradient Boosting for regression with MSE loss):
  1. Initialize predictions F(x) = mean(Y) for all samples. This is the
     optimal constant prediction that minimizes MSE before any trees are added.
  2. For each boosting round t = 1..T:
     a. Compute pseudo-residuals: r_i = y_i - F(x_i)
        (negative gradient of MSE loss, which is simply the error)
     b. Fit a regression tree to the residuals r_i
     c. Update predictions: F(x) += learning_rate * tree_prediction
  3. Final prediction: initial mean + sum of all scaled tree corrections

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from DecisionTreeGradientBoostRegression import DecisionTreeRegressor as DecisionTreeRegressor

#%% Gradient Boosting Regressor
class GradientBoostingRegressor():
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, min_samp_split=2, random_state=None):
        # boosting parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samp_split = min_samp_split
        self.random_state = random_state
        
        # initialize trees (single flat list, not per-class dict)
        self.trees = []
        self.init_prediction_ = None  # mean(Y) baseline
        # random_state is kept for API compatibility: the algorithm is deterministic
        # (no subsampling), so the global np.random state is not touched

    def fit(self, X, Y):
        # clear any existing trees
        self.trees = []

        X = np.asarray(X, dtype=float)
        num_samples = X.shape[0]
        Y_flat = np.asarray(Y, dtype=float).ravel()
        
        # step 1: initialize predictions to mean of target values
        # this is the optimal constant prediction for MSE loss,
        # analogous to log-prior initialization in the classifier
        self.init_prediction_ = np.mean(Y_flat)
        
        # current predictions for all training samples
        predictions = np.full(num_samples, self.init_prediction_)
        
        # step 2: build regression trees sequentially
        for i in range(self.n_estimators):
            # step 2a: compute pseudo-residuals (negative gradient of MSE)
            # for MSE loss L = (y - F)^2 / 2, the negative gradient is exactly (y - F)
            # (with L = (y - F)^2 it is 2(y - F); the factor 2 is absorbed by the leaf values)
            residuals = Y_flat - predictions

            # step 2b: fit a regression tree to the residuals
            tree = DecisionTreeRegressor(
                min_samp_split=self.min_samp_split,
                max_depth=self.max_depth
            )
            tree.fit(X, residuals)
            
            # get tree predictions (corrections)
            tree_predictions = tree.predict(X)
            
            # step 2c: update predictions with scaled corrections
            predictions += self.learning_rate * tree_predictions
            
            # store the tree
            self.trees.append(tree)
    
    def predict(self, X):
        # start from the initial mean prediction
        num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        predictions = np.full(num_samples, self.init_prediction_)
        
        # accumulate corrections from all trees
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
        
        return predictions
    
    def get_staged_predictions(self, X):
        # return predictions at each boosting stage (useful for analysis)
        num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        staged_predictions = []
        
        # start from the initial mean prediction
        predictions = np.full(num_samples, self.init_prediction_)
        
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions = predictions + self.learning_rate * tree_predictions
            # store a copy of predictions at this stage
            staged_predictions.append(predictions.copy())
        
        return staged_predictions
    
    def get_feature_importance(self, num_features):
        # calculate feature importances across all trees
        if not self.trees:
            return None
        
        importances = np.zeros(num_features)
        
        # aggregate variance reduction from all trees
        for tree in self.trees:
            tree_importances = self.calculate_tree_importance(tree.root, num_features)
            importances += tree_importances
        
        # normalize by number of trees
        importances = importances / len(self.trees)
        
        return importances
    
    def calculate_tree_importance(self, node, num_features):
        # recursively calculate feature importance for a single tree
        importances = np.zeros(num_features)
        
        if node is None or node.value is not None:
            return importances
        
        # add importance of current node (variance reduction)
        if node.var_reduction and node.feature_index < num_features:
            importances[node.feature_index] += node.var_reduction
        
        # add importances from children
        importances += self.calculate_tree_importance(node.left, num_features)
        importances += self.calculate_tree_importance(node.right, num_features)
        
        return importances