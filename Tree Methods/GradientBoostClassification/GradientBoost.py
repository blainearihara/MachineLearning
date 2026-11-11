# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Gradient Boosting Classifier using one-vs-rest with softmax probabilities.

Key differences from Random Forest and AdaBoost:
- Random Forest: trains classification trees independently, majority vote.
- AdaBoost: trains classification stumps sequentially, re-weights samples,
  weighted vote using alpha.
- Gradient Boosting: trains REGRESSION trees sequentially, each tree fits
  the negative gradient (pseudo-residuals) of a loss function. Trees predict
  continuous corrections, not class labels. Final prediction combines all
  corrections through a softmax over accumulated raw scores.

Algorithm (Gradient Boosting for multiclass with log-loss):
  1. Initialize raw scores F_k = log(prior_k) for each class k, where
     prior_k is the proportion of class k in the training data. This gives
     the model a head start by encoding the base rates.
  2. For each boosting round t = 1..T:
     a. Compute probabilities via softmax: p_k = exp(F_k) / sum(exp(F_j))
     b. For each class k:
        - Compute pseudo-residuals: r_ik = y_ik - p_ik
          (where y_ik = 1 if sample i is class k, else 0)
        - Fit a regression tree to the residuals r_ik, passing p_ik so
          leaf values can use the Newton-Raphson optimal step:
          leaf = sum(residuals) / sum(p * (1 - p))
        - Update raw scores: F_k += learning_rate * tree_prediction
  3. Final prediction: class with highest accumulated raw score (or softmax prob)

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from DecisionTreeGradientBoost import DecisionTreeRegressor as DecisionTreeRegressor

#%% Gradient Boosting Classifier
class GradientBoostingClassifier():
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, min_samp_split=2, random_state=None):
        # boosting parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samp_split = min_samp_split
        self.random_state = random_state
        
        # initialize trees (one list per class) and class info
        self.trees = {}
        self.classes_ = None
        self.n_classes_ = None
        self.init_raw_scores_ = None  # log-prior initial scores
        # random_state is kept for API compatibility: the algorithm is deterministic,
        # so the global np.random state is not touched

    def softmax(self, raw_scores):
        # compute softmax probabilities from raw scores
        # subtract max for numerical stability (prevents overflow in exp)
        shifted = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probabilities
    
    def fit(self, X, Y):
        # clear any existing trees
        self.trees = {}

        X = np.asarray(X, dtype=float)
        num_samples = X.shape[0]
        Y_flat = np.asarray(Y).ravel()
        
        # determine classes
        self.classes_ = np.unique(Y_flat)
        self.n_classes_ = len(self.classes_)
        
        # create one-hot encoded target matrix
        # shape: (num_samples, n_classes)
        Y_onehot = np.zeros((num_samples, self.n_classes_))
        for k, cls in enumerate(self.classes_):
            Y_onehot[:, k] = (Y_flat == cls).astype(float)
        
        # initialize tree lists for each class
        for k in range(self.n_classes_):
            self.trees[k] = []
        
        # step 1: initialize raw scores to log-prior of each class
        # prior_k = count of class k / total samples
        # F_k(0) = log(prior_k) gives the model a head start by encoding
        # the base rate of each class, which is especially helpful for
        # imbalanced datasets where starting at zero means the first
        # several rounds are just learning the base rates
        self.init_raw_scores_ = np.zeros(self.n_classes_)
        for k, cls in enumerate(self.classes_):
            prior_k = np.sum(Y_flat == cls) / num_samples
            # clip to avoid log(0) for classes with no samples
            prior_k = np.clip(prior_k, 1e-10, 1.0)
            self.init_raw_scores_[k] = np.log(prior_k)
        
        # broadcast initial scores to all samples
        # shape: (num_samples, n_classes)
        raw_scores = np.tile(self.init_raw_scores_, (num_samples, 1))
        
        # step 2: build regression trees sequentially
        for i in range(self.n_estimators):
            # step 2a: compute current probabilities via softmax
            probabilities = self.softmax(raw_scores)
            
            # step 2b: for each class, compute residuals and fit a tree
            for k in range(self.n_classes_):
                # current class probabilities for Newton-Raphson leaf values
                probs_k = probabilities[:, k]
                
                # pseudo-residuals: negative gradient of log-loss
                # residual = true_probability(one-hot) - predicted_probability
                residuals = Y_onehot[:, k] - probs_k
                
                # fit a regression tree to the residuals
                # pass probabilities so leaves can compute optimal step
                tree = DecisionTreeRegressor(
                    min_samp_split=self.min_samp_split,
                    max_depth=self.max_depth,
                    n_classes=self.n_classes_
                )
                tree.fit(X, residuals, probs_k)
                
                # get tree predictions (Newton-Raphson optimized corrections)
                tree_predictions = tree.predict(X)
                
                # step 2c: update raw scores with scaled corrections
                raw_scores[:, k] += self.learning_rate * tree_predictions
                
                # store the tree
                self.trees[k].append(tree)
    
    def predict_proba(self, X):
        # compute class probabilities for new data
        num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        
        # start from log-prior initial scores
        raw_scores = np.tile(self.init_raw_scores_, (num_samples, 1))
        
        # accumulate corrections from all trees
        for k in range(self.n_classes_):
            for tree in self.trees[k]:
                tree_predictions = tree.predict(X)
                raw_scores[:, k] += self.learning_rate * tree_predictions
        
        # convert raw scores to probabilities
        probabilities = self.softmax(raw_scores)
        return probabilities
    
    def predict(self, X):
        # predict class labels
        probabilities = self.predict_proba(X)
        
        # class with highest probability
        final_predictions = []
        for probs in probabilities:
            best_class_idx = np.argmax(probs)
            final_predictions.append(self.classes_[best_class_idx])
        
        return final_predictions
    
    def get_staged_predictions(self, X):
        # return predictions at each boosting stage (useful for analysis)
        num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        staged_predictions = []
        
        # start from log-prior initial scores
        raw_scores = np.tile(self.init_raw_scores_, (num_samples, 1))
        
        n_rounds = len(self.trees[0]) if self.trees else 0      # rounds actually fitted
        for i in range(n_rounds):
            # add contributions from round i for each class
            for k in range(self.n_classes_):
                tree_predictions = self.trees[k][i].predict(X)
                raw_scores[:, k] += self.learning_rate * tree_predictions
            
            # prediction at this stage
            probabilities = self.softmax(raw_scores)
            stage_preds = []
            for probs in probabilities:
                best_class_idx = np.argmax(probs)
                stage_preds.append(self.classes_[best_class_idx])
            staged_predictions.append(stage_preds)
        
        return staged_predictions
    
    def get_feature_importance(self, num_features):
        # calculate feature importances across all trees and classes
        if not self.trees:
            return None
        
        importances = np.zeros(num_features)
        total_trees = 0
        
        # aggregate variance reduction from all trees across all classes
        for k in range(self.n_classes_):
            for tree in self.trees[k]:
                tree_importances = self.calculate_tree_importance(tree.root, num_features)
                importances += tree_importances
                total_trees += 1
        
        # normalize by total number of trees
        if total_trees > 0:
            importances = importances / total_trees
        
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