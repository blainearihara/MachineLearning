# -*- coding: utf-8 -*-
"""
Created on Wed May 07 2026

XGBoost Classifier using one-vs-rest with softmax probabilities.

Key differences from Gradient Boosting:
- Gradient Boosting: uses variance reduction for splits, computes gradients
  (residuals) but derives Hessians implicitly from probabilities, applies
  (K-1)/K correction, tree is loss-agnostic during construction.
- XGBoost: derives splits, leaf values, and pruning ALL directly from the
  regularized loss function. The tree receives explicit gradient and Hessian
  arrays and uses them in every decision. L2 regularization (lambda) shrinks
  leaf values, minimum gain (gamma) prunes unnecessary splits, and column/row
  subsampling adds further regularization.

Algorithm (XGBoost for multiclass with softmax log-loss):
  1. Initialize raw scores F_k = 0 for all samples and classes.
     (XGBoost uses zero initialization by default, unlike sklearn's log-prior)
  2. For each boosting round t = 1..T:
     a. Compute probabilities via softmax: p_k = exp(F_k) / sum(exp(F_j))
     b. For each class k:
        - Compute gradients: g_ik = p_ik - y_ik (first derivative of log-loss)
        - Compute Hessians: h_ik = p_ik * (1 - p_ik) (second derivative)
        - Optionally subsample rows (subsample parameter)
        - Fit a regression tree using the XGBoost gain formula:
            Gain = 0.5 * [G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam)] - gamma
          with regularized leaf values: w = -G/(H+lam)
        - Update raw scores: F_k += learning_rate * tree_prediction
  3. Final prediction: class with highest accumulated raw score

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from DecisionTreeXGBoost import DecisionTreeXGBoost as DecisionTreeXGBoost

#%% XGBoost Classifier
class XGBoostClassifier():
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, min_samp_split=2,
                 reg_lambda=1.0, gamma=0.0, subsample=1.0, colsample_bylevel=1.0,
                 random_state=None):
        # boosting parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samp_split = min_samp_split
        
        # regularization parameters (key XGBoost additions)
        self.reg_lambda = reg_lambda  # L2 regularization on leaf weights
        self.gamma = gamma  # minimum loss reduction to make a split
        
        # subsampling parameters
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < colsample_bylevel <= 1:
            raise ValueError("colsample_bylevel must be in (0, 1]")
        self.subsample = subsample  # fraction of rows to sample per round
        self.colsample_bylevel = colsample_bylevel  # fraction of features per depth level

        # private random generator (the global np.random state is never touched)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # initialize trees (one list per class) and class info
        self.trees = {}
        self.classes_ = None
        self.n_classes_ = None

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
        # restart the generator so that refitting with the same random_state is reproducible
        self.rng = np.random.default_rng(self.random_state)

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
        
        # step 1: initialize raw scores to zero
        # XGBoost uses zero initialization by default, unlike sklearn's Gradient
        # Boosting which uses log-prior. The first few rounds of boosting learn
        # the base rates instead.
        raw_scores = np.zeros((num_samples, self.n_classes_))
        
        # step 2: build regression trees sequentially
        for i in range(self.n_estimators):
            # step 2a: compute current probabilities via softmax
            probabilities = self.softmax(raw_scores)
            
            # row subsampling: select a random subset of samples for this round
            if self.subsample < 1.0:
                n_subsample = max(1, int(num_samples * self.subsample))
                subsample_indices = self.rng.choice(num_samples, n_subsample, replace=False)
            else:
                subsample_indices = np.arange(num_samples)
            
            X_sub = X[subsample_indices]
            
            # step 2b: for each class, compute gradients/hessians and fit a tree
            for k in range(self.n_classes_):
                probs_k = probabilities[:, k]
                
                # gradient: first derivative of log-loss with softmax
                # g_i = p_ik - y_ik (note: opposite sign from gradient boosting residuals)
                # XGBoost convention: gradient points in the direction of increasing loss
                gradients = probs_k - Y_onehot[:, k]
                
                # hessian: second derivative of log-loss with softmax
                # h_i = p_ik * (1 - p_ik)
                hessians = probs_k * (1 - probs_k)
                
                # apply subsampling
                grad_sub = gradients[subsample_indices]
                hess_sub = hessians[subsample_indices]
                
                # fit a tree using XGBoost gain formula
                tree = DecisionTreeXGBoost(
                    min_samp_split=self.min_samp_split,
                    max_depth=self.max_depth,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                    colsample_bylevel=self.colsample_bylevel,
                    random_state=int(self.rng.integers(2**31 - 1))   # per-tree seed from the model's generator
                )
                tree.fit(X_sub, grad_sub, hess_sub)
                
                # get tree predictions for ALL samples (not just subsample)
                # this ensures raw_scores stays aligned with the full dataset
                tree_predictions = tree.predict(X)
                
                # step 2c: update raw scores with scaled corrections
                raw_scores[:, k] += self.learning_rate * tree_predictions
                
                # store the tree
                self.trees[k].append(tree)
    
    def predict_proba(self, X):
        # compute class probabilities for new data
        num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        
        # start from zero (XGBoost zero initialization)
        raw_scores = np.zeros((num_samples, self.n_classes_))
        
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
        
        # start from zero
        raw_scores = np.zeros((num_samples, self.n_classes_))
        
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
        # uses total gain (sum of gain at each split involving the feature)
        if not self.trees:
            return None
        
        importances = np.zeros(num_features)
        total_trees = 0
        
        # aggregate gain from all trees across all classes
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
        
        # add importance of current node: its raw loss reduction (the stored gain
        # has gamma subtracted, so gamma is added back; always positive)
        if node.gain is not None and node.feature_index < num_features:
            importances[node.feature_index] += node.gain + self.gamma
        
        # add importances from children
        importances += self.calculate_tree_importance(node.left, num_features)
        importances += self.calculate_tree_importance(node.right, num_features)
        
        return importances