# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 2026

AdaBoost (Adaptive Boosting) Classifier - SAMME variant for multiclass.

Key differences from Random Forest:
- Random Forest: trains trees independently on bootstrap samples, equal vote.
- AdaBoost: trains trees sequentially, each tree focuses on samples the
  previous trees got wrong by increasing their weights. Each tree gets a
  weighted vote (alpha) based on how accurate it was.

Algorithm (SAMME - Stagewise Additive Modeling using a Multi-class
Exponential loss function):
  1. Initialize uniform sample weights: w_i = 1/N
  2. For each estimator t = 1..T:
     a. Train a weak learner (decision stump) using weighted samples
     b. Compute weighted error: err = sum(w_i * I(y_i != pred_i)) / sum(w_i)
     c. Compute learner weight: alpha = learning_rate * 0.5 * (log((1-err)/err) + log(K-1))
        where K = number of classes
     d. Update sample weights: w_i *= exp(+alpha) if misclassified, exp(-alpha) if correct
     e. Normalize weights so they sum to 1
     Steps c-e are the symmetric (binary AdaBoost) form of SAMME: after
     normalization they give exactly the SAMME weights w_i *= exp(2*alpha * I(y_i != pred_i)),
     and halving every alpha does not change the weighted vote below.
     Boosting stops early if a learner is no better than chance (err >= 1 - 1/K)
     or if a learner is perfect (err = 0).
  3. Final prediction: weighted majority vote using alpha values

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from DecisionTreeAdaBoost import DecisionTreeClassifier as DecisionTreeClassifier

#%% AdaBoost Classifier
class AdaBoostClassifier():
    def __init__(self, n_estimators=50, max_depth=1, learning_rate=1.0, random_state=None):
        # boosting parameters
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be at least 1")
        if not learning_rate > 0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # kept for API compatibility: the algorithm is deterministic, so the global
        # np.random state is not touched
        self.random_state = random_state

        # initialize stumps and their weights
        self.trees = []
        self.alphas = []

    def fit(self, X, Y):
        # clear any existing trees and alphas
        self.trees = []
        self.alphas = []
        self.estimator_errors_ = []    # weighted training error of each kept learner

        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        num_samples = X.shape[0]
        # flatten Y for comparison operations
        Y_flat = Y.ravel()

        # determine number of classes
        self.classes_, class_counts = np.unique(Y_flat, return_counts=True)
        n_classes = len(self.classes_)
        # prediction used if no learner could be kept: the most frequent class
        self.fallback_class_ = self.classes_[np.argmax(class_counts)]
        if n_classes < 2:
            return      # only one class: nothing to boost, predict() returns that class

        # step 1: initialize uniform sample weights
        sample_weights = np.ones(num_samples) / num_samples
        # step 2: build each weak learner sequentially
        for i in range(self.n_estimators):
            # create and train a decision stump with current sample weights
            tree = DecisionTreeClassifier(
                min_samp_split=2,
                max_depth=self.max_depth
            )
            tree.fit(X, Y, sample_weights)

            # get predictions on training data
            predictions = np.array(tree.predict(X))

            # step 2b: compute weighted error
            correct = (predictions == Y_flat).astype(int)
            misclassified = (predictions != Y_flat).astype(int)
            raw_error = np.sum(sample_weights * misclassified) / np.sum(sample_weights)

            # clip error to avoid division by zero or log of zero
            weighted_error = np.clip(raw_error, 1e-10, 1 - 1e-10)

            # stop boosting if this learner is no better than random guessing
            # (its alpha would be zero or negative)
            if weighted_error >= (1 - 1.0/n_classes):
                break
            
            # step 2c: compute learner weight (alpha) using SAMME formula
            # for binary (K=2): log(K-1) = log(1) = 0, reduces to standard AdaBoost
            # for multiclass (K>2): the log(K-1) term adjusts for random chance
            alpha = self.learning_rate * 0.5*(np.log((1 - weighted_error) / weighted_error) + np.log(n_classes - 1))
            
            # step 2d: update sample weights (increase weight of misclassified)
            sample_weights = sample_weights * np.exp(alpha * misclassified)
            # step 2d: update sample weights (decrease weight of correctly classified)
            sample_weights = sample_weights * np.exp(-alpha * correct)
            
            # step 2e: normalize weights to sum to 1
            sample_weights = sample_weights / np.sum(sample_weights)
            
            # store the tree and its weight
            self.trees.append(tree)
            self.alphas.append(alpha)
            self.estimator_errors_.append(raw_error)

            # a perfect learner leaves the normalized weights unchanged, so every
            # further round would refit the same tree: stop here
            if raw_error <= 0:
                break

    def predict(self, X):
        # get weighted predictions from all trees
        num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        if not self.trees:
            # no learner was kept (one class only, or the first learner was no better
            # than chance): predict the most frequent training class
            return [self.fallback_class_] * num_samples
        
        # accumulate weighted votes for each class per sample
        # shape: (num_samples, num_classes)
        class_scores = np.zeros((num_samples, len(self.classes_)))
        
        for tree, alpha in zip(self.trees, self.alphas):
            predictions = np.array(tree.predict(X))
            # add alpha to the score of the predicted class for each sample
            for j, cls in enumerate(self.classes_):
                class_scores[:, j] += alpha * (predictions == cls).astype(float)
        
        # final prediction: class with highest accumulated weighted vote
        final_predictions = []
        for scores in class_scores:
            best_class_idx = np.argmax(scores)
            final_predictions.append(self.classes_[best_class_idx])
        
        return final_predictions
    
    def get_staged_predictions(self, X):
        # return predictions at each boosting stage (useful for analysis)
        num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        staged_predictions = []
        class_scores = np.zeros((num_samples, len(self.classes_)))
        
        for tree, alpha in zip(self.trees, self.alphas):
            predictions = np.array(tree.predict(X))
            for j, cls in enumerate(self.classes_):
                class_scores[:, j] += alpha * (predictions == cls).astype(float)
            
            # prediction at this stage
            stage_preds = []
            for scores in class_scores:
                best_class_idx = np.argmax(scores)
                stage_preds.append(self.classes_[best_class_idx])
            staged_predictions.append(stage_preds)
        
        return staged_predictions
    
    def get_estimator_weights(self):
        # return the alpha weights for each estimator
        return self.alphas
    
    def get_estimator_errors(self, X, Y):
        # return the UNWEIGHTED error rate of each estimator on the given data
        # (the weighted training errors used to compute alpha are in self.estimator_errors_)
        Y_flat = np.asarray(Y).ravel()
        errors = []
        for tree in self.trees:
            predictions = np.array(tree.predict(X))
            error = np.mean(predictions != Y_flat)
            errors.append(error)
        return errors