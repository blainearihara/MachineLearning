# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Decision Tree Regressor modified for Gradient Boosting.

Key differences from the Random Forest and AdaBoost versions:
- This is a REGRESSION tree, not a classification tree. Gradient Boosting
  for classification works by fitting regression trees to the negative
  gradient (pseudo-residuals) of the loss function, not to class labels.
- Splitting criterion uses variance reduction (MSE) instead of Gini/entropy.
- Leaf values use the Newton-Raphson optimal step for log-loss rather than
  a simple mean. The optimal leaf value is:
      leaf = sum(residuals) / sum(p * (1 - p))
  where p is the current predicted probability for the class being fit.
  This uses the second derivative (Hessian) of log-loss to produce better
  calibrated updates per tree, matching sklearn's implementation.
- No sample weights needed (unlike AdaBoost), because the gradient/residual
  values themselves encode where the model is currently struggling.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

#%% Node class (unchanged from original)
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_reduction=None, value=None):
        # Decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_reduction = var_reduction
        
        # Leaf node
        self.value = value

#%% Regression Decision Tree for Gradient Boosting
class DecisionTreeRegressor():
    def __init__(self, min_samp_split=2, max_depth=3, n_classes=2, random_state=None):
        # initialize root of the tree
        self.root = None
        self.n_classes = n_classes
        
        # stopping conditions (max_depth=None means no depth limit)
        self.min_samp_split = min_samp_split
        self.max_depth = max_depth
        # kept for API compatibility: the tree is deterministic, so the global
        # np.random state is not touched
        self.random_state = random_state

    def build_tree(self, dataset, probabilities, curr_depth=0):
        # recursive function to build the tree
        # dataset has columns: [features..., residuals]
        # probabilities is a parallel array of p_k for the class being fit
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samp, num_feat = np.shape(X)

        # split until stopping conditions are met
        # (a node whose residuals are all equal cannot be improved and becomes a leaf)
        depth_ok = self.max_depth is None or curr_depth < self.max_depth
        if num_samp>=self.min_samp_split and depth_ok and np.ptp(Y) > 0:
            # find the best split
            best_split = self.get_best_split(dataset, num_samp, num_feat, probabilities)
            # check if variance reduction is positive
            # (best_split is empty when every feature is constant in this node,
            #  e.g. all rows are identical; the node then becomes a leaf)
            if best_split.get('var_reduction', -float('inf'))>0:
                # recur left
                left_subtree = self.build_tree(best_split['dataset_left'], best_split['probs_left'], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split['dataset_right'], best_split['probs_right'], curr_depth+1)
                # return decision node
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['var_reduction'])
            
        # compute leaf node using Newton-Raphson optimal value
        leaf_value = self.calculate_leaf_value(Y, probabilities)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samp, num_feat, probabilities):
        # dictionary to store best split
        best_split = {}
        max_var_reduction = -float('inf')
        
        # loop over all the features
        for feature_index in range(num_feat):
            feature_values = dataset[:, feature_index]
            # possible_thresholds = np.unique(feature_values)
            sorted_unique = np.sort(np.unique(feature_values))
            midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2
            # loop over all the feature values present in the data
            for threshold in midpoints:
                # get current split
                dataset_left, dataset_right, probs_left, probs_right = self.split(dataset, feature_index, threshold, probabilities)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y = dataset[:, -1]
                    left_y = dataset_left[:, -1]
                    right_y = dataset_right[:, -1]
                    # compute variance reduction. Best split will yield the highest variance reduction 
                    # (i.e., creates the tightest groupings of numbers chosen) 
                    curr_var_reduction = self.variance_reduction(y, left_y, right_y)
                    # update best split if needed
                    if curr_var_reduction>=max_var_reduction:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['probs_left'] = probs_left
                        best_split['probs_right'] = probs_right
                        best_split['var_reduction'] = curr_var_reduction
                        max_var_reduction = curr_var_reduction
                        
        # return best split
        return best_split
    
    def variance_reduction(self, parent, l_child, r_child):
        # compute reduction in variance from this split
        # analogous to information_gain but for regression (MSE-based)
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def split(self, dataset, feature_index, threshold, probabilities):
        # split dataset and corresponding probabilities
        left_mask = dataset[:, feature_index] <= threshold
        right_mask = ~left_mask
        dataset_left = dataset[left_mask]
        dataset_right = dataset[right_mask]
        probs_left = probabilities[left_mask]
        probs_right = probabilities[right_mask]
        return dataset_left, dataset_right, probs_left, probs_right
    
    def calculate_leaf_value(self, residuals, probabilities):
        # Newton-Raphson optimal leaf value for log-loss
        # numerator: sum of residuals (first derivative / gradient)
        # denominator: sum of p * (1 - p) (second derivative / Hessian)
        # this produces better calibrated updates than a simple mean
        numerator = np.sum(residuals)
        denominator = np.sum(probabilities * (1 - probabilities))
        # avoid division by zero
        if abs(denominator) < 1e-10:
            return 0.0
        
        # apply (K-1)/K correction for multiclass softmax coupling
        # the K class probabilities are constrained to sum to 1 through softmax,
        # so each tree's update must be dampened to account for this dependency
        correction = (self.n_classes - 1) / self.n_classes
        return correction * (numerator / denominator)
    
    def print_tree(self, tree=None, indent='    '):
        if tree is None:
            tree = self.root
            
        if tree.value is not None:
            print(round(tree.value, 4))
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", round(tree.var_reduction, 4))
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
        
    def fit(self, X, Y, probabilities=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        # Y can be 1D (residuals) or 2D, handle both
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X.ndim != 2 or Y.shape != (X.shape[0], 1):
            raise ValueError("X must be 2-D (n, m) and Y must have n residuals")
        if X.shape[0] == 0:
            raise ValueError("cannot fit a tree to an empty data set")
        # if no probabilities provided, use p = 0.5 everywhere: every Hessian is then
        # 0.25 and each leaf becomes a scaled mean, (K-1)/K * 4 * mean(residuals)
        if probabilities is None:
            probabilities = np.full(X.shape[0], 0.5)
        probabilities = np.asarray(probabilities, dtype=float).ravel()
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset, probabilities)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
        