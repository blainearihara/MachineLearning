# -*- coding: utf-8 -*-
"""
Created on Wed May 07 2026

Decision Tree Regressor modified for XGBoost.

Key differences from the Gradient Boosting version:
- Splitting criterion uses XGBoost's gain formula derived directly from the
  regularized loss function, instead of variance reduction:
      Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
  where G = sum of gradients, H = sum of Hessians, lambda = L2 regularization,
  and gamma = minimum loss reduction required to make a split.
- Leaf values use the regularized optimal step:
      w = -G / (H + lambda)
  instead of G/H. The lambda term shrinks leaf predictions toward zero,
  preventing any single leaf from being too extreme.
- Post-hoc pruning: the tree is grown (up to max_depth) with every split whose
  raw gain is positive, and each node stores gain - gamma together with its
  gradient/Hessian sums G and H. Afterwards, bottom-up, any split whose two
  children are leaves and whose gain - gamma is negative is collapsed into a
  single leaf with the exact regularized value -G / (H + lambda). This is more
  effective than pre-stopping because sometimes a split that looks bad on
  its own enables a very good split below it (such a split is kept).
- Column subsampling: a random subset of features is drawn once per depth
  level of each tree (colsample_bylevel), adding regularization through randomness.
- The tree works directly with gradient and Hessian arrays instead of
  probabilities, making it agnostic to the loss function being used.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

#%% Node class for XGBoost
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, gain=None, value=None, G=None, H=None):
        # Decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain  # XGBoost gain minus gamma (replaces var_reduction)

        # Leaf node
        self.value = value

        # sums of gradients and Hessians of the samples in this node
        # (needed to give a pruned subtree its exact leaf value)
        self.G = G
        self.H = H

#%% Regression Decision Tree for XGBoost
class DecisionTreeXGBoost():
    def __init__(self, min_samp_split=2, max_depth=6, reg_lambda=1.0, gamma=0.0,
                 colsample_bylevel=1.0, random_state=None):
        # initialize root of the tree
        self.root = None
        
        # stopping conditions (max_depth=None means no depth limit)
        self.min_samp_split = min_samp_split
        self.max_depth = max_depth

        # regularization parameters
        if reg_lambda < 0 or gamma < 0:
            raise ValueError("reg_lambda and gamma must be non-negative")
        self.reg_lambda = reg_lambda  # L2 regularization on leaf weights
        self.gamma = gamma  # minimum gain required to keep a split (applied when pruning)

        # column subsampling
        if not 0 < colsample_bylevel <= 1:
            raise ValueError("colsample_bylevel must be in (0, 1]")
        self.colsample_bylevel = colsample_bylevel

        # private random generator (the global np.random state is never touched);
        # random_state may be None, an int seed, or a np.random.Generator
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self._level_features = {}

    def build_tree(self, X, gradients, hessians, curr_depth=0):
        # recursive function to build the tree
        # X = feature matrix, gradients = first derivatives, hessians = second derivatives
        num_samp, num_feat = np.shape(X)
        G, H = np.sum(gradients), np.sum(hessians)

        # split until stopping conditions are met
        depth_ok = self.max_depth is None or curr_depth < self.max_depth
        if num_samp>=self.min_samp_split and depth_ok:
            # find the best split using XGBoost gain formula
            best_split = self.get_best_split(X, gradients, hessians, num_samp, num_feat, curr_depth)
            # grow every split whose raw gain (before subtracting gamma) is positive
            # (beyond floating-point rounding); gamma is applied afterwards by prune_tree
            if best_split.get('gain', -float('inf')) + self.gamma > 1e-12:
                # recur left
                left_subtree = self.build_tree(
                    best_split['X_left'], best_split['grad_left'],
                    best_split['hess_left'], curr_depth+1
                )
                # recur right
                right_subtree = self.build_tree(
                    best_split['X_right'], best_split['grad_right'],
                    best_split['hess_right'], curr_depth+1
                )
                # return decision node
                return Node(best_split['feature_index'], best_split['threshold'],
                           left_subtree, right_subtree, best_split['gain'], G=G, H=H)

        # compute leaf node using regularized optimal weight
        leaf_value = self.calculate_leaf_value(gradients, hessians)
        # return leaf node
        return Node(value=leaf_value, G=G, H=H)

    def get_best_split(self, X, gradients, hessians, num_samp, num_feat, curr_depth=None):
        # dictionary to store best split
        best_split = {}
        max_gain = -float('inf')

        # determine features to consider (column subsampling, one draw per depth level);
        # if none of them can split this node, the remaining features are tried
        feature_indices = list(self.get_feature_subset(num_feat, curr_depth))
        remaining = [j for j in range(num_feat) if j not in set(feature_indices)]

        # precompute parent statistics
        G_total = np.sum(gradients)
        H_total = np.sum(hessians)

        # loop over selected features
        for feature_index in feature_indices + (remaining if num_feat > len(feature_indices) else []):
            if best_split and feature_index in remaining:
                break    # a valid split was found among the sampled features
            feature_values = X[:, feature_index]
            sorted_unique = np.sort(np.unique(feature_values))
            # use midpoint thresholds for better generalization
            midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2
            
            # loop over all midpoint thresholds
            for threshold in midpoints:
                # get current split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # check if childs are not null
                if np.sum(left_mask)>0 and np.sum(right_mask)>0:
                    # compute gradient and hessian sums for each child
                    G_left = np.sum(gradients[left_mask])
                    H_left = np.sum(hessians[left_mask])
                    G_right = np.sum(gradients[right_mask])
                    H_right = np.sum(hessians[right_mask])
                    
                    # compute XGBoost gain
                    curr_gain = self.compute_gain(G_left, H_left, G_right, H_right, G_total, H_total)
                    
                    # update best split if needed
                    if curr_gain>=max_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['X_left'] = X[left_mask]
                        best_split['X_right'] = X[right_mask]
                        best_split['grad_left'] = gradients[left_mask]
                        best_split['grad_right'] = gradients[right_mask]
                        best_split['hess_left'] = hessians[left_mask]
                        best_split['hess_right'] = hessians[right_mask]
                        best_split['gain'] = curr_gain
                        max_gain = curr_gain
                        
        # return best split
        return best_split
    
    def compute_gain(self, G_left, H_left, G_right, H_right, G_total, H_total):
        # XGBoost gain formula derived from the regularized objective
        # measures how much the regularized loss improves from making this split
        # Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
        #
        # first term: score if we split into left child
        # second term: score if we split into right child
        # third term: score if we don't split (parent)
        # gamma: penalty for adding a new leaf (controls tree complexity)
        score_left = self._score(G_left, H_left)
        score_right = self._score(G_right, H_right)
        score_parent = self._score(G_total, H_total)
        gain = 0.5 * (score_left + score_right - score_parent) - self.gamma
        return gain

    def _score(self, G, H):
        # G^2 / (H + lambda), defined as 0 when H + lambda is 0 (e.g. lambda = 0 and all
        # Hessians are 0) instead of dividing by zero
        denom = H + self.reg_lambda
        return (G ** 2) / denom if denom > 0 else 0.0

    def get_feature_subset(self, num_feat, depth=None):
        # select random subset of features for column subsampling;
        # with a depth, the subset is drawn once per depth level and reused
        # for every node at that level (colsample_bylevel semantics)
        if self.colsample_bylevel >= 1.0:
            return range(num_feat)
        if depth is not None and depth in self._level_features:
            return self._level_features[depth]
        max_feat = max(1, int(num_feat * self.colsample_bylevel))
        subset = self.rng.choice(num_feat, max_feat, replace=False)
        if depth is not None:
            self._level_features[depth] = subset
        return subset

    def calculate_leaf_value(self, gradients, hessians):
        # regularized optimal leaf weight from XGBoost objective
        # w* = -G / (H + lambda)
        # the lambda term shrinks the prediction toward zero, preventing
        # any single leaf from having an outsized influence
        G = np.sum(gradients)
        H = np.sum(hessians)
        return self._leaf_value(G, H)

    def _leaf_value(self, G, H):
        # -G / (H + lambda), or 0 when H + lambda is 0 (no curvature information)
        denom = H + self.reg_lambda
        return -G / denom if denom > 0 else 0.0
    
    def prune_tree(self, node):
        # post-hoc pruning: remove splits where gain is negative after
        # accounting for gamma. This is more effective than pre-stopping
        # because a "bad" split might enable good splits below it.
        # We prune bottom-up: first prune children, then check this node.
        if node is None or node.value is not None:
            return node
        
        # recursively prune children first
        node.left = self.prune_tree(node.left)
        node.right = self.prune_tree(node.right)
        
        # if both children are now leaves and this node's gain is negative,
        # collapse this subtree into a single leaf
        if (node.left is not None and node.left.value is not None and
            node.right is not None and node.right.value is not None):
            if node.gain is not None and node.gain < 0:
                # merge: the collapsed leaf gets the exact regularized value
                # -G / (H + lambda) of all the samples in this node
                return Node(value=self._leaf_value(node.G, node.H), G=node.G, H=node.H)

        return node

    def print_tree(self, tree=None, indent='    '):
        if tree is None:
            tree = self.root
            
        if tree.value is not None:
            print(round(tree.value, 4))
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", round(tree.gain, 4))
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
        
    def fit(self, X, gradients, hessians):
        X = np.asarray(X, dtype=float)
        gradients = np.asarray(gradients, dtype=float).ravel()
        hessians = np.asarray(hessians, dtype=float).ravel()
        if X.ndim != 2 or not (len(gradients) == len(hessians) == X.shape[0]):
            raise ValueError("X must be 2-D (n, m) with n gradients and n Hessians")
        if X.shape[0] == 0:
            raise ValueError("cannot fit a tree to an empty data set")
        # restart the generator and the per-level feature draws for this tree
        self.rng = np.random.default_rng(self.random_state)
        self._level_features = {}
        # build tree from feature matrix and gradient/hessian arrays
        self.root = self.build_tree(X, gradients, hessians)
        # apply post-hoc pruning
        self.root = self.prune_tree(self.root)

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