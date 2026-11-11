# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 2026

Decision Tree Classifier modified for AdaBoost.
Key difference from the Random Forest version: this tree supports
sample_weights in its splitting criteria, impurity calculations,
and leaf value computations. AdaBoost requires weighted learning
so that each successive tree focuses more on previously misclassified
samples.

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

#%% Node class (unchanged from original)
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # Decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # Leaf node
        self.value = value

#%% Modified Decision Tree for AdaBoost (supports sample weights)
class DecisionTreeClassifier():
    def __init__(self, min_samp_split=2, max_depth=1, random_state=None):
        # initialize root of the tree
        self.root = None
        
        # stopping conditions (max_depth=None means no depth limit)
        self.min_samp_split = min_samp_split
        self.max_depth = max_depth
        # kept for API compatibility; the tree is deterministic and uses no randomness,
        # so the global np.random state is not touched
        self.random_state = random_state

    def build_tree(self, dataset, sample_weights, curr_depth=0):
        # recursive function to build the tree
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samp, num_feat = np.shape(X)

        # split until stopping conditions are met
        depth_ok = self.max_depth is None or curr_depth < self.max_depth
        if num_samp>=self.min_samp_split and depth_ok:
            # find the best split (pass sample_weights)
            best_split = self.get_best_split(dataset, num_samp, num_feat, sample_weights)
            # check if information gain is positive
            # (best_split is empty when no threshold separates the samples,
            #  e.g. all rows are identical; the node then becomes a leaf)
            if best_split.get('info_gain', -float('inf'))>0:
                # recur left
                left_subtree = self.build_tree(best_split['dataset_left'], best_split['weights_left'], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split['dataset_right'], best_split['weights_right'], curr_depth+1)
                # return decision node
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])
            
        # compute leaf node using weighted majority vote
        leaf_value = self.calculate_leaf_value(Y, sample_weights)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samp, num_feat, sample_weights):
        # dictionary to store best split
        best_split = {}
        max_info_gain = -float('inf')
        
        # loop over all the features
        for feature_index in range(num_feat):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split (including weight splits)
                dataset_left, dataset_right, weights_left, weights_right = self.split(dataset, feature_index, threshold, sample_weights)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y = dataset[:, -1]
                    left_y = dataset_left[:, -1]
                    right_y = dataset_right[:, -1]
                    # compute weighted information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, sample_weights, weights_left, weights_right, 'gini')
                    # update best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['weights_left'] = weights_left
                        best_split['weights_right'] = weights_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def information_gain(self, parent, l_child, r_child, parent_weights, l_weights, r_weights, mode='entropy'):
        # weighted proportions based on sum of weights, not count of samples
        total_weight = np.sum(parent_weights)
        if total_weight <= 0:
            return 0.0       # a node carrying no weight cannot be improved
        weight_l = np.sum(l_weights) / total_weight
        weight_r = np.sum(r_weights) / total_weight
        if mode=='gini':
            gain = self.gini_index(parent, parent_weights) - (weight_l*self.gini_index(l_child, l_weights) + weight_r*self.gini_index(r_child, r_weights))
        else:
            gain = self.entropy(parent, parent_weights) - (weight_l*self.entropy(l_child, l_weights) + weight_r*self.entropy(r_child, r_weights))
        return gain
    
    def split(self, dataset, feature_index, threshold, sample_weights):
        # split dataset and corresponding weights
        left_mask = dataset[:, feature_index] <= threshold
        right_mask = ~left_mask
        dataset_left = dataset[left_mask]
        dataset_right = dataset[right_mask]
        weights_left = sample_weights[left_mask]
        weights_right = sample_weights[right_mask]
        return dataset_left, dataset_right, weights_left, weights_right
    
    def entropy(self, y, weights):
        # weighted entropy calculation
        class_labels = np.unique(y)
        entropy = 0
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return 0.0       # empty / zero-weight node: treat as pure
        for cls in class_labels:
            # weighted proportion of this class
            p_cls = np.sum(weights[y == cls]) / total_weight
            if p_cls > 0:
                entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y, weights):
        # weighted gini index calculation
        class_labels = np.unique(y)
        gini = 0
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return 0.0       # empty / zero-weight node: treat as pure
        for cls in class_labels:
            # weighted proportion of this class
            p_cls = np.sum(weights[y == cls]) / total_weight
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, y, weights):
        # weighted majority vote for leaf prediction
        class_labels = np.unique(y)
        best_class = None
        max_weight = -float('inf')
        for cls in class_labels:
            # sum of weights for samples belonging to this class
            cls_weight = np.sum(weights[y == cls])
            if cls_weight > max_weight:
                max_weight = cls_weight
                best_class = cls
        return best_class
    
    def print_tree(self, tree=None, indent='    '):
        if tree is None:
            tree = self.root
            
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
        
    def fit(self, X, Y, sample_weights=None):
        X = np.asarray(X)
        Y = np.asarray(Y)
        # accept labels as a 1-D vector or as a column vector
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X.ndim != 2 or Y.shape != (X.shape[0], 1):
            raise ValueError("X must be 2-D (n, m) and Y must have n labels")
        if X.shape[0] == 0:
            raise ValueError("cannot fit a tree to an empty data set")
        # non-numeric labels are stored as objects so the features stay numeric
        if Y.dtype.kind not in 'biuf':
            Y = Y.astype(object)
        # if no weights provided, use uniform weights
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        sample_weights = np.asarray(sample_weights, dtype=float).ravel()
        if sample_weights.shape[0] != X.shape[0] or np.any(sample_weights < 0):
            raise ValueError("sample_weights must be n non-negative numbers")
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset, sample_weights)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)