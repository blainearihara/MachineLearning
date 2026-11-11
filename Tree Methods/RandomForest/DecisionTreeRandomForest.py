# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 12:05:11 2025

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

#%% Modified Decision Tree for Random Forest
class DecisionTreeClassifier():
    def __init__(self, min_samp_split=2, max_depth=2, max_features=None, random_state=None):
        # initialize root of the tree
        self.root = None
        
        # stopping conditions (max_depth=None means no depth limit)
        self.min_samp_split = min_samp_split
        self.max_depth = max_depth
        self.max_features = max_features  # for random feature selection
        self.random_state = random_state

        # private random generator (the global np.random state is never touched);
        # random_state may be None, an int seed, or a np.random.Generator
        self.rng = np.random.default_rng(self.random_state)

    def build_tree(self, dataset, curr_depth=0):
        # recursive function to build the tree
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samp, num_feat = np.shape(X)

        # split until stopping conditions are met
        depth_ok = self.max_depth is None or curr_depth < self.max_depth
        if num_samp>=self.min_samp_split and depth_ok:
            # find the best split
            best_split = self.get_best_split(dataset, num_samp, num_feat)
            # check if information gain is positive
            # (best_split is empty when no threshold separates the samples, e.g. a node
            #  made only of copies of one bootstrapped row; the node then becomes a leaf)
            if best_split.get('info_gain', -float('inf'))>0:
                # recur left
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)
                # return decision node
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])
            
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samp, num_feat):
        # dictionary to store best split
        best_split = {}
        max_info_gain = -float('inf')
        
        # determine features to consider
        feature_indices = list(self.get_random_features(num_feat))
        # if none of the sampled features can split this node (all constant here),
        # continue with the remaining features, as scikit-learn does
        remaining = [j for j in range(num_feat) if j not in set(feature_indices)]

        for candidates in (feature_indices, remaining):
            # loop over selected features
            for feature_index in candidates:
                feature_values = dataset[:, feature_index]
                possible_thresholds = np.unique(feature_values)
                # loop over all the feature values present in the data
                for threshold in possible_thresholds:
                    # get current split
                    dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                    # check if childs are not null
                    if len(dataset_left)>0 and len(dataset_right)>0:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        # compute information gain
                        curr_info_gain = self.information_gain(y, left_y, right_y, 'gini')
                        # update best split if needed
                        if curr_info_gain>max_info_gain:
                            best_split['feature_index'] = feature_index
                            best_split['threshold'] = threshold
                            best_split['dataset_left'] = dataset_left
                            best_split['dataset_right'] = dataset_right
                            best_split['info_gain'] = curr_info_gain
                            max_info_gain = curr_info_gain
            if best_split:
                break

        # return best split
        return best_split

    def get_random_features(self, num_feat):
        # select random subset of features
        mf = self.max_features
        if mf is None:
            return range(num_feat)
        elif mf == 'sqrt':
            max_feat = int(np.sqrt(num_feat))
        elif mf == 'log2':
            max_feat = int(np.log2(num_feat))
        elif isinstance(mf, (int, np.integer)) and not isinstance(mf, bool):
            max_feat = int(mf)
        elif isinstance(mf, (float, np.floating)) and 0 < mf <= 1:
            # a float is a fraction of the features (as in scikit-learn)
            max_feat = int(mf * num_feat)
        else:
            raise ValueError("max_features must be None, 'sqrt', 'log2', a positive int, or a float in (0, 1]")

        # always search at least one and at most num_feat features
        max_feat = max(1, min(max_feat, num_feat))
        return self.rng.choice(num_feat, max_feat, replace=False)
    
    def information_gain(self, parent, l_child, r_child, mode='entropy'):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=='gini':
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, y):
        # majority class; ties go to the smallest label (independent of sample order)
        class_labels, counts = np.unique(y, return_counts=True)
        return class_labels[np.argmax(counts)]

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
        
    def fit(self, X, Y):
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
        # restart the generator so that refitting with the same random_state is reproducible
        self.rng = np.random.default_rng(self.random_state)
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

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
        
        