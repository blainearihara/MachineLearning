# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 12:07:18 2025

@author: blain
"""
#%% Libraries
import os
import sys
import pandas as pd
import numpy as np

from DecisionTreeRandomForest import DecisionTreeClassifier as DecisionTreeClassifier

#%% Random Forest Classifier
class RandomForestClassifier():
    def __init__(self, n_estimators=10, min_samp_split=2, max_depth=2, max_features='sqrt', bootstrap=True, random_state=None):
        # forest parameters
        self.n_estimators = n_estimators
        self.min_samp_split = min_samp_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be at least 1")

        # initialize forest
        self.trees = []
        self.bootstrap_indices_ = []   # row indices each tree was trained on

        # private random generator for reproducibility (the global np.random state
        # is never touched); random_state may be None, an int seed, or a Generator
        self.rng = np.random.default_rng(self.random_state)

    def bootstrap_sample(self, X, Y):
        # create bootstrap sample (n draws with replacement)
        num_samples = X.shape[0]
        indices = self.rng.choice(num_samples, num_samples, replace=True)
        return X[indices], Y[indices]

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        # clear any existing trees and restart the generator, so that fitting the
        # same object twice with the same random_state gives the same forest
        self.trees = []
        self.bootstrap_indices_ = []
        self.rng = np.random.default_rng(self.random_state)
        self.n_features_ = X.shape[1]

        # build each tree in the forest
        for i in range(self.n_estimators):
            # create bootstrap sample if enabled (indices recorded for inspection)
            if self.bootstrap:
                indices = self.rng.choice(X.shape[0], X.shape[0], replace=True)
            else:
                indices = np.arange(X.shape[0])
            X_sample, Y_sample = X[indices], Y[indices]
            self.bootstrap_indices_.append(indices)

            # create and train tree, with its own seed drawn from the forest's generator
            tree = DecisionTreeClassifier(
                min_samp_split=self.min_samp_split,
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=int(self.rng.integers(2**31 - 1))
            )
            tree.fit(X_sample, Y_sample)

            # add tree to forest
            self.trees.append(tree)

    def predict(self, X):
        if not self.trees:
            raise ValueError("the forest has no trees; call fit first")
        # get predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            tree_predictions.append(predictions)
        
        # transpose to get predictions per sample
        tree_predictions = np.array(tree_predictions).T
        
        # majority vote for each sample
        final_predictions = []
        for sample_preds in tree_predictions:
            final_predictions.append(self.majority_vote(sample_preds))
        
        return final_predictions
    
    def majority_vote(self, predictions):
        # return most common prediction; ties go to the smallest label (deterministic)
        class_labels, counts = np.unique(np.asarray(predictions), return_counts=True)
        return class_labels[np.argmax(counts)]

    def get_feature_importance(self, feature_names=None):
        # calculate feature importances across all trees
        # (sum of the stored, unweighted info_gain of every split on each feature,
        #  averaged over trees; feature_names is optional)
        if not self.trees:
            return None

        # number of features: from feature_names if given, else from the training data
        num_features = len(feature_names) if feature_names is not None else getattr(self, 'n_features_', 0)
        if num_features == 0:
            return None
        
        importances = np.zeros(num_features)
        
        # aggregate importances from all trees
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
        
        # add importance of current node
        if node.info_gain and node.feature_index < num_features:
            importances[node.feature_index] += node.info_gain
        
        # add importances from children
        importances += self.calculate_tree_importance(node.left, num_features)
        importances += self.calculate_tree_importance(node.right, num_features)
        
        return importances
    
    
    