# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 00:07:01 2025

@author: blain
"""

#%% Libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree  # For visualization

from GetKaggleDataset import get_dataset
from DecisionTree import DecisionTreeClassifier as dtc
from sklearn.tree import DecisionTreeClassifier  # Import sklearn version
from sklearn import tree  # For visualization

from sklearn.model_selection import GridSearchCV

#%% Main
if __name__ == "__main__":
    
    df = get_dataset("arshid/iris-flower-dataset", "Iris.csv")
    
    #%% Inspect data
    print(df.head())    # first 5 rows
    print(df.sample(5)) # 5 random rows
    print(f'Shape: \n{df.shape}')
    print(f'Info: \n{df.info}')
    print(f'Columns: \n{df.columns}')
    
    # Grab specific columns
    col_names = ['sepal_length', 'sepal_width','petal_length','petal_width', 'species']
    data = df[col_names]
    
    # Train-Test Split4
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    
    ''' Use custom made Decision Tree Classifier Model '''
    # Fit the model
    classifier = dtc(min_samp_split=3, max_depth=3)
    classifier.fit(X_train,Y_train)
    classifier.print_tree()
    
    # Test the model
    Y_pred = classifier.predict(X_test)
    test_score = accuracy_score(Y_test, Y_pred)
    print(f'Test score: {test_score:.4f}')
    
    
    #%% Using Scikit-learn DecisionTreeClassifier
    ''' Use SciKit Learn Decision Tree Classifier '''
    # Create the classifier with similar parameters
    sklearn_classifier = DecisionTreeClassifier(
        min_samples_split=3,    # Equivalent to min_samp_split
        max_depth=3,           # Same as your custom implementation
        random_state=42,       # For reproducible results
        criterion='gini'       # Can also use 'entropy'
    )
    # Fit the model
    sklearn_classifier.fit(X_train, Y_train)
    # Make predictions
    Y_pred_sklearn = sklearn_classifier.predict(X_test)
    # Evaluate the model
    sklearn_accuracy = accuracy_score(Y_test, Y_pred_sklearn)
    print(f'Scikit-learn Decision Tree Accuracy: {sklearn_accuracy:.4f}')
    # Additional evaluation metrics
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred_sklearn))
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred_sklearn))

    # Visualize the decision tree
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        sklearn_classifier,
        feature_names=feature_names,
        class_names=np.unique(Y),
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree Visualization")
    plt.show()
    
    
    ''' Perform Grid-Search for Hyperparameter Tuning '''
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, Y_train)
    print(f'\nBest parameters: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_:.4f}')
    
    # Use the best model
    best_classifier = grid_search.best_estimator_
    Y_pred_best = best_classifier.predict(X_test)
    best_accuracy = accuracy_score(Y_test, Y_pred_best)
    print(f'Best model test accuracy: {best_accuracy:.4f}')
    pass

 