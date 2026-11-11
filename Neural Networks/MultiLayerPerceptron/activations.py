# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Activation functions, their derivatives, and the classification loss
functions used by the multi-layer perceptron.

This module is the MLP analogue of the splitting-criterion helpers in the
tree code (gini_index / entropy / variance_reduction). Where a decision tree
needs an impurity measure and a leaf rule, a neural network needs:
  - a nonlinearity to place between affine layers (identity/logistic/tanh/relu),
  - an *output* activation that turns raw scores into probabilities
    (logistic for binary, softmax for multiclass),
  - the derivative of each hidden nonlinearity for backpropagation, and
  - a loss (log-loss / cross-entropy) to score the predicted probabilities.

Design note (faithful to scikit-learn):
  All activations operate IN PLACE on their argument and all hidden
  derivatives operate IN PLACE on the incoming `delta`. This matches
  sklearn's _base.py and lets the forward/backward passes avoid allocating
  fresh arrays for every layer.

  The hidden derivatives take the *post-activation* output Z (not the
  pre-activation input), because for logistic/tanh/relu the derivative is
  cheapest to express in terms of the activation's own output:
      logistic'(z) = Z * (1 - Z)
      tanh'(z)     = 1 - Z**2
      relu'(z)     = 1 where Z > 0 else 0   (Z == 0 marks the dead units)

@author: blain
"""

#%% Libraries
import numpy as np
from scipy.special import expit, xlogy


#%% Output / hidden activation functions (in-place)
def inplace_identity(X):
    # f(x) = x. Used as the output activation for regression; a no-op here.
    pass


def inplace_logistic(X):
    # f(x) = 1 / (1 + exp(-x)). expit is the numerically stable sigmoid.
    expit(X, out=X)


def inplace_tanh(X):
    # f(x) = tanh(x)
    np.tanh(X, out=X)


def inplace_relu(X):
    # f(x) = max(0, x)
    np.maximum(X, 0, out=X)


def inplace_softmax(X):
    # f(x)_k = exp(x_k) / sum_j exp(x_j), computed row-wise.
    # Subtract the row max first for numerical stability (prevents overflow).
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]


# name -> activation function (mirrors sklearn ACTIVATIONS)
ACTIVATIONS = {
    'identity': inplace_identity,
    'logistic': inplace_logistic,
    'tanh': inplace_tanh,
    'relu': inplace_relu,
    'softmax': inplace_softmax,
}


#%% Hidden-layer derivatives (in-place on delta; Z is the post-activation output)
def inplace_identity_derivative(Z, delta):
    # derivative is 1 everywhere, so delta is unchanged
    pass


def inplace_logistic_derivative(Z, delta):
    # logistic'(z) = Z * (1 - Z)
    delta *= Z
    delta *= (1 - Z)


def inplace_tanh_derivative(Z, delta):
    # tanh'(z) = 1 - Z**2
    delta *= (1 - Z ** 2)


def inplace_relu_derivative(Z, delta):
    # relu'(z) = 0 where the unit was clipped to zero, 1 otherwise
    delta[Z == 0] = 0


# name -> derivative function (mirrors sklearn DERIVATIVES)
DERIVATIVES = {
    'identity': inplace_identity_derivative,
    'logistic': inplace_logistic_derivative,
    'tanh': inplace_tanh_derivative,
    'relu': inplace_relu_derivative,
}


#%% Classification losses (cross-entropy / log-loss)
def binary_log_loss(y_true, y_prob):
    # Binary cross-entropy averaged over samples. Used when the output
    # activation is logistic (binary classification, single output unit).
    #   L = -mean( y*log(p) + (1-y)*log(1-p) )
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -(xlogy(y_true, y_prob).sum() +
             xlogy(1 - y_true, 1 - y_prob).sum()) / y_prob.shape[0]


def log_loss(y_true, y_prob):
    # Multiclass cross-entropy averaged over samples. Used when the output
    # activation is softmax. If a single-column probability sneaks in, expand
    # it to two columns so the formula is well defined (matches sklearn).
    #   L = -mean( sum_k y_k * log(p_k) )
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)
    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)
    return -xlogy(y_true, y_prob).sum() / y_prob.shape[0]


# name -> loss function (mirrors sklearn LOSS_FUNCTIONS)
LOSS_FUNCTIONS = {
    'log_loss': log_loss,
    'binary_log_loss': binary_log_loss,
}
