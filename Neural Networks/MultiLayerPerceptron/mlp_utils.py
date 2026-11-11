# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Small, self-contained reimplementations of the scikit-learn utilities that the
MLP relies on, so the classifier can stand on its own without importing
sklearn internals:

  - check_random_state : normalize a seed/None/RandomState into a RandomState
  - gen_batches        : yield slice objects that walk a dataset in minibatches
  - LabelBinarizer     : one-hot for multiclass, single 0/1 column for binary
  - train_test_split   : (only what early-stopping needs) optional stratified split

These mirror sklearn's public behavior closely enough that the MLP's training
loop, label handling, and early-stopping path behave the same way.

@author: blain
"""

#%% Libraries
import numpy as np


#%% Random state
def check_random_state(seed):
    # Turn seed into a np.random.RandomState instance (sklearn semantics).
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy RandomState instance' % seed)


#%% Minibatch slicing
def gen_batches(n, batch_size):
    # Yield consecutive slices of length batch_size that cover range(n);
    # the final slice may be shorter. Mirrors sklearn.utils.gen_batches
    # with min_batch_size=0.
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


#%% Label binarization
class LabelBinarizer():
    """
    Encode class labels for a neural-network output layer.

    - binary (2 classes)      -> a single column of 0/1 (paired with logistic)
    - multiclass (3+ classes) -> one-hot columns          (paired with softmax)

    inverse_transform maps the network's probability output back to class
    labels: a 0.5 threshold for the binary case, argmax for multiclass.
    """
    def fit(self, y):
        self.classes_ = np.unique(np.asarray(y).ravel())
        n_classes = len(self.classes_)
        if n_classes <= 2:
            self.y_type_ = 'binary'
        else:
            self.y_type_ = 'multiclass'
        # map label value -> column index
        self._class_to_index = {cls: i for i, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        y = np.asarray(y).ravel()
        indices = np.array([self._class_to_index[val] for val in y])
        if self.y_type_ == 'binary':
            # positive class is classes_[1]; column is 1 there, else 0
            Y = (indices == 1).astype(np.float64).reshape(-1, 1)
        else:
            Y = np.zeros((len(y), len(self.classes_)), dtype=np.float64)
            Y[np.arange(len(y)), indices] = 1.0
        return Y

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, Y):
        Y = np.asarray(Y)
        if self.y_type_ == 'binary':
            scores = Y if Y.ndim == 1 else Y[:, 0]
            indices = (scores > 0.5).astype(int)
        else:
            indices = np.argmax(Y, axis=1)
        return self.classes_[indices]


#%% Train/validation split (used only for early stopping)
def train_test_split(X, y, test_size=0.1, random_state=None, stratify=None):
    # A trimmed-down split. Supports an optional stratify array so that, for
    # single-output classification, the validation fold preserves class ratios
    # (this is what sklearn's MLP does for early stopping).
    rng = check_random_state(random_state)
    n = X.shape[0]
    indices = np.arange(n)

    if stratify is not None:
        strat = np.asarray(stratify).ravel()
        train_idx, test_idx = [], []
        for cls in np.unique(strat):
            cls_idx = indices[strat == cls]
            rng.shuffle(cls_idx)
            n_cls_test = int(round(test_size * len(cls_idx)))
            test_idx.extend(cls_idx[:n_cls_test])
            train_idx.extend(cls_idx[n_cls_test:])
        train_idx = np.array(train_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        perm = rng.permutation(n)
        n_test = int(round(test_size * n))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
