# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 2026

mytorch.nn — neural-network building blocks layered on top of the autograd
engine, mirroring torch.nn.

Each layer is a Module: it owns Parameters (tensors with requires_grad=True),
exposes them through .parameters(), and defines forward(). Because every layer's
forward() is written purely in terms of autograd Tensor operations, none of them
needs a hand-written backward — the engine differentiates the whole network
automatically. That is the key payoff over the sklearn clone: dropout and
batch-norm are just two more Modules whose forward() composes engine ops, and
they slot into any architecture for free.

Provided:
  Module          - base class: parameter/sub-module registration, train()/eval()
  Parameter       - a Tensor that is a learnable parameter
  Linear          - y = x W^T + b      (weight stored (out, in) like PyTorch)
  ReLU            - max(0, x)
  Dropout         - inverted dropout; p is the DROP probability (PyTorch convention)
  BatchNorm1d     - batch stats in train, running stats in eval
  Sequential      - run a list of modules in order
  CrossEntropyLoss- log-softmax + NLL on raw logits and integer class targets

@author: blain
"""

#%% Libraries
import math

import numpy as np

from . import engine
from .engine import Tensor


#%% Parameter
class Parameter(Tensor):
    """A Tensor marked as a learnable parameter (always requires grad)."""
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


#%% Module base class
class Module():
    """
    Base class for all layers/models. Assigning a Parameter or a Module as an
    attribute auto-registers it, so .parameters() can gather everything and
    train()/eval() can recurse through the whole tree.
    """
    def __init__(self):
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, 'training', True)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        seen = set()
        out = []
        for p in self._parameters.values():
            if p is not None and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        for m in self._modules.values():
            if m is not None:
                for p in m.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        object.__setattr__(self, 'training', mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


#%% Linear (fully-connected) layer
class Linear(Module):
    """
    Applies y = x W^T + b. Weight is stored with shape (out_features, in_features)
    and initialized with PyTorch's default: U(-k, k) with k = 1/sqrt(in_features)
    (Kaiming-uniform with a=sqrt(5)); bias uses the same bound.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0 / math.sqrt(in_features)
        w = np.random.uniform(-bound, bound, size=(out_features, in_features))
        self.weight = Parameter(w)
        if bias:
            b = np.random.uniform(-bound, bound, size=(out_features,))
            self.bias = Parameter(b)
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


#%% ReLU
class ReLU(Module):
    def forward(self, x):
        return x.relu()


#%% Dropout
class Dropout(Module):
    """
    Inverted dropout. `p` is the probability of zeroing a unit (PyTorch
    convention). At train time, surviving units are scaled by 1/(1-p) so the
    expected activation is unchanged; at eval time dropout is the identity.

    (Note: the CS231n-style reference file treats p as the KEEP probability —
    this implementation follows PyTorch and treats p as the DROP probability.)
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        mask = (np.random.rand(*x.shape) >= self.p).astype(x.data.dtype) / keep
        return x * Tensor(mask)   # mask is a constant -> grad flows straight through


#%% BatchNorm1d
class BatchNorm1d(Module):
    """
    Batch normalization for (N, C) inputs.

    Train: normalize with the per-feature batch mean/variance, then scale and
    shift by the learnable gamma (weight) and beta (bias). The running
    statistics are updated as running = (1 - momentum)*running + momentum*batch,
    using the unbiased batch variance (PyTorch convention).
    Eval: normalize with the stored running statistics.

    The normalization is written with plain engine ops (mean, subtract, pow,
    multiply), so its backward pass — including the gradient that flows through
    the batch statistics — is produced automatically by autograd.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features))    # gamma
        self.bias = Parameter(np.zeros(num_features))      # beta
        # running buffers (not Parameters: no grad, not touched by the optimizer)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=0, keepdims=True)   # biased var

            # update running statistics with the unbiased batch variance
            N = x.shape[0]
            batch_mean = mean.data.ravel()
            batch_var = var.data.ravel()
            unbiased = batch_var * N / (N - 1) if N > 1 else batch_var
            self.running_mean = ((1 - self.momentum) * self.running_mean +
                                 self.momentum * batch_mean)
            self.running_var = ((1 - self.momentum) * self.running_var +
                                self.momentum * unbiased)

            x_hat = (x - mean) * ((var + self.eps) ** -0.5)
        else:
            mean = Tensor(self.running_mean.reshape(1, -1))
            inv_std = Tensor((1.0 / np.sqrt(self.running_var + self.eps)).reshape(1, -1))
            x_hat = (x - mean) * inv_std

        return x_hat * self.weight + self.bias


#%% Sequential container
class Sequential(Module):
    """Chain modules together; forward() runs them in the given order."""
    def __init__(self, *layers):
        super().__init__()
        self._seq = list(layers)
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer

    def forward(self, x):
        for layer in self._seq:
            x = layer(x)
        return x


#%% Cross-entropy loss (functional + module)
def cross_entropy(logits, target):
    """
    Mean cross-entropy over a batch from raw logits and integer class targets.
    Computed with a numerically stable log-softmax; the backward is the standard
    fused form (softmax - onehot)/N, attached as this node's local gradient.
    """
    x = logits.data
    N = x.shape[0]

    # stable log-softmax
    x_max = x.max(axis=1, keepdims=True)
    z = x - x_max
    exp = np.exp(z)
    sum_exp = exp.sum(axis=1, keepdims=True)
    log_softmax = z - np.log(sum_exp)

    target = np.asarray(target).astype(int)
    loss_val = -log_softmax[np.arange(N), target].mean()

    out = Tensor(loss_val, _children=(logits,), _op='cross_entropy')
    out.requires_grad = logits.requires_grad and engine.is_grad_enabled()

    def _backward():
        softmax = exp / sum_exp
        grad = softmax.copy()
        grad[np.arange(N), target] -= 1.0
        grad = grad / N
        engine._accumulate(logits, grad * out.grad)
    if out.requires_grad:
        out._backward = _backward
    return out


class CrossEntropyLoss(Module):
    """Cross-entropy loss module (logits in, scalar loss out)."""
    def forward(self, logits, target):
        return cross_entropy(logits, target)
