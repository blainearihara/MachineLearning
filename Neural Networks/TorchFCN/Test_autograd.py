# -*- coding: utf-8 -*-
"""
Unit tests for the mytorch autograd engine.

The decisive test for an autograd engine is a NUMERICAL GRADIENT CHECK: compare
the analytic gradients produced by loss.backward() against finite-difference
estimates of the same gradients. If they agree, the engine's chain-rule
plumbing is correct and any network composed from these layers will train
correctly. We check three architectures:

  1. Linear -> ReLU -> Linear            (the core, no bn/dropout)
  2. Linear -> BatchNorm1d -> ReLU -> Linear
  3. Linear -> ReLU -> Dropout -> Linear (mask held fixed via a seed)

Checks run in float64 for finite-difference precision. We also sanity-check
BatchNorm's train/eval statistics behavior.
"""

#%% Libraries
import numpy as np

import mytorch as torch
import mytorch.nn as nn


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


#%% Generic numerical gradient check
def numeric_grad_check(build_model, label, mask_seed=0):
    torch.set_default_dtype(np.float64)   # finite differences need float64
    np.random.seed(0)
    N, D, C = 8, 6, 4
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=N)

    model = build_model(D, C)
    model.train()
    criterion = nn.CrossEntropyLoss()

    def loss_value():
        # reseed so any dropout mask is identical across +h / -h evaluations
        np.random.seed(mask_seed)
        return criterion(model(torch.tensor(X)), y)

    # analytic gradients from backward()
    loss = loss_value()
    model.zero_grad()
    loss.backward()
    analytic = {id(p): p.grad.copy() for p in model.parameters()}

    # finite-difference gradients
    h = 1e-5
    worst = 0.0
    for p in model.parameters():
        num = np.zeros_like(p.data)
        it = np.nditer(p.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old = p.data[idx]
            p.data[idx] = old + h
            lp = loss_value().data
            p.data[idx] = old - h
            lm = loss_value().data
            p.data[idx] = old
            num[idx] = (lp - lm) / (2 * h)
            it.iternext()
        worst = max(worst, rel_error(num, analytic[id(p)]))

    torch.set_default_dtype(np.float32)
    status = 'OK' if worst < 1e-6 else 'CHECK'
    print(f"  {label:42s} max rel grad error = {worst:.3e}  ({status})")
    return worst


#%% Model builders
def build_plain(D, C):
    return nn.Sequential(
        nn.Linear(D, 10), nn.ReLU(), nn.Linear(10, C))


def build_batchnorm(D, C):
    return nn.Sequential(
        nn.Linear(D, 10), nn.BatchNorm1d(10), nn.ReLU(), nn.Linear(10, C))


def build_dropout(D, C):
    return nn.Sequential(
        nn.Linear(D, 10), nn.ReLU(), nn.Dropout(0.3), nn.Linear(10, C))


#%% BatchNorm statistics behavior
def batchnorm_behavior():
    torch.set_default_dtype(np.float32)
    np.random.seed(0)
    bn = nn.BatchNorm1d(5)
    bn.train()
    X = torch.tensor(np.random.randn(200, 5) * 3 + 7)   # mean 7, std 3
    out = bn(X).data
    print("  train-mode output mean (~0):", np.round(out.mean(axis=0), 4))
    print("  train-mode output std  (~1):", np.round(out.std(axis=0), 4))
    print("  running_mean moved toward 7:", np.round(bn.running_mean, 3))


#%% Run
if __name__ == "__main__":
    print("=" * 72)
    print("NUMERICAL GRADIENT CHECK (analytic backward vs finite differences)")
    print("=" * 72)
    numeric_grad_check(build_plain, "Linear -> ReLU -> Linear")
    numeric_grad_check(build_batchnorm, "Linear -> BatchNorm1d -> ReLU -> Linear")
    numeric_grad_check(build_dropout, "Linear -> ReLU -> Dropout -> Linear")
    print()
    print("=" * 72)
    print("BATCHNORM TRAIN-MODE STATISTICS")
    print("=" * 72)
    batchnorm_behavior()
