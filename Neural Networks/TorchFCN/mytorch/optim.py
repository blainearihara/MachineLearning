# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 2026

mytorch.optim — optimizers, mirroring torch.optim.

An optimizer is handed `model.parameters()` once; each call to step() reads the
.grad that backward() deposited on every parameter and updates .data in place.
zero_grad() clears the grads between iterations (set to None by default, the
PyTorch behavior). This is the same SGD / Adam math as the sklearn-clone
optimizers, but here it consumes gradients produced by the autograd engine
rather than a hand-written backprop routine.

@author: blain
"""

#%% Libraries
import numpy as np


#%% Base optimizer
class Optimizer():
    def __init__(self, params, defaults):
        params = list(params)
        self.param_groups = [dict(params=params, **defaults)]
        self.state = {}     # id(param) -> per-parameter state dict

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad = np.zeros_like(p.data)


#%% Stochastic gradient descent
class SGD(Optimizer):
    """SGD with optional momentum, Nesterov momentum, and weight decay."""
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0,
                 nesterov=False):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, nesterov=nesterov))

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p.data
                if momentum != 0:
                    st = self.state.setdefault(id(p), {})
                    buf = st.get('momentum_buffer')
                    if buf is None:
                        buf = g.copy()
                    else:
                        buf = momentum * buf + g
                    st['momentum_buffer'] = buf
                    if nesterov:
                        g = g + momentum * buf
                    else:
                        g = buf
                p.data = p.data - lr * g


#%% Adam
class Adam(Optimizer):
    """
    Adam optimizer. Keeps bias-corrected first/second moment estimates and
    applies p <- p - lr * mhat / (sqrt(vhat) + eps), with eps OUTSIDE the sqrt
    (the PyTorch convention).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p.data

                st = self.state.setdefault(id(p), {})
                if not st:
                    st['step'] = 0
                    st['exp_avg'] = np.zeros_like(p.data)
                    st['exp_avg_sq'] = np.zeros_like(p.data)

                st['step'] += 1
                t = st['step']
                m = st['exp_avg']
                v = st['exp_avg_sq']

                # in-place moment updates
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * (g * g)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)
