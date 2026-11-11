# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Stochastic optimizers for the multi-layer perceptron: plain/momentum/Nesterov
SGD and Adam. These are the MLP's counterpart to the role that bootstrap
sampling + majority vote play in the random forest, or that the additive
learning-rate-scaled updates play in gradient boosting: they define how the
model's parameters move from one step to the next.

Faithful-to-sklearn design:
  - Every optimizer owns its own state (velocities for SGD, first/second moment
    estimates for Adam) shaped like the parameter list it was handed.
  - update_params() applies the update IN PLACE via `param += update`. Because
    the parameter list shares the very same ndarray objects as the network's
    coefs_ and intercepts_, the in-place add is what actually mutates the model.
  - _get_updates() returns the *additive* step (already negated), so the caller
    never has to know the sign convention.

@author: blain
"""

#%% Libraries
import numpy as np


#%% Base optimizer
class BaseOptimizer():
    """
    Base class for optimizers. Holds the (possibly scheduled) learning rate
    and the generic in-place parameter-update loop.
    """
    def __init__(self, learning_rate_init=0.1):
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)

    def update_params(self, params, grads):
        # compute the per-parameter additive update and apply it in place
        updates = self._get_updates(grads)
        for param, update in zip(params, updates):
            param += update

    def iteration_ends(self, time_step):
        # hook called at the end of each epoch (used by SGD's lr schedules)
        pass

    def trigger_stopping(self, msg, verbose):
        # default behavior: when no improvement is seen, stop
        if verbose:
            print(msg + ' Stopping.')
        return True


#%% Stochastic gradient descent (with optional momentum / Nesterov)
class SGDOptimizer(BaseOptimizer):
    """
    Stochastic gradient descent with momentum, Nesterov momentum, and the
    'constant' / 'invscaling' / 'adaptive' learning-rate schedules.

    Parameters
    ----------
    params : list of ndarray
        The network parameters (coefs_ + intercepts_), used only to shape the
        velocity buffers.
    learning_rate_init : float
    lr_schedule : {'constant', 'adaptive', 'invscaling'}
    momentum : float in [0, 1]
    nesterov : bool
    power_t : float, exponent for the 'invscaling' schedule
    """
    def __init__(self, params, learning_rate_init=0.1, lr_schedule='constant',
                 momentum=0.9, nesterov=True, power_t=0.5):
        super().__init__(learning_rate_init)
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.velocities = [np.zeros_like(param) for param in params]

    def iteration_ends(self, time_step):
        # decay the learning rate for the 'invscaling' schedule
        if self.lr_schedule == 'invscaling':
            self.learning_rate = (float(self.learning_rate_init) /
                                  (time_step + 1) ** self.power_t)

    def trigger_stopping(self, msg, verbose):
        # 'adaptive' schedule shrinks the lr instead of stopping, until the lr
        # gets too small; every other schedule simply stops.
        if self.lr_schedule != 'adaptive':
            if verbose:
                print(msg + ' Stopping.')
            return True

        if self.learning_rate <= 1e-6:
            if verbose:
                print(msg + ' Learning rate too small. Stopping.')
            return True

        self.learning_rate /= 5.0
        if verbose:
            print(msg + ' Setting learning rate to %f' % self.learning_rate)
        return False

    def _get_updates(self, grads):
        # v <- momentum*v - lr*grad   (heavy-ball momentum)
        updates = [self.momentum * velocity - self.learning_rate * grad
                   for velocity, grad in zip(self.velocities, grads)]
        self.velocities = updates

        # Nesterov look-ahead: apply the momentum term a second time
        if self.nesterov:
            updates = [self.momentum * velocity - self.learning_rate * grad
                       for velocity, grad in zip(self.velocities, grads)]

        return updates


#%% Adam
class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer (Kingma & Ba, 2014). Keeps running first- and second-moment
    estimates of the gradient and folds the bias correction directly into an
    effective per-step learning rate, exactly as scikit-learn does.

    Parameters
    ----------
    params : list of ndarray
        Network parameters, used to shape the moment buffers.
    learning_rate_init : float
    beta_1, beta_2 : float in [0, 1)
        Exponential decay rates for the first and second moment estimates.
    epsilon : float
        Numerical stabilizer in the denominator.
    """
    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        super().__init__(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        self.t += 1
        # first moment (mean) and second moment (uncentered variance)
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        # bias-corrected effective learning rate
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates
