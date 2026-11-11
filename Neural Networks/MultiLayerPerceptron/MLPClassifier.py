# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2026

Custom implementation of sklearn.neural_network.MLPClassifier.

A fully-connected (multi-layer perceptron) classifier trained by backprop, made
to match scikit-learn's MLPClassifier as faithfully as possible: same
constructor arguments and defaults, same parameterization (coefs_/intercepts_),
same Glorot-style initialization, the same forward/backprop equations, the same
SGD/Adam/L-BFGS solvers, and the same .fit / .predict / .predict_proba API.

How this differs in spirit from the tree ensembles
---------------------------------------------------
The random forest and gradient-boosting files split cleanly into
  base learner (a decision tree)  ->  ensemble (forest / boosted additive model).
A neural network does not decompose along those lines. Instead of many small
models combined by voting or addition, an MLP is ONE model made of a *stack of
stages*: an affine map, a nonlinearity, and (optionally) the structural pieces
covered in the reference files such as dropout or batch-norm. So the natural
modularization here is by *operation*:
    activations.py  -> nonlinearities, their derivatives, and the loss
    optimizers.py   -> how parameters move each step (SGD / Adam)
    mlp_utils.py    -> label encoding, batching, RNG, validation split
    MLPClassifier.py-> the network: init, forward, backprop, fit, predict

The math, briefly
-----------------
Forward pass, layer i = 0 .. L-1 with parameters (W_i, b_i):
    z_{i+1} = a_i W_i + b_i
    a_{i+1} = g(z_{i+1})           # g = hidden activation, except the last layer
    a_L     = out(z_L)            # out = softmax (multiclass) or logistic (binary)

Loss = cross-entropy(a_L, y) + (alpha/2) * sum||W||^2 / n_samples

Backprop. The combination of softmax + cross-entropy (or logistic + binary
cross-entropy) makes the output-layer error collapse to the clean form
    delta_L = a_L - y
and then for hidden layers
    delta_i = (delta_{i+1} W_i^T) * g'(a_i)
The parameter gradients are
    dW_i = (a_i^T delta_i + alpha * W_i) / n_samples
    db_i = mean(delta_i, axis=0)

@author: blain
"""

#%% Libraries
import warnings

import numpy as np
import scipy.optimize

from activations import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from optimizers import SGDOptimizer, AdamOptimizer
from mlp_utils import check_random_state, gen_batches, LabelBinarizer, train_test_split

_STOCHASTIC_SOLVERS = ('sgd', 'adam')


#%% L-BFGS parameter (un)packing helpers
def _pack(coefs_, intercepts_):
    # flatten all weight matrices and bias vectors into one 1-D vector
    return np.hstack([layer.ravel() for layer in coefs_ + intercepts_])


#%% MLP Classifier
class MLPClassifier():
    """
    Multi-layer Perceptron classifier.

    Parameters mirror sklearn.neural_network.MLPClassifier (same names and
    defaults), so an instance is a drop-in replacement:

        hidden_layer_sizes=(100,), activation='relu', solver='adam',
        alpha=0.0001, batch_size='auto', learning_rate='constant',
        learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
        random_state=None, tol=1e-4, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False,
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        n_iter_no_change=10, max_fun=15000

    Learned attributes after fit: classes_, n_outputs_, n_layers_,
    out_activation_, coefs_, intercepts_, loss_, loss_curve_, n_iter_, t_.
    """

    def __init__(self, hidden_layer_sizes=(100,), activation='relu',
                 solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None,
                 tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

        # classification always uses (multiclass / binary) log-loss
        self.loss = 'log_loss'

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _init_coef(self, fan_in, fan_out, dtype):
        # Glorot/Xavier-style uniform init, exactly as sklearn:
        #   bound = sqrt(factor / (fan_in + fan_out)), factor=6 (2 for logistic)
        factor = 6.0
        if self.activation == 'logistic':
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init

    def _initialize(self, y, layer_units, dtype):
        # bookkeeping
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]
        self.n_layers_ = len(layer_units)

        # choose the output activation from the label type
        if self._label_binarizer.y_type_ == 'multiclass':
            self.out_activation_ = 'softmax'
        else:
            self.out_activation_ = 'logistic'

        # initialize all weight matrices / bias vectors
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i],
                                                        layer_units[i + 1],
                                                        dtype)
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        # solver-specific state
        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
            else:
                self.best_loss_ = np.inf

    # ------------------------------------------------------------------ #
    # Forward / backward
    # ------------------------------------------------------------------ #
    def _forward_pass(self, activations):
        # activations[0] is the input; fill activations[1..L].
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = np.dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]
            # apply hidden nonlinearity to every layer except the output
            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])
        # output activation (softmax / logistic)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])
        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        # dW = (a^T delta + alpha*W) / n ;  db = mean(delta)
        coef_grads[layer] = np.dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= n_samples
        intercept_grads[layer] = np.mean(deltas[layer], axis=0)

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        n_samples = X.shape[0]

        # forward
        activations = self._forward_pass(activations)

        # loss = cross-entropy + L2 penalty (averaged over samples)
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / n_samples

        # output-layer error: softmax+CE (or logistic+BCE) -> a_L - y
        last = self.n_layers_ - 2
        deltas[last] = activations[-1] - y
        self._compute_loss_grad(last, n_samples, activations, deltas,
                                coef_grads, intercept_grads)

        # backpropagate through the hidden layers
        inplace_derivative = DERIVATIVES[self.activation]
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])
            self._compute_loss_grad(i - 1, n_samples, activations, deltas,
                                    coef_grads, intercept_grads)

        return loss, coef_grads, intercept_grads

    def _forward_pass_fast(self, X):
        # Inference-only forward pass that allocates fresh arrays (does not
        # touch the training activation buffers).
        X = np.asarray(X, dtype=np.float64)
        activation = X
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            activation = np.dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)
        return activation

    # ------------------------------------------------------------------ #
    # Fit drivers
    # ------------------------------------------------------------------ #
    def _validate_input(self, X, y, incremental=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        if not incremental or not hasattr(self, 'classes_'):
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_

        Y = self._label_binarizer.transform(y).astype(np.float64)
        return X, Y

    def _fit(self, X, y, incremental=False):
        # normalize hidden_layer_sizes into a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, '__iter__'):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError('hidden_layer_sizes must be > 0, got %s.' %
                             hidden_layer_sizes)
        if self.solver not in _STOCHASTIC_SOLVERS and self.solver != 'lbfgs':
            raise ValueError("solver must be 'sgd', 'adam' or 'lbfgs', got %r" %
                             self.solver)

        first_pass = (not hasattr(self, 'coefs_') or
                      (not self.warm_start and not incremental))

        X, y = self._validate_input(X, y, incremental)
        n_samples, n_features = X.shape
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        self.n_outputs_ = y.shape[1]

        # full layer sizes including input and output
        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        self._random_state = check_random_state(self.random_state)

        if first_pass:
            self._initialize(y, layer_units, X.dtype)

        # preallocate forward activations and backward gradient buffers
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)
        coef_grads = [np.empty((n_in, n_out)) for n_in, n_out in
                      zip(layer_units[:-1], layer_units[1:])]
        intercept_grads = [np.empty(n_out) for n_out in layer_units[1:]]

        # run the chosen solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(X, y, activations, deltas, coef_grads,
                                 intercept_grads, layer_units, incremental)
        elif self.solver == 'lbfgs':
            self._fit_lbfgs(X, y, activations, deltas, coef_grads,
                            intercept_grads, layer_units)

        return self

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads, layer_units, incremental):
        # one flat list of parameter arrays shared with the optimizer; the
        # optimizer mutates these arrays in place, which updates coefs_/intercepts_
        params = self.coefs_ + self.intercepts_

        if not incremental or not hasattr(self, '_optimizer'):
            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(
                    params, self.learning_rate_init, self.learning_rate,
                    self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(
                    params, self.learning_rate_init, self.beta_1,
                    self.beta_2, self.epsilon)

        # set up an early-stopping validation fold if requested
        early_stopping = self.early_stopping and not incremental
        if early_stopping:
            should_stratify = self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(
                X, y, random_state=self._random_state,
                test_size=self.validation_fraction, stratify=stratify)
            y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        # batch size
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = int(np.clip(self.batch_size, 1, n_samples))

        try:
            for it in range(self.max_iter):
                if self.shuffle:
                    self._random_state.shuffle(sample_idx)

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        X_batch = X[sample_idx[batch_slice]]
                        y_batch = y[sample_idx[batch_slice]]
                    else:
                        X_batch = X[batch_slice]
                        y_batch = y[batch_slice]

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_batch, y_batch, activations, deltas, coef_grads,
                        intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)

                    # one optimizer step over the whole parameter list
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]
                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)

                if self.verbose:
                    print('Iteration %d, loss = %.8f' % (self.n_iter_, self.loss_))

                # track convergence / best model
                self._update_no_improvement_count(early_stopping, X_val, y_val)
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    if early_stopping:
                        msg = ('Validation score did not improve more than '
                               'tol=%f for %d consecutive epochs.'
                               % (self.tol, self.n_iter_no_change))
                    else:
                        msg = ('Training loss did not improve more than tol=%f'
                               ' for %d consecutive epochs.'
                               % (self.tol, self.n_iter_no_change))
                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

            if self.n_iter_ == self.max_iter:
                warnings.warn(
                    'Stochastic Optimizer: Maximum iterations (%d) reached and '
                    'the optimization hasn\'t converged yet.' % self.max_iter)
        except KeyboardInterrupt:
            warnings.warn('Training interrupted by user.')

        # restore the best parameters found on the validation fold
        if early_stopping:
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if early_stopping:
            self.validation_scores_.append(self._score(X_val, y_val))
            if self.verbose:
                print('Validation score: %f' % self.validation_scores_[-1])

            last_valid_score = self.validation_scores_[-1]
            if last_valid_score < (self.best_validation_score_ + self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy() for i in self.intercepts_]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

    # ------------------------------------------------------------------ #
    # L-BFGS solver
    # ------------------------------------------------------------------ #
    def _unpack(self, packed_parameters):
        # inverse of _pack: copy slices of the flat vector back into coefs_/intercepts_
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)
            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]

    def _loss_grad_lbfgs(self, packed_coef_inter, X, y, activations, deltas,
                         coef_grads, intercept_grads):
        # scipy minimize callback: returns (loss, flat_gradient)
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads)
        grad = _pack(coef_grads, intercept_grads)
        return loss, grad

    def _fit_lbfgs(self, X, y, activations, deltas, coef_grads,
                   intercept_grads, layer_units):
        # build index tables describing where each layer lives in the flat vector
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]
            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        packed_coef_inter = _pack(self.coefs_, self.intercepts_)

        iprint = 1 if self.verbose else -1
        opt_res = scipy.optimize.minimize(
            self._loss_grad_lbfgs, packed_coef_inter, method='L-BFGS-B',
            jac=True,
            options={
                'maxfun': self.max_fun,
                'maxiter': self.max_iter,
                'iprint': iprint,
                'gtol': self.tol,
            },
            args=(X, y, activations, deltas, coef_grads, intercept_grads))

        self.n_iter_ = opt_res.nit
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)

        if not opt_res.success:
            warnings.warn('L-BFGS failed to converge (status=%d): %s'
                          % (opt_res.status, opt_res.message))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y."""
        return self._fit(X, y, incremental=False)

    def predict(self, X):
        """Predict class labels for samples in X."""
        y_pred = self._forward_pass_fast(X)
        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()
        return self._label_binarizer.inverse_transform(y_pred)

    def predict_proba(self, X):
        """Return the class membership probabilities for samples in X."""
        y_pred = self._forward_pass_fast(X)
        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()
        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

    def predict_log_proba(self, X):
        """Return the natural log of the class membership probabilities."""
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)

    def _score(self, X, y):
        # accuracy of label predictions (used internally for early stopping)
        return np.mean(self.predict(X) == np.asarray(y).ravel())

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == np.asarray(y).ravel())
