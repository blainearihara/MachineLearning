# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 2026

mytorch.engine — a small reverse-mode automatic differentiation engine.

This is the core that makes `loss.backward()` work, i.e. the piece that the
scikit-learn clone did not have. Where the sklearn MLP hard-coded one fixed
backprop routine, here every operation on a Tensor records itself on a dynamic
computational graph, and backward() walks that graph in reverse topological
order applying the chain rule. That is exactly PyTorch's mechanism, so models
are built by composing operations and trained with the standard
    optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
loop.

A Tensor holds:
  data        : the numpy array of values
  grad        : the accumulated gradient dL/d(this tensor), same shape as data
  requires_grad: whether gradients should flow into this tensor
  _backward   : a closure that pushes grad from this node to its inputs
  _prev       : the input tensors that produced this one (graph edges)

The one genuinely fiddly part of any such engine is broadcasting-aware
backward: when an op broadcasts (e.g. adding a length-C bias to an N x C
matrix), the gradient flowing back to the smaller operand must be summed over
the broadcast dimensions to restore its original shape. That is handled in one
place, _unbroadcast(), and reused by every elementwise op.

@author: blain
"""

#%% Libraries
import numpy as np

# Default floating dtype for new tensors (PyTorch defaults to float32).
_default_dtype = np.float32

# Global switch toggled by `with no_grad():` to skip graph construction.
_grad_enabled = True

# Public dtype aliases
float32 = np.float32
float64 = np.float64


#%% Engine-level helpers
def is_grad_enabled():
    return _grad_enabled


def set_default_dtype(dtype):
    """Set the dtype used for newly created tensors (e.g. float64 for grad checks)."""
    global _default_dtype
    _default_dtype = dtype


def manual_seed(seed):
    """Seed the (numpy) RNG used for initialization, shuffling and dropout."""
    np.random.seed(seed)


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing numpy broadcasting in the backward pass."""
    if grad.shape == shape:
        return grad
    # collapse extra leading dimensions that broadcasting added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # collapse dimensions that were size-1 in the original operand
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _accumulate(t, grad):
    """Accumulate `grad` into tensor t.grad, respecting broadcasting and requires_grad."""
    if not t.requires_grad:
        return
    grad = _unbroadcast(np.asarray(grad), t.data.shape)
    if t.grad is None:
        t.grad = grad.astype(t.data.dtype, copy=True)
    else:
        t.grad = t.grad + grad.astype(t.data.dtype, copy=False)


#%% no_grad context manager
class no_grad():
    """Disable graph building inside a `with` block (used for inference/eval)."""
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev
        return False


#%% Tensor
class Tensor():
    """An n-dimensional array that records operations for reverse-mode autodiff."""

    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=_default_dtype)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._backward = lambda: None     # local backward; no-op for leaves
        self._prev = set(_children)        # parents in the graph
        self._op = _op

    # ---- properties / conveniences (mirroring the torch.Tensor surface) ----
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.t()

    def item(self):
        return self.data.item()

    def numpy(self):
        return self.data

    def argmax(self, axis=None):
        return self.data.argmax(axis=axis)

    def detach(self):
        # a tensor sharing the same values but cut off from the graph
        return Tensor(self.data, requires_grad=False)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------ #
    # Elementwise / linear-algebra ops, each with its local backward rule
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        out.requires_grad = (self.requires_grad or other.requires_grad) and _grad_enabled

        def _backward():
            _accumulate(self, out.grad)
            _accumulate(other, out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out
    __radd__ = __add__

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, _children=(self, other), _op='-')
        out.requires_grad = (self.requires_grad or other.requires_grad) and _grad_enabled

        def _backward():
            _accumulate(self, out.grad)
            _accumulate(other, -out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        out.requires_grad = (self.requires_grad or other.requires_grad) and _grad_enabled

        def _backward():
            _accumulate(self, out.grad * other.data)
            _accumulate(other, out.grad * self.data)
        if out.requires_grad:
            out._backward = _backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _children=(self, other), _op='/')
        out.requires_grad = (self.requires_grad or other.requires_grad) and _grad_enabled

        def _backward():
            _accumulate(self, out.grad / other.data)
            _accumulate(other, -out.grad * self.data / (other.data ** 2))
        if out.requires_grad:
            out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)

    def __neg__(self):
        return self * -1.0

    def __pow__(self, p):
        assert isinstance(p, (int, float)), 'only scalar powers are supported'
        out = Tensor(self.data ** p, _children=(self,), _op=f'**{p}')
        out.requires_grad = self.requires_grad and _grad_enabled

        def _backward():
            _accumulate(self, (p * self.data ** (p - 1)) * out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')
        out.requires_grad = (self.requires_grad or other.requires_grad) and _grad_enabled

        def _backward():
            _accumulate(self, out.grad @ np.swapaxes(other.data, -1, -2))
            _accumulate(other, np.swapaxes(self.data, -1, -2) @ out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def t(self):
        out = Tensor(self.data.T, _children=(self,), _op='t')
        out.requires_grad = self.requires_grad and _grad_enabled

        def _backward():
            _accumulate(self, out.grad.T)
        if out.requires_grad:
            out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), _children=(self,), _op='relu')
        out.requires_grad = self.requires_grad and _grad_enabled

        def _backward():
            _accumulate(self, (self.data > 0).astype(self.data.dtype) * out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,), _op='exp')
        out.requires_grad = self.requires_grad and _grad_enabled

        def _backward():
            _accumulate(self, out.data * out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), _children=(self,), _op='log')
        out.requires_grad = self.requires_grad and _grad_enabled

        def _backward():
            _accumulate(self, (1.0 / self.data) * out.grad)
        if out.requires_grad:
            out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     _children=(self,), _op='sum')
        out.requires_grad = self.requires_grad and _grad_enabled

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            _accumulate(self, np.broadcast_to(g, self.data.shape))
        if out.requires_grad:
            out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        # mean = sum / count, composed from sum so autograd handles the backward
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = int(np.prod([self.data.shape[a] for a in axis]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------ #
    # Reverse-mode differentiation
    # ------------------------------------------------------------------ #
    def backward(self, gradient=None):
        """Populate .grad on every graph node that requires grad, via the chain rule."""
        # build a topological ordering of the graph rooted at self
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # seed the output gradient (dL/dL = 1 for a scalar loss)
        if gradient is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(gradient, dtype=self.data.dtype)

        # propagate backwards
        for v in reversed(topo):
            v._backward()


#%% Factory
def tensor(data, requires_grad=False):
    """Create a Tensor (mirrors torch.tensor)."""
    return Tensor(data, requires_grad=requires_grad)
