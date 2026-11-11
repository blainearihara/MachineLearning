# -*- coding: utf-8 -*-
"""
DecisionTreeRegression_Walkthrough.py -- the code cells of DecisionTreeRegression_Walkthrough.ipynb, extracted for step-through debugging.

How to use
- Open this file side by side with DecisionTreeRegression_Walkthrough.ipynb. Each "# %%" block below is one code cell of the
  notebook, in the same order, with the code copied verbatim. The block title gives the code-cell
  number and the notebook section (Markdown heading) the cell sits under.
- Set breakpoints anywhere and run under the debugger (VS Code: F5 / "Python Debugger: Debug Python
  File"; Spyder: Debug file), or run cell by cell (VS Code: "Run Cell"; Spyder: Ctrl+Enter).
- The explanations, derivations and saved outputs live in the notebook; this file has code only.
- Plots: each plt.show() opens a window and, in a plain script, pauses until you close it. To keep
  going without closing windows, uncomment the plt.ion() line in the preamble below.

Generated from the notebook; if the notebook changes, regenerate this file rather than editing both.
"""

# %% [Preamble — not in the notebook] make the script behave like the notebook kernel
# The notebook runs with its own folder as the working directory, and its setup cell relies on that
# (it imports the repo modules from Path.cwd()). Running a script from elsewhere would break that, so
# switch to this file's folder first. Everything after this block is the notebook's code, unchanged.
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parent)
# import matplotlib.pyplot as _plt; _plt.ion()   # optional: non-blocking plot windows


#===================================================================================================
# %% [Code cell 1  |  notebook cell 3]  1. Setup
#===================================================================================================
# Setup: imports, reproducibility, plot style, and the import path to the repo modules
import sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

NB_DIR = Path.cwd()                       # Jupyter starts the kernel in the notebook's own folder
assert (NB_DIR / "RegressionTree.py").exists(), "Open/run this notebook from inside the DecisionTreeRegression/ folder"
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))       # makes the repo modules in this folder importable

SEED = 0
rng = np.random.default_rng(SEED)         # the notebook's own generator for all synthetic data; it is independent
                                          # of NumPy's global random state and of the models' private generators
np.set_printoptions(precision=4, suppress=True)

# One plot style shared by every walkthrough notebook in this repo
plt.rcParams.update({
    "figure.figsize": (7, 4.5), "figure.dpi": 100,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9, "legend.frameon": False,
})
CLASS_COLORS  = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]   # class 0, 1, 2, 3 (points)
REGION_COLORS = ["#c6dbef", "#fcbba1", "#c7e9c0", "#dadaeb"]   # class 0, 1, 2, 3 (decision regions)


#===================================================================================================
# %% [Code cell 2  |  notebook cell 4]  1. Setup
#===================================================================================================
# Teaching helpers — Illustration only, not the repo implementation.
# Synthetic-data generators, a train/test split, a decision-region plotter, and read-only
# inspectors that walk the Node objects built by the repo code (they never modify a tree).

def make_xor(n, noise=0.0, rng=rng):
    """Points uniform on [-1, 1]^2; class 1 when x0*x1 < 0 (2nd and 4th quadrants). `noise` = label-flip rate."""
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] * X[:, 1] < 0).astype(float)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return X, y

def make_moons(n, noise=0.2, rng=rng):
    """Two interleaving half-circles with Gaussian jitter (the classic 'moons' toy problem)."""
    n0 = n // 2; n1 = n - n0
    t0, t1 = rng.uniform(0, np.pi, n0), rng.uniform(0, np.pi, n1)
    X = np.r_[np.c_[np.cos(t0), np.sin(t0)], np.c_[1 - np.cos(t1), 0.5 - np.sin(t1)]]
    X = X + noise * rng.normal(size=(n, 2))
    return X, np.r_[np.zeros(n0), np.ones(n1)]

def make_blobs(n, centers, std=1.0, rng=rng):
    """Gaussian clusters; the label is the index of the cluster a point came from."""
    centers = np.asarray(centers, dtype=float)
    y = rng.integers(0, len(centers), size=n)
    X = centers[y] + std * rng.normal(size=(n, centers.shape[1]))
    return X, y.astype(float)

def noisy_sine(n, noise=0.3, rng=rng):
    """1-D regression toy problem: y = sin(x) + Gaussian noise, x uniform on [0, 2*pi]."""
    x = np.sort(rng.uniform(0, 2 * np.pi, n))
    return x.reshape(-1, 1), np.sin(x) + noise * rng.normal(size=n)

def train_test_split_np(X, y, test_frac=0.3, rng=rng):
    """Random train/test split with NumPy only."""
    idx = rng.permutation(len(X))
    n_test = int(round(test_frac * len(X)))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]

def iter_nodes(node):
    """Yield every node of a tree in pre-order (works with every Node class in this repo)."""
    if node is None:
        return
    yield node
    if node.value is None:                       # decision node: recurse into both children
        yield from iter_nodes(node.left)
        yield from iter_nodes(node.right)

def tree_depth(node):
    """Number of edges on the longest root-to-leaf path (a single leaf has depth 0)."""
    return 0 if node.value is not None else 1 + max(tree_depth(node.left), tree_depth(node.right))

def count_leaves(node):
    return sum(1 for nd in iter_nodes(node) if nd.value is not None)

def plot_decision_regions(predict_fn, X, y, ax=None, title="", grid_n=120, pad=0.4, legend=True):
    """Colour a grid by predict_fn's class label, then overlay the data points coloured by true class."""
    ax = ax or plt.gca()
    g0 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, grid_n)
    g1 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, grid_n)
    G0, G1 = np.meshgrid(g0, g1)
    Z = np.asarray(predict_fn(np.c_[G0.ravel(), G1.ravel()]), dtype=float).reshape(G0.shape)
    K = int(max(Z.max(), y.max())) + 1
    ax.contourf(G0, G1, Z, levels=np.arange(K + 1) - 0.5, cmap=ListedColormap(REGION_COLORS[:K]))
    for k in range(K):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=16, c=CLASS_COLORS[k], edgecolor="white", linewidth=0.4, label=f"class {k}")
    ax.set_title(title); ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
    if legend:
        ax.legend(loc="upper right", frameon=True, framealpha=0.8)
    return ax


#===================================================================================================
# %% [Code cell 3  |  notebook cell 5]  1. Setup
#===================================================================================================
# The implementation under study: imported from RegressionTree.py, never copied into this notebook
import RegressionTree as rt_mod
from RegressionTree import DecisionTreeRegressor, Node

print("module:", Path(rt_mod.__file__).name)
print("DecisionTreeRegressor methods:", [k for k in vars(DecisionTreeRegressor) if not k.startswith("__")])


#===================================================================================================
# %% [Code cell 4  |  notebook cell 7]  2. The problem: regression under squared error
#===================================================================================================
SIGMA = 0.3
X_sin, y_sin = noisy_sine(120, noise=SIGMA)
X_sin_test, y_sin_test = noisy_sine(2000, noise=SIGMA)
x_grid = np.linspace(0, 2 * np.pi, 400).reshape(-1, 1)

fig, ax = plt.subplots()
ax.scatter(X_sin, y_sin, s=14, color=CLASS_COLORS[0], alpha=0.7, label="training data (n = 120)")
ax.plot(x_grid, np.sin(x_grid), "k--", lw=1.5, label="true mean $\\mu(x) = \\sin x$")
ax.set(title="Noisy sine: the running regression example", xlabel="$x_0$", ylabel="$y$")
ax.legend(); plt.show()


#===================================================================================================
# %% [Code cell 5  |  notebook cell 9]  4. Core concepts, theory and math  >  4.1 The leaf value: the mean minimises squared error
#===================================================================================================
# Illustration only — SSE(c) for one leaf's targets; the minimum sits at the mean (Eq. 1)
y_leaf = np.array([2.1, 2.9, 3.4, 1.8, 4.0, 3.1])
cs = np.linspace(0, 6, 601)
sse = ((y_leaf[None, :] - cs[:, None]) ** 2).sum(axis=1)
c_best = cs[np.argmin(sse)]

fig, ax = plt.subplots()
ax.plot(cs, sse, lw=2, label="SSE(c) = $\\sum_i (y_i - c)^2$")
ax.axvline(y_leaf.mean(), color="k", ls="--", label=f"mean $\\bar y$ = {y_leaf.mean():.3f}")
ax.plot(c_best, sse.min(), "r*", ms=14, label=f"grid minimum c = {c_best:.2f}")
ax.set(title="Choosing a leaf's constant prediction", xlabel="constant prediction c", ylabel="sum of squared errors")
ax.legend(); plt.show()


#===================================================================================================
# %% [Code cell 6  |  notebook cell 11]  4. Core concepts, theory and math  >  4.1 The leaf value: the mean minimises squared error
#===================================================================================================
reg = DecisionTreeRegressor()                       # fresh instance, used only for its helper methods
print("calculate_leaf_value(y_leaf) =", reg.calculate_leaf_value(y_leaf))


#===================================================================================================
# %% [Code cell 7  |  notebook cell 12]  4. Core concepts, theory and math  >  4.1 The leaf value: the mean minimises squared error
#===================================================================================================
# Check — Eq. (1): the repo's leaf value is the SSE-minimising constant, and Eq. (2) links SSE to variance
v = reg.calculate_leaf_value(y_leaf)
assert np.isclose(v, y_leaf.mean()) and abs(v - c_best) <= 0.01            # grid resolution is 0.01
assert np.isclose(((y_leaf - v) ** 2).sum(), len(y_leaf) * np.var(y_leaf))
print(f"✓ leaf value {v:.4f} = mean = SSE minimiser; SSE at the mean = |S|·Var(S) = {len(y_leaf) * np.var(y_leaf):.4f}")


#===================================================================================================
# %% [Code cell 8  |  notebook cell 14]  4. Core concepts, theory and math  >  4.2 Scoring a split: variance reduction
#===================================================================================================
# Illustration only — the three equivalent forms of the variance reduction for one split
y_par = np.array([1.0, 1.4, 0.6, 3.6, 4.4, 4.0, 2.2, 2.6])
yL, yR = y_par[:3], y_par[3:]
wL, wR = len(yL) / len(y_par), len(yR) / len(y_par)
d_var  = np.var(y_par) - (wL * np.var(yL) + wR * np.var(yR))                               # Eq. (3)
sse    = lambda a: ((a - a.mean()) ** 2).sum()
d_sse  = (sse(y_par) - sse(yL) - sse(yR)) / len(y_par)                                      # Eq. (4)
d_mean = wL * wR * (yL.mean() - yR.mean()) ** 2                                              # Eq. (5)
print(f"Eq.(3) variance form   : {d_var:.6f}")
print(f"Eq.(4) SSE form / |S|  : {d_sse:.6f}")
print(f"Eq.(5) w_L w_R (Δmean)²: {d_mean:.6f}   (w_L = {wL:.3f}, means {yL.mean():.3f} vs {yR.mean():.3f})")


#===================================================================================================
# %% [Code cell 9  |  notebook cell 16]  4. Core concepts, theory and math  >  4.2 Scoring a split: variance reduction
#===================================================================================================
d_repo = reg.variance_reduction(y_par, yL, yR)
print(f"variance_reduction(...) = {d_repo:.6f}")


#===================================================================================================
# %% [Code cell 10  |  notebook cell 17]  4. Core concepts, theory and math  >  4.2 Scoring a split: variance reduction
#===================================================================================================
# Check — the repo matches all three forms, and is never negative over many random splits
assert np.isclose(d_repo, d_var) and np.isclose(d_repo, d_sse) and np.isclose(d_repo, d_mean)
worst = np.inf
for _ in range(1000):
    y_ = rng.normal(size=rng.integers(2, 30)); s = rng.integers(1, len(y_))
    dr = reg.variance_reduction(y_, y_[:s], y_[s:])
    assert np.isclose(dr, (s / len(y_)) * (1 - s / len(y_)) * (y_[:s].mean() - y_[s:].mean()) ** 2)
    worst = min(worst, dr)
print(f"✓ variance_reduction = Eq.(3) = Eq.(4) = Eq.(5); over 1,000 random splits it matched Eq.(5) "
      f"and its minimum was {worst:.2e} (≥ 0 up to rounding)")


#===================================================================================================
# %% [Code cell 11  |  notebook cell 19]  4. Core concepts, theory and math  >  4.3 Searching for the best split
#===================================================================================================
# Illustration only — Eq. (5) evaluated at every candidate threshold of the 1-D sine data
def scan_var_red(x, y):
    taus, gains = [], []
    for tau in np.unique(x)[:-1]:                          # the largest value would leave the right child empty
        left = x <= tau
        gains.append(left.mean() * (1 - left.mean()) * (y[left].mean() - y[~left].mean()) ** 2)
        taus.append(tau)
    return np.array(taus), np.array(gains)

taus, gains = scan_var_red(X_sin.ravel(), y_sin)
tau_best = taus[np.argmax(gains)]                          # argmax returns the first maximum, like the repo

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(taus, gains, color=CLASS_COLORS[2])
axes[0].axvline(tau_best, color="k", ls="--", label=f"best τ = {tau_best:.3f}")
axes[0].set(title="Variance reduction of every threshold", xlabel="threshold τ on $x_0$", ylabel="Δ"); axes[0].legend()
left = X_sin.ravel() <= tau_best
axes[1].scatter(X_sin, y_sin, s=12, alpha=0.6, color=CLASS_COLORS[0], label="training data")
axes[1].hlines([y_sin[left].mean(), y_sin[~left].mean()], [0, tau_best], [tau_best, 2 * np.pi], colors="r", lw=2.5,
               label="child means after the best split")
axes[1].axvline(tau_best, color="k", ls="--")
axes[1].set(title="The best single split", xlabel="$x_0$", ylabel="$y$"); axes[1].legend()
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 12  |  notebook cell 21]  4. Core concepts, theory and math  >  4.3 Searching for the best split
#===================================================================================================
ds_sin = np.c_[X_sin, y_sin]
best = reg.get_best_split(ds_sin, *X_sin.shape)
print({k: (f"array of shape {v.shape}" if isinstance(v, np.ndarray) else v) for k, v in best.items()})


#===================================================================================================
# %% [Code cell 13  |  notebook cell 22]  4. Core concepts, theory and math  >  4.3 Searching for the best split
#===================================================================================================
# Check — the repo's search returns the arg-max of the independent scan, Eq. (6)
assert best["feature_index"] == 0 and best["threshold"] == tau_best
assert np.isclose(best["var_red"], gains.max())
print(f"✓ get_best_split chose x_0 <= {tau_best:.4f} with Δ = {gains.max():.4f}, the maximum over all {len(taus)} candidates")


#===================================================================================================
# %% [Code cell 14  |  notebook cell 24]  4. Core concepts, theory and math  >  4.4 Recursion and stopping
#===================================================================================================
print("max_depth | actual depth | leaves")
for md in [0, 1, 2, 3, 4, 5]:
    t = DecisionTreeRegressor(min_samples_split=2, max_depth=md)
    t.fit(X_sin, y_sin)                                     # 1-D targets are accepted (a column vector also works)
    print(f"{md:>9} | {tree_depth(t.root):>12} | {count_leaves(t.root):>6}")
    assert tree_depth(t.root) == md
print("✓ every tree is exactly max_depth deep")


#===================================================================================================
# %% [Code cell 15  |  notebook cell 26]  4. Core concepts, theory and math  >  4.4 Recursion and stopping
#===================================================================================================
for X_dup, y_dup, desc in [(np.array([[1.0], [1.0], [1.0]]), np.array([0.5, 1.5, 2.5]), "identical x, different y"),
                           (np.array([[3.0], [3.0]]), np.array([2.0, 2.0]), "identical x, identical y")]:
    t = DecisionTreeRegressor(min_samples_split=2, max_depth=3)
    t.fit(X_dup, y_dup)
    print(f"{desc:<26}: fit succeeded -> a single leaf predicting {t.root.value:.3f} (the mean)")
    assert t.root.value is not None and np.isclose(t.root.value, y_dup.mean())


#===================================================================================================
# %% [Code cell 16  |  notebook cell 28]  4. Core concepts, theory and math  >  4.5 Prediction: a step function
#===================================================================================================
# Illustration only — route points through the repo's Node objects by hand
def route(x, node):
    while node.value is None:
        node = node.left if x[node.feature_index] <= node.threshold else node.right
    return node.value

t3 = DecisionTreeRegressor(min_samples_split=2, max_depth=3); t3.fit(X_sin, y_sin.reshape(-1, 1))
pred_repo = np.array(t3.predict(x_grid))
pred_hand = np.array([route(x, t3.root) for x in x_grid])
assert np.array_equal(pred_repo, pred_hand)
print(f"✓ predict() equals hand routing on all {len(x_grid)} grid points; "
      f"the tree has {count_leaves(t3.root)} leaves, so predict() takes {len(np.unique(pred_repo))} distinct values")


#===================================================================================================
# %% [Code cell 17  |  notebook cell 30]  6. Worked numerical example
#===================================================================================================
X_toy = np.arange(1, 9, dtype=float).reshape(-1, 1)
y_toy = np.array([1.0, 1.4, 0.6, 3.6, 4.4, 4.0, 2.2, 2.6])
ds_toy = np.c_[X_toy, y_toy]
print(" i  x0     y")
for i, (a, b_) in enumerate(ds_toy):
    print(f"{i:>2} {a:>3.0f} {b_:>5.1f}")
print(f"\nroot: mean = {y_toy.mean():.4f}, Var = {np.var(y_toy):.4f}, SSE = {len(y_toy) * np.var(y_toy):.4f}")


#===================================================================================================
# %% [Code cell 18  |  notebook cell 32]  6. Worked numerical example
#===================================================================================================
def candidate_table(ds, title):
    """Every candidate split of a node, scored by the repo's split + variance_reduction, cross-checked by Eq. (5)."""
    y = ds[:, -1]; rows = []
    for tau in np.unique(ds[:, 0]):
        L, R = reg.split(ds, 0, tau)
        if len(L) and len(R):
            yl, yr = L[:, -1], R[:, -1]
            rows.append((tau, len(yl), len(yr), yl.mean(), yr.mean(), np.var(yl), np.var(yr),
                         reg.variance_reduction(y, yl, yr), len(yl) * len(yr) / len(y) ** 2 * (yl.mean() - yr.mean()) ** 2))
    best_d = max(r[7] for r in rows)
    print(title)
    print("  tau | n_L n_R | mean_L mean_R | Var_L  Var_R |   Δ (repo)  Δ (Eq. 5)")
    for tau, nl, nr, ml, mr, vl, vr, d, d5 in rows:
        print(f"  {tau:>3.0f} | {nl:>3} {nr:>3} | {ml:>6.3f} {mr:>6.3f} | {vl:>5.3f} {vr:>5.3f} | {d:>9.4f}  {d5:>9.4f}"
              + ("   <- best" if np.isclose(d, best_d) else ""))
        assert np.isclose(d, d5)

candidate_table(ds_toy, "Root (8 samples):")


#===================================================================================================
# %% [Code cell 19  |  notebook cell 34]  6. Worked numerical example
#===================================================================================================
root_split = reg.get_best_split(ds_toy, *X_toy.shape)
print(f"get_best_split at the root: x_{root_split['feature_index']} <= {root_split['threshold']}, Δ = {root_split['var_red']:.4f}\n")
candidate_table(root_split["dataset_left"], "Left child (samples 0-2):")
print()
candidate_table(root_split["dataset_right"], "Right child (samples 3-7):")


#===================================================================================================
# %% [Code cell 20  |  notebook cell 36]  6. Worked numerical example
#===================================================================================================
toy = DecisionTreeRegressor(min_samples_split=2, max_depth=2)
toy.fit(X_toy, y_toy.reshape(-1, 1))
toy.print_tree()

r = toy.root
assert (r.feature_index, r.threshold) == (0, 3.0)
assert (r.left.threshold, r.right.threshold) == (2.0, 6.0)
leaf_vals = [r.left.left.value, r.left.right.value, r.right.left.value, r.right.right.value]
assert np.allclose(leaf_vals, [1.2, 0.6, 4.0, 2.4]) and tree_depth(r) == 2
pred_train = np.array(toy.predict(X_toy))
print("\n✓ fit() built the hand-derived tree; leaf values (means):", np.round(leaf_vals, 4))
print("training predictions:", np.round(pred_train, 3))
print(f"training SSE: root only {len(y_toy) * np.var(y_toy):.4f} -> after 1 split "
      f"{len(y_toy) * (np.var(y_toy) - root_split['var_red']):.4f} -> full tree {np.sum((y_toy - pred_train) ** 2):.4f}")


#===================================================================================================
# %% [Code cell 21  |  notebook cell 38]  6. Worked numerical example
#===================================================================================================
print("predict([[6.5], [20.0], [0.0]]) =", np.round(toy.predict(np.array([[6.5], [20.0], [0.0]])), 3))
xx = np.linspace(0, 9, 500).reshape(-1, 1)
fig, ax = plt.subplots()
ax.scatter(X_toy, y_toy, s=60, color=CLASS_COLORS[0], edgecolor="k", zorder=3, label="8 training samples")
ax.step(xx.ravel(), toy.predict(xx), where="post", color="r", lw=2, label="tree prediction (4 leaves)")
for tthr in (2, 3, 6):
    ax.axvline(tthr, color="gray", ls=":", lw=1)
ax.set(title="Worked example: the fitted step function", xlabel="$x_0$", ylabel="$y$"); ax.legend(); plt.show()


#===================================================================================================
# %% [Code cell 22  |  notebook cell 40]  7. Visual experiments  >  7.1 The fit as the tree grows
#===================================================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for ax, md in zip(axes.ravel(), [1, 2, 3, 4, 6, 10]):
    t = DecisionTreeRegressor(min_samples_split=2, max_depth=md); t.fit(X_sin, y_sin.reshape(-1, 1))
    mse_tr = np.mean((np.array(t.predict(X_sin)) - y_sin) ** 2)
    mse_te = np.mean((np.array(t.predict(X_sin_test)) - y_sin_test) ** 2)
    ax.scatter(X_sin, y_sin, s=8, alpha=0.4, color=CLASS_COLORS[0])
    ax.plot(x_grid, np.sin(x_grid), "k--", lw=1)
    ax.plot(x_grid, t.predict(x_grid), color="r", lw=1.8)
    ax.set(title=f"max_depth={md} (depth {tree_depth(t.root)}, {count_leaves(t.root)} leaves)\ntrain MSE {mse_tr:.3f} / test MSE {mse_te:.3f}",
           xlabel="$x_0$")
axes[0, 0].set_ylabel("$y$"); axes[1, 0].set_ylabel("$y$")
axes[0, 0].plot([], [], "r", label="tree prediction"); axes[0, 0].plot([], [], "k--", label="true $\\sin x$"); axes[0, 0].legend()
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 23  |  notebook cell 42]  7. Visual experiments  >  7.2 Training and test error versus depth
#===================================================================================================
depths, mse_tr, mse_te = [], [], []
for md in range(1, 13):
    t = DecisionTreeRegressor(min_samples_split=2, max_depth=md); t.fit(X_sin, y_sin.reshape(-1, 1))
    depths.append(tree_depth(t.root))
    mse_tr.append(np.mean((np.array(t.predict(X_sin)) - y_sin) ** 2))
    mse_te.append(np.mean((np.array(t.predict(X_sin_test)) - y_sin_test) ** 2))

fig, ax = plt.subplots()
ax.plot(depths, mse_tr, "o-", label="training MSE")
ax.plot(depths, mse_te, "s-", label="test MSE (2,000 fresh points)")
ax.axhline(SIGMA**2, color="gray", ls="--", label="irreducible error $\\sigma^2 = 0.09$")
ax.set(title="Noisy sine: error vs. tree depth", xlabel="tree depth", ylabel="mean squared error")
ax.legend(); plt.show()
print(f"best test MSE {min(mse_te):.4f} at depth {depths[int(np.argmin(mse_te))]}; deepest: train {mse_tr[-1]:.4f}, test {mse_te[-1]:.4f}")


#===================================================================================================
# %% [Code cell 24  |  notebook cell 44]  7. Visual experiments  >  7.3 Bias–variance decomposition, measured
#===================================================================================================
# Monte-Carlo bias-variance decomposition (data simulation is illustration; every fit uses the repo's tree)
n_sets, n_train = 100, 60
xg = np.linspace(0.05, 2 * np.pi - 0.05, 200).reshape(-1, 1)
mu = np.sin(xg.ravel())
results = {}
for md in [1, 3, 10]:
    preds = np.empty((n_sets, len(xg)))
    for s in range(n_sets):
        Xs, ys = noisy_sine(n_train, noise=SIGMA)
        t = DecisionTreeRegressor(min_samples_split=2, max_depth=md); t.fit(Xs, ys.reshape(-1, 1))
        preds[s] = t.predict(xg)
    f_bar = preds.mean(axis=0)
    bias2, var = np.mean((f_bar - mu) ** 2), np.mean(preds.var(axis=0))
    y_new = mu[None, :] + SIGMA * rng.normal(size=preds.shape)          # fresh noisy targets for every fit
    results[md] = dict(preds=preds, f_bar=f_bar, bias2=bias2, var=var, test=np.mean((y_new - preds) ** 2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (md, r_) in zip(axes, results.items()):
    for p in r_["preds"][:25]:
        ax.plot(xg, p, color=CLASS_COLORS[1], alpha=0.12, lw=1)
    ax.plot(xg, r_["f_bar"], color=CLASS_COLORS[0], lw=2.5, label="average prediction $\\bar f$")
    ax.plot(xg, mu, "k--", lw=1.5, label="truth $\\mu = \\sin x$")
    ax.set(title=f"max_depth={md}: bias² {r_['bias2']:.3f}, variance {r_['var']:.3f}", xlabel="$x_0$")
axes[0].set_ylabel("prediction"); axes[0].plot([], [], color=CLASS_COLORS[1], label="25 individual trees"); axes[0].legend()
plt.tight_layout(); plt.show()

print(" max_depth |  bias²  variance   noise | bias²+var+noise | measured test MSE")
for md, r_ in results.items():
    total = r_["bias2"] + r_["var"] + SIGMA**2
    print(f"{md:>10} | {r_['bias2']:.4f}  {r_['var']:.4f}  {SIGMA**2:.4f} | {total:>15.4f} | {r_['test']:>17.4f}")
    assert abs(total - r_["test"]) / r_["test"] < 0.05


#===================================================================================================
# %% [Code cell 25  |  notebook cell 46]  7. Visual experiments  >  7.4 Two dimensions, and no extrapolation
#===================================================================================================
X2 = rng.uniform(0, 3, size=(300, 2))
y2 = np.sin(X2[:, 0]) * np.cos(X2[:, 1]) + 0.1 * rng.normal(size=300)
g = np.linspace(0, 3, 80); G0, G1 = np.meshgrid(g, g); grid2 = np.c_[G0.ravel(), G1.ravel()]

fig, axes = plt.subplots(1, 4, figsize=(19, 4))
im = axes[0].contourf(G0, G1, np.sin(G0) * np.cos(G1), levels=20, cmap="coolwarm", vmin=-1, vmax=1)
axes[0].set(title="true $\\mu = \\sin x_0 \\cos x_1$", xlabel="$x_0$", ylabel="$x_1$")
for ax, md in zip(axes[1:3], [3, 7]):
    t = DecisionTreeRegressor(min_samples_split=5, max_depth=md); t.fit(X2, y2.reshape(-1, 1))
    ax.contourf(G0, G1, np.array(t.predict(grid2)).reshape(G0.shape), levels=20, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(title=f"tree, max_depth={md} ({count_leaves(t.root)} leaves)", xlabel="$x_0$")
fig.colorbar(im, ax=axes[:3], shrink=0.85, label="prediction / $\\mu$")

t_ex = DecisionTreeRegressor(min_samples_split=2, max_depth=5); t_ex.fit(X_sin, y_sin.reshape(-1, 1))
x_wide = np.linspace(-2, 2 * np.pi + 2, 500).reshape(-1, 1)
axes[3].plot(x_wide, np.sin(x_wide), "k--", label="true $\\sin x$")
axes[3].plot(x_wide, t_ex.predict(x_wide), "r", label="tree prediction")
axes[3].axvspan(-2, 0, color="gray", alpha=0.15); axes[3].axvspan(2 * np.pi, 2 * np.pi + 2, color="gray", alpha=0.15, label="outside training range")
axes[3].set(title="Extrapolation is flat", xlabel="$x_0$", ylabel="$y$"); axes[3].legend(fontsize=8)
plt.show()


#===================================================================================================
# %% [Code cell 26  |  notebook cell 48]  8. Hyperparameters
#===================================================================================================
md_grid, mss_grid = [1, 2, 3, 4, 5, 7, 10], [2, 5, 10, 20, 40]
mse_grid = np.zeros((len(md_grid), len(mss_grid)))
for a, md in enumerate(md_grid):
    for b_, mss in enumerate(mss_grid):
        t = DecisionTreeRegressor(min_samples_split=mss, max_depth=md); t.fit(X_sin, y_sin.reshape(-1, 1))
        mse_grid[a, b_] = np.mean((np.array(t.predict(X_sin_test)) - y_sin_test) ** 2)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(mse_grid, cmap="viridis_r", aspect="auto", origin="lower")
ax.set_xticks(range(len(mss_grid)), mss_grid); ax.set_yticks(range(len(md_grid)), md_grid)
for a in range(len(md_grid)):
    for b_ in range(len(mss_grid)):
        ax.text(b_, a, f"{mse_grid[a, b_]:.3f}", ha="center", va="center", fontsize=8,
                color="w" if mse_grid[a, b_] > np.median(mse_grid) else "k")
ax.set(title="Test MSE on the noisy sine (lower is better)", xlabel="min_samples_split", ylabel="max_depth")
ax.grid(False); fig.colorbar(im, label="test MSE"); plt.show()
a_best, b_best = np.unravel_index(np.argmin(mse_grid), mse_grid.shape)
print(f"best: max_depth={md_grid[a_best]}, min_samples_split={mss_grid[b_best]}, test MSE {mse_grid[a_best, b_best]:.4f} "
      f"(irreducible error {SIGMA**2:.2f})")
