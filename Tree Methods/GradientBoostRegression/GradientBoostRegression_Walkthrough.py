# -*- coding: utf-8 -*-
"""
GradientBoostRegression_Walkthrough.py -- the code cells of GradientBoostRegression_Walkthrough.ipynb, extracted for step-through debugging.

How to use
- Open this file side by side with GradientBoostRegression_Walkthrough.ipynb. Each "# %%" block below is one code cell of the
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
assert (NB_DIR / "GradientBoostRegression.py").exists(), "Open/run this notebook from inside the GradientBoostRegression/ folder"
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
# The implementation under study: imported from this folder, never copied into the notebook
from GradientBoostRegression import GradientBoostingRegressor
from DecisionTreeGradientBoostRegression import DecisionTreeRegressor

print("GradientBoostingRegressor methods:", [k for k in vars(GradientBoostingRegressor) if not k.startswith("__")])
print("DecisionTreeRegressor methods:    ", [k for k in vars(DecisionTreeRegressor) if not k.startswith("__")])

# Illustration helper: which leaf (Node object) does each row reach? Used to group samples by leaf.
def leaf_of(x, node):
    while node.value is None:
        node = node.left if x[node.feature_index] <= node.threshold else node.right
    return node


#===================================================================================================
# %% [Code cell 4  |  notebook cell 7]  2. The problem: regression with squared error
#===================================================================================================
SIGMA = 0.2
mu = lambda x: np.sin(2 * x) + 0.5 * x
def make_curve(n):
    x = np.sort(rng.uniform(0, 3, n))
    return x.reshape(-1, 1), mu(x) + SIGMA * rng.normal(size=n)
X_tr, y_tr = make_curve(150)
X_te, y_te = make_curve(1000)
xg = np.linspace(0, 3, 400).reshape(-1, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, depth in zip(axes, [2, 10]):
    t = DecisionTreeRegressor(max_depth=depth); t.fit(X_tr, y_tr)
    mse = np.mean((t.predict(X_te) - y_te) ** 2)
    ax.scatter(X_tr, y_tr, s=10, alpha=0.5, color=CLASS_COLORS[0], label="training data")
    ax.plot(xg, mu(xg), "k--", label="true $\\mu(x)$"); ax.plot(xg, t.predict(xg), "r", label=f"one tree, max_depth={depth}")
    ax.set(title=f"A single tree, max_depth={depth}: test MSE {mse:.3f}", xlabel="$x_0$"); ax.legend()
axes[0].set_ylabel("$y$"); plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 5  |  notebook cell 9]  4. Core concepts, theory and math  >  4.1 Start from the best constant
#===================================================================================================
gb0 = GradientBoostingRegressor(n_estimators=0); gb0.fit(X_tr, y_tr)
print(f"init_prediction_ = {gb0.init_prediction_:.6f},  mean(y) = {y_tr.mean():.6f}")
assert np.isclose(gb0.init_prediction_, y_tr.mean()) and np.allclose(gb0.predict(X_te), y_tr.mean())
print("✓ F_0 is the training mean, and a model with zero trees predicts it everywhere")


#===================================================================================================
# %% [Code cell 6  |  notebook cell 11]  4. Core concepts, theory and math  >  4.2 The negative gradient of the squared loss is the residual
#===================================================================================================
# Illustration only — finite-difference check of Eq. (2) at a few training points
F = np.full(len(y_tr), y_tr.mean())                        # current predictions F_0
L = lambda F_: 0.5 * np.sum((y_tr - F_) ** 2)
for i in [0, 50, 120]:
    h = 1e-6; Fp = F.copy(); Fp[i] += h
    print(f"i = {i:>3}: -dL/dF_i ≈ {-(L(Fp) - L(F)) / h:+.5f}   residual y_i - F_i = {y_tr[i] - F[i]:+.5f}")
    assert np.isclose(-(L(Fp) - L(F)) / h, y_tr[i] - F[i], atol=1e-4)


#===================================================================================================
# %% [Code cell 7  |  notebook cell 13]  4. Core concepts, theory and math  >  4.3 One boosting round: fit a tree to the residuals, then take a damped step
#===================================================================================================
eta = 0.5
r1 = y_tr - y_tr.mean()                                         # residuals after F_0
tree1 = DecisionTreeRegressor(max_depth=2); tree1.fit(X_tr, r1)
tree1.print_tree()

gb1 = GradientBoostingRegressor(n_estimators=1, max_depth=2, learning_rate=eta); gb1.fit(X_tr, y_tr)
print("\nsame tree inside GradientBoostingRegressor:", np.allclose(gb1.trees[0].predict(X_te), tree1.predict(X_te)))


#===================================================================================================
# %% [Code cell 8  |  notebook cell 14]  4. Core concepts, theory and math  >  4.3 One boosting round: fit a tree to the residuals, then take a damped step
#===================================================================================================
# Check — Eq. (3): every leaf holds the mean residual of the training samples that reach it,
# and the one-round model is F_0 + η·f_1
groups = {}
for x_i, r_i in zip(X_tr, r1):
    groups.setdefault(id(leaf_of(x_i, tree1.root)), []).append(r_i)
for nd in iter_nodes(tree1.root):
    if nd.value is not None:
        assert np.isclose(nd.value, np.mean(groups[id(nd)]))
        print(f"leaf value {nd.value:+.4f} = mean of {len(groups[id(nd)]):>3} residuals")
assert np.allclose(gb1.predict(X_te), y_tr.mean() + eta * tree1.predict(X_te))
print("✓ leaves are mean residuals (Eq. 3) and predict() = F_0 + η·f_1(x)")


#===================================================================================================
# %% [Code cell 9  |  notebook cell 16]  4. Core concepts, theory and math  >  4.4 Training error can only go down (for $0 < \eta < 2$)
#===================================================================================================
gbm = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.3); gbm.fit(X_tr, y_tr)
stages = [np.full(len(y_tr), gbm.init_prediction_)] + gbm.get_staged_predictions(X_tr)
sse = np.array([np.sum((y_tr - s) ** 2) for s in stages])
assert np.all(np.diff(sse) <= 1e-9)
# exact decrease: Σ_leaves |ℓ| r̄_ℓ² η(2-η), with r̄ from the residuals before each round
for t in [0, 10, 40]:
    r = y_tr - stages[t]; tr = gbm.trees[t]; leaves = {}
    for x_i, r_i in zip(X_tr, r):
        leaves.setdefault(id(leaf_of(x_i, tr.root)), []).append(r_i)
    predicted = sum(len(v) * np.mean(v) ** 2 for v in leaves.values()) * 0.3 * (2 - 0.3)
    assert np.isclose(sse[t] - sse[t + 1], predicted)
    print(f"round {t + 1:>2}: SSE {sse[t]:8.4f} -> {sse[t + 1]:8.4f}  (drop {sse[t] - sse[t + 1]:.5f} = formula {predicted:.5f})")
print(f"✓ training SSE is non-increasing over all {len(gbm.trees)} rounds, and each drop equals Σ|ℓ| r̄² η(2-η)")


#===================================================================================================
# %% [Code cell 10  |  notebook cell 18]  4. Core concepts, theory and math  >  4.5 The tree: midpoint thresholds
#===================================================================================================
t_mid = DecisionTreeRegressor(max_depth=1); t_mid.fit(np.array([[1.0], [3.0]]), np.array([0.0, 1.0]))
print("threshold chosen for training points x = 1 and x = 3:", t_mid.root.threshold)
assert t_mid.root.threshold == 2.0


#===================================================================================================
# %% [Code cell 11  |  notebook cell 20]  4. Core concepts, theory and math  >  4.5 The tree: midpoint thresholds
#===================================================================================================
X_dup = np.array([[1.0, 5.0]] * 4 + [[2.0, 1.0]] * 4); y_dup = np.array([0., 1., 2., 3., 10., 11., 12., 13.])
g_dup = GradientBoostingRegressor(n_estimators=20, max_depth=3, learning_rate=0.5); g_dup.fit(X_dup, y_dup)
print("duplicate rows with different targets: fit succeeded; predictions for the two distinct inputs:",
      np.round(g_dup.predict(np.array([[1.0, 5.0], [2.0, 1.0]])), 3), "(group means are 1.5 and 11.5)")


#===================================================================================================
# %% [Code cell 12  |  notebook cell 22]  6. Worked numerical example: three rounds with stumps
#===================================================================================================
x_toy = np.arange(1, 9, dtype=float).reshape(-1, 1)
y_toy = np.array([1.0, 1.4, 1.2, 3.2, 3.0, 3.4, 6.0, 5.6])
toy = GradientBoostingRegressor(n_estimators=3, max_depth=1, learning_rate=0.5); toy.fit(x_toy, y_toy)

F = np.full(8, toy.init_prediction_)
print(f"F_0 = mean(y) = {F[0]:.4f};  training MSE {np.mean((y_toy - F) ** 2):.4f}")
for t, tree in enumerate(toy.trees):
    r = y_toy - F
    left = x_toy.ravel() <= tree.root.threshold
    print(f"\n--- round {t + 1}")
    print(f"residuals r = y - F           : {np.round(r, 4)}")
    print(f"stump: x_0 <= {tree.root.threshold}  (variance reduction {tree.root.var_reduction:.4f})")
    print(f"leaf values = mean residuals   : left {tree.root.left.value:+.4f} (mean {r[left].mean():+.4f}), "
          f"right {tree.root.right.value:+.4f} (mean {r[~left].mean():+.4f})")
    assert np.isclose(tree.root.left.value, r[left].mean()) and np.isclose(tree.root.right.value, r[~left].mean())
    F = F + 0.5 * tree.predict(x_toy)
    print(f"F_{t + 1} = F_{t} + 0.5·f_{t + 1}            : {np.round(F, 4)}")
    print(f"training MSE                   : {np.mean((y_toy - F) ** 2):.4f}")
assert np.allclose(F, toy.predict(x_toy))
print("\n✓ the hand-computed F_3 equals toy.predict()")


#===================================================================================================
# %% [Code cell 13  |  notebook cell 24]  6. Worked numerical example: three rounds with stumps
#===================================================================================================
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
xx = np.linspace(0.5, 8.5, 400).reshape(-1, 1)
stages_toy = toy.get_staged_predictions(xx)
for k, ax in enumerate(axes):
    ax.scatter(x_toy, y_toy, color=CLASS_COLORS[0], s=50, zorder=3, label="data")
    ax.plot(xx, np.full(len(xx), toy.init_prediction_) if k == 0 else stages_toy[k - 1], "r", lw=2, label=f"$F_{k}$")
    ax.set(title=f"after {k} tree(s)", xlabel="$x_0$"); ax.legend(loc="upper left")
axes[0].set_ylabel("$y$"); plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 14  |  notebook cell 26]  7. Visual experiments  >  7.1 The fit after 1, 5, 20 and 200 trees
#===================================================================================================
gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.1); gb.fit(X_tr, y_tr)
stages_grid = gb.get_staged_predictions(xg)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
for ax, T in zip(axes, [1, 5, 20, 200]):
    ax.scatter(X_tr, y_tr, s=8, alpha=0.4, color=CLASS_COLORS[0])
    ax.plot(xg, mu(xg), "k--", lw=1); ax.plot(xg, stages_grid[T - 1], "r", lw=1.8)
    test_pred = gb.get_staged_predictions(X_te)[T - 1]
    ax.set(title=f"{T} tree(s): test MSE {np.mean((test_pred - y_te) ** 2):.3f}", xlabel="$x_0$")
axes[0].set_ylabel("$y$"); plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 15  |  notebook cell 28]  7. Visual experiments  >  7.2 Residuals shrink round by round
#===================================================================================================
stages_tr = [np.full(len(y_tr), gb.init_prediction_)] + gb.get_staged_predictions(X_tr)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.4), sharey=True)
for ax, t in zip(axes, [0, 1, 5, 50]):
    r = y_tr - stages_tr[t]
    ax.axhline(0, color="gray", lw=1); ax.scatter(X_tr, r, s=10, color=CLASS_COLORS[1])
    ax.set(title=f"residuals before tree {t + 1}: RMS {np.sqrt(np.mean(r ** 2)):.3f}", xlabel="$x_0$")
axes[0].set_ylabel("$r = y - F_t(x)$"); plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 16  |  notebook cell 30]  7. Visual experiments  >  7.3 Learning rate and number of trees
#===================================================================================================
fig, ax = plt.subplots()
lr_results = {}
for eta_, col in zip([1.0, 0.3, 0.1, 0.03], CLASS_COLORS):
    g_ = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=eta_); g_.fit(X_tr, y_tr)
    tr_mse = [np.mean((p - y_tr) ** 2) for p in g_.get_staged_predictions(X_tr)]
    te_mse = [np.mean((p - y_te) ** 2) for p in g_.get_staged_predictions(X_te)]
    lr_results[eta_] = (np.min(te_mse), int(np.argmin(te_mse)) + 1, te_mse[-1], tr_mse[-1])
    ax.plot(range(1, 301), te_mse, color=col, label=f"η = {eta_}: test")
    ax.plot(range(1, 301), tr_mse, color=col, ls=":", label=f"η = {eta_}: train")
ax.axhline(SIGMA**2, color="gray", ls="--", label="σ² = 0.04")
ax.set(ylim=(0, 0.15), title="Train (dotted) and test (solid) MSE vs. number of trees", xlabel="number of trees", ylabel="MSE")
ax.legend(ncol=2, fontsize=8); plt.show()
for eta_, (best, at, last, trl) in lr_results.items():
    print(f"η = {eta_:<4}: best test MSE {best:.4f} at {at:>3} trees; after 300 trees test {last:.4f}, train {trl:.4f}")


#===================================================================================================
# %% [Code cell 17  |  notebook cell 32]  7. Visual experiments  >  7.4 Interactions: why tree depth matters
#===================================================================================================
X2 = rng.uniform(-1, 1, size=(300, 2)); y2 = 4 * X2[:, 0] * X2[:, 1] + 0.1 * rng.normal(size=300)
X2_te = rng.uniform(-1, 1, size=(1000, 2)); y2_te = 4 * X2_te[:, 0] * X2_te[:, 1] + 0.1 * rng.normal(size=1000)
g0_, g1_ = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60)); grid2 = np.c_[g0_.ravel(), g1_.ravel()]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
axes[0].contourf(g0_, g1_, 4 * g0_ * g1_, levels=20, cmap="coolwarm"); axes[0].set(title="true $4x_0x_1$", xlabel="$x_0$", ylabel="$x_1$")
for ax, depth in zip(axes[1:], [1, 3]):
    g_ = GradientBoostingRegressor(n_estimators=150, max_depth=depth, learning_rate=0.1); g_.fit(X2, y2)
    mse = np.mean((g_.predict(X2_te) - y2_te) ** 2)
    ax.contourf(g0_, g1_, g_.predict(grid2).reshape(g0_.shape), levels=20, cmap="coolwarm", vmin=-4, vmax=4)
    ax.set(title=f"150 trees of max_depth={depth}: test MSE {mse:.3f}", xlabel="$x_0$")
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 18  |  notebook cell 34]  7. Visual experiments  >  7.5 Feature importance
#===================================================================================================
X6 = rng.uniform(0, 1, size=(200, 6))
y6 = 10 * np.sin(np.pi * X6[:, 0] * X6[:, 1]) + 20 * (X6[:, 2] - 0.5) ** 2 + 10 * X6[:, 3] + 5 * X6[:, 4] + rng.normal(size=200)
g6 = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.1); g6.fit(X6, y6)
imp = g6.get_feature_importance(6)
manual = np.zeros(6)
for t in g6.trees:
    for nd in iter_nodes(t.root):
        if nd.value is None:
            manual[nd.feature_index] += nd.var_reduction
assert np.allclose(imp, manual / len(g6.trees))
fig, ax = plt.subplots()
ax.bar([f"x{j}" for j in range(6)], imp, color=[CLASS_COLORS[0]] * 5 + ["gray"])
ax.set(title="Repo feature importance (mean total var_reduction per tree)", ylabel="importance"); plt.show()
print("importances:", np.round(imp, 3), "| ✓ equals the per-node sum of var_reduction / number of trees")


#===================================================================================================
# %% [Code cell 19  |  notebook cell 36]  8. Hyperparameters
#===================================================================================================
idx = rng.permutation(len(X_tr)); tr_idx, va_idx = idx[:110], idx[110:]
g_es = GradientBoostingRegressor(n_estimators=400, max_depth=2, learning_rate=0.1); g_es.fit(X_tr[tr_idx], y_tr[tr_idx])
val_mse = [np.mean((p - y_tr[va_idx]) ** 2) for p in g_es.get_staged_predictions(X_tr[va_idx])]
best_T = int(np.argmin(val_mse)) + 1
test_mse = [np.mean((p - y_te) ** 2) for p in g_es.get_staged_predictions(X_te)]
print(f"validation-selected T = {best_T}: test MSE {test_mse[best_T - 1]:.4f}   (all 400 trees: {test_mse[-1]:.4f}; best possible: {min(test_mse):.4f})")
