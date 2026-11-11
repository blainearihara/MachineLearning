# -*- coding: utf-8 -*-
"""
XGBoost_Walkthrough.py -- the code cells of XGBoost_Walkthrough.ipynb, extracted for step-through debugging.

How to use
- Open this file side by side with XGBoost_Walkthrough.ipynb. Each "# %%" block below is one code cell of the
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
assert (NB_DIR / "XGBoost.py").exists(), "Open/run this notebook from inside the XGBoost/ folder"
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
import copy
from XGBoost import XGBoostClassifier
from DecisionTreeXGBoost import DecisionTreeXGBoost, Node

print("XGBoostClassifier methods:  ", [k for k in vars(XGBoostClassifier) if not k.startswith("__")])
print("DecisionTreeXGBoost methods:", [k for k in vars(DecisionTreeXGBoost) if not k.startswith("__")])

def log_loss(P, y):                         # Illustration helper: mean cross-entropy
    return -np.mean(np.log(np.clip(P[np.arange(len(y)), y.astype(int)], 1e-15, 1)))
def staged_proba_xgb(model, X, T):          # Illustration helper: probabilities after T rounds (zero start)
    raw = np.zeros((len(X), model.n_classes_))
    for k in range(model.n_classes_):
        for tree in model.trees[k][:T]:
            raw[:, k] += model.learning_rate * tree.predict(X)
    return model.softmax(raw)
def leaves(node):
    return 1 if node.value is not None else leaves(node.left) + leaves(node.right)


#===================================================================================================
# %% [Code cell 4  |  notebook cell 7]  2. The problem
#===================================================================================================
centers = [[0.0, 0.0], [2.2, 0.0], [1.1, 1.9]]
X_tr, y_tr = make_blobs(240, centers=centers, std=0.9)
X_te, y_te = make_blobs(1000, centers=centers, std=0.9)
print("training classes:", np.bincount(y_tr.astype(int)), "| the same three overlapping blobs as in the classification notebook")


#===================================================================================================
# %% [Code cell 5  |  notebook cell 9]  4. Core concepts, theory and math  >  4.2 Optimal leaf weights
#===================================================================================================
tree_l1 = DecisionTreeXGBoost(reg_lambda=1.0)
g_leaf = rng.normal(0.2, 0.3, size=6); h_leaf = rng.uniform(0.05, 0.25, size=6)
vs = np.linspace(-5, 5, 200001)
obj = g_leaf.sum() * vs + 0.5 * (h_leaf.sum() + 1.0) * vs**2               # Illustration only: one bracket of Eq. (2)
print(f"grid minimiser {vs[np.argmin(obj)]:+.4f} | -G/(H+λ) = {-g_leaf.sum() / (h_leaf.sum() + 1):+.4f} | "
      f"calculate_leaf_value = {tree_l1.calculate_leaf_value(g_leaf, h_leaf):+.4f}")
assert np.isclose(tree_l1.calculate_leaf_value(g_leaf, h_leaf), vs[np.argmin(obj)], atol=1e-4)

lams = np.linspace(0, 10, 101)
fig, ax = plt.subplots()
for nleaf, col in [(2, CLASS_COLORS[1]), (20, CLASS_COLORS[0])]:
    G_, H_ = -0.4 * nleaf, 0.2 * nleaf                                       # same per-sample g and h, different leaf sizes
    ax.plot(lams, [DecisionTreeXGBoost(reg_lambda=l).calculate_leaf_value(np.full(nleaf, G_ / nleaf), np.full(nleaf, H_ / nleaf)) for l in lams],
            color=col, label=f"leaf with {nleaf} samples (Newton step at λ=0: {-G_ / H_:.1f})")
ax.set(title="Leaf value -G/(H+λ) shrinks with λ — fastest for small leaves", xlabel="λ (reg_lambda)", ylabel="leaf value $v^*$"); ax.legend(); plt.show()


#===================================================================================================
# %% [Code cell 6  |  notebook cell 11]  4. Core concepts, theory and math  >  4.3 The split gain
#===================================================================================================
t_gain = DecisionTreeXGBoost(reg_lambda=1.0, gamma=0.3)
for _ in range(300):
    g_ = rng.normal(size=10); h_ = rng.uniform(0.01, 0.25, 10); s = int(rng.integers(1, 10))
    GL, HL, GR, HR = g_[:s].sum(), h_[:s].sum(), g_[s:].sum(), h_[s:].sum()
    eq5 = 0.5 * (GL**2 / (HL + 1) + GR**2 / (HR + 1) - (GL + GR)**2 / (HL + HR + 1)) - 0.3
    obj_before = -0.5 * (GL + GR)**2 / (HL + HR + 1) + 0.3 * 1           # Eq. (4), one leaf
    obj_after = -0.5 * (GL**2 / (HL + 1) + GR**2 / (HR + 1)) + 0.3 * 2   # Eq. (4), two leaves
    assert np.isclose(t_gain.compute_gain(GL, HL, GR, HR, GL + GR, HL + HR), eq5)
    assert np.isclose(eq5, obj_before - obj_after)
print("✓ compute_gain = Eq. (5) = the decrease of the regularised objective (Eq. 4), on 300 random splits")


#===================================================================================================
# %% [Code cell 7  |  notebook cell 13]  4. Core concepts, theory and math  >  4.5 Post-pruning with $\gamma$
#===================================================================================================
X_x = rng.uniform(-1, 1, size=(80, 2)); y_x = (X_x[:, 0] * X_x[:, 1] < 0).astype(float)
p_x = np.full(80, 0.5); g_x = p_x - y_x; h_x = p_x * (1 - p_x)

def describe(node, depth=0):
    if node.value is not None:
        return f"{'    ' * depth}leaf {node.value:+.3f}\n"
    return (f"{'    ' * depth}X_{node.feature_index} <= {node.threshold:+.2f}   gain-γ = {node.gain:+.3f}\n"
            + describe(node.left, depth + 1) + describe(node.right, depth + 1))

for gam in [0.0, 4.0, 10.0]:
    d = DecisionTreeXGBoost(max_depth=2, reg_lambda=1.0, gamma=gam)
    grown = d.build_tree(X_x, g_x, h_x)                     # grown with raw gain > 0, nothing pruned yet
    pruned = d.prune_tree(copy.deepcopy(grown))
    print(f"γ = {gam}: {leaves(grown)} leaves grown -> {leaves(pruned)} after pruning")
    print(describe(pruned))


#===================================================================================================
# %% [Code cell 8  |  notebook cell 15]  4. Core concepts, theory and math  >  4.5 Post-pruning with $\gamma$
#===================================================================================================
# Check — a collapsed node gets the exact value -G/(H+λ) of all its samples
d = DecisionTreeXGBoost(max_depth=2, reg_lambda=1.0, gamma=10.0)
collapsed = d.prune_tree(d.build_tree(X_x, g_x, h_x))
print(f"fully pruned tree: one leaf {collapsed.value:+.4f};  -G/(H+λ) over all 80 samples = {-g_x.sum() / (h_x.sum() + 1):+.4f}")
assert collapsed.value is not None and np.isclose(collapsed.value, -g_x.sum() / (h_x.sum() + 1))


#===================================================================================================
# %% [Code cell 9  |  notebook cell 17]  4. Core concepts, theory and math  >  4.6 Row and column subsampling
#===================================================================================================
tc = DecisionTreeXGBoost(max_depth=3, colsample_bylevel=0.5, random_state=0)
Xc = rng.normal(size=(60, 6)); gc = rng.normal(size=60); hc = np.full(60, 0.25)
tc.fit(Xc, gc, hc)
print("features drawn per depth level:", {lvl: sorted(np.asarray(f).tolist()) for lvl, f in tc._level_features.items()})
used = {}
def collect(node, depth=0):
    if node.value is None:
        used.setdefault(depth, set()).add(int(node.feature_index)); collect(node.left, depth + 1); collect(node.right, depth + 1)
collect(tc.root)
print("features actually split on per level:", {k: sorted(v) for k, v in used.items()})
assert all(used[l] <= set(np.asarray(tc._level_features[l]).tolist()) for l in used)
print("✓ every node at a given depth splits on a feature from that level's draw")


#===================================================================================================
# %% [Code cell 10  |  notebook cell 19]  4. Core concepts, theory and math  >  4.6 Row and column subsampling
#===================================================================================================
m1 = XGBoostClassifier(n_estimators=5, max_depth=3, subsample=0.6, colsample_bylevel=0.5, random_state=0)
m1.fit(X_tr, y_tr); a = m1.predict_proba(X_te)
m1.fit(X_tr, y_tr); b = m1.predict_proba(X_te)
np.random.seed(7); before = np.random.random(); np.random.seed(7)
XGBoostClassifier(n_estimators=3, subsample=0.5, random_state=0).fit(X_tr, y_tr); after = np.random.random()
assert np.allclose(a, b) and before == after
print("✓ random_state=0 refits reproducibly, and fitting leaves NumPy's global random state untouched")


#===================================================================================================
# %% [Code cell 11  |  notebook cell 21]  6. Worked numerical example: one round, three classes
#===================================================================================================
x_toy = np.arange(1, 10, dtype=float).reshape(-1, 1)
y_toy = np.array([0, 0, 0, 1, 1, 2, 1, 2, 2], dtype=float)
toy = XGBoostClassifier(n_estimators=1, max_depth=1, learning_rate=0.5, reg_lambda=1.0, gamma=0.0, random_state=0)
toy.fit(x_toy, y_toy)
P0 = np.full((9, 3), 1 / 3); Y1 = np.eye(3)[y_toy.astype(int)]
G_all, H_all = None, None
for k in range(3):
    g = P0[:, k] - Y1[:, k]; h = P0[:, k] * (1 - P0[:, k]); tree = toy.trees[k][0]
    print(f"--- class {k}: g = {np.round(g, 3)},  h = {h[0]:.4f} each;  G = {g.sum():+.3f}, H = {h.sum():.3f}")
    print("    τ   |  G_L     H_L   |  G_R     H_R   | Gain (Eq. 5)")
    gains = []
    for tau in np.arange(1.5, 9, 1.0):
        L = x_toy.ravel() <= tau
        gain = tree.compute_gain(g[L].sum(), h[L].sum(), g[~L].sum(), h[~L].sum(), g.sum(), h.sum())
        gains.append(gain)
        print(f"   {tau:>3} | {g[L].sum():+.3f}  {h[L].sum():.3f} | {g[~L].sum():+.3f}  {h[~L].sum():.3f} | {gain:.4f}")
    L = x_toy.ravel() <= tree.root.threshold
    print(f"    chosen: x_0 <= {tree.root.threshold}  (max Gain {max(gains):.4f});"
          f"  leaves -G/(H+λ): left {-g[L].sum() / (h[L].sum() + 1):+.4f}, right {-g[~L].sum() / (h[~L].sum() + 1):+.4f}"
          f"  | repo: {tree.root.left.value:+.4f}, {tree.root.right.value:+.4f}")
    assert np.isclose(tree.root.left.value, -g[L].sum() / (h[L].sum() + 1)) and np.isclose(tree.root.right.value, -g[~L].sum() / (h[~L].sum() + 1))
raw1 = np.column_stack([0.5 * toy.trees[k][0].predict(x_toy) for k in range(3)])
P1 = toy.softmax(raw1)
print("\nprobabilities after one round:\n", np.round(P1, 3))
print("predicted:", P1.argmax(1).tolist(), " true:", y_toy.astype(int).tolist(), f" log-loss {log_loss(P1, y_toy):.4f}")
assert np.allclose(P1, toy.predict_proba(x_toy))


#===================================================================================================
# %% [Code cell 12  |  notebook cell 23]  7. Visual experiments  >  7.1 $\gamma$ and $\lambda$: tree size, leaf size and generalisation
#===================================================================================================
rows = []
for gam in [0.0, 0.5, 2.0]:
    for lam in [0.0, 1.0, 10.0]:
        m_ = XGBoostClassifier(n_estimators=40, max_depth=4, learning_rate=0.3, reg_lambda=lam, gamma=gam, random_state=1)
        m_.fit(X_tr, y_tr)
        all_trees = [t for k in m_.trees for t in m_.trees[k]]
        mean_leaves = np.mean([leaves(t.root) for t in all_trees])
        max_abs_leaf = np.max([abs(nd.value) for t in all_trees for nd in iter_nodes(t.root) if nd.value is not None])
        P_ = m_.predict_proba(X_te)
        rows.append((gam, lam, mean_leaves, max_abs_leaf, log_loss(P_, y_te), np.mean(P_.argmax(1) == y_te)))
print(" γ    λ   | mean leaves/tree | max |leaf value| | test log-loss | test acc")
for r_ in rows:
    print(f"{r_[0]:>4} {r_[1]:>4} | {r_[2]:>16.2f} | {r_[3]:>16.3f} | {r_[4]:>13.3f} | {r_[5]:.3f}")


#===================================================================================================
# %% [Code cell 13  |  notebook cell 25]  7. Visual experiments  >  7.2 Learning curves, and a comparison with plain gradient boosting
#===================================================================================================
sys.path.insert(0, str(NB_DIR.parent / "GradientBoostClassification"))
from GradientBoost import GradientBoostingClassifier            # sibling folder's plain gradient boosting

def staged_proba_gbc(model, X, T):                               # log-prior start
    raw = np.tile(model.init_raw_scores_, (len(X), 1))
    for k in range(model.n_classes_):
        for tree in model.trees[k][:T]:
            raw[:, k] += model.learning_rate * tree.predict(X)
    return model.softmax(raw)

T_max, Ts = 80, np.arange(0, 81, 4)
models = {"plain GB (log-prior start, variance splits)": GradientBoostingClassifier(n_estimators=T_max, max_depth=3, learning_rate=0.3),
          "XGBoost-style, λ=0, γ=0": XGBoostClassifier(n_estimators=T_max, max_depth=3, learning_rate=0.3, reg_lambda=0.0, random_state=1),
          "XGBoost-style, λ=1, γ=0.5": XGBoostClassifier(n_estimators=T_max, max_depth=3, learning_rate=0.3, reg_lambda=1.0, gamma=0.5, random_state=1),
          "XGBoost-style, λ=1, γ=0.5, subsample=0.7": XGBoostClassifier(n_estimators=T_max, max_depth=3, learning_rate=0.3, reg_lambda=1.0, gamma=0.5, subsample=0.7, random_state=1)}
fig, ax = plt.subplots(figsize=(8, 4.6)); summary = {}
for (name, m_), col in zip(models.items(), CLASS_COLORS):
    m_.fit(X_tr, y_tr)
    sp = staged_proba_gbc if isinstance(m_, GradientBoostingClassifier) else staged_proba_xgb
    ll = [log_loss(sp(m_, X_te, T), y_te) for T in Ts]
    assert np.allclose(sp(m_, X_te, T_max), m_.predict_proba(X_te))
    summary[name] = (min(ll), Ts[int(np.argmin(ll))], ll[-1], np.mean(np.array(m_.predict(X_te)) == y_te))
    ax.plot(Ts, ll, color=col, label=name)
ax.set(title="Test log-loss vs. rounds (depth 3, η = 0.3)", xlabel="rounds", ylabel="test log-loss", ylim=(0.4, 1.3)); ax.legend(fontsize=8); plt.show()
for name, (best, at, last, acc_) in summary.items():
    print(f"{name:<44}: best {best:.3f} at {at:>2} rounds, final {last:.3f}, final accuracy {acc_:.3f}")


#===================================================================================================
# %% [Code cell 14  |  notebook cell 27]  7. Visual experiments  >  7.3 Decision regions under strong and weak regularisation
#===================================================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
for ax, (lam, gam) in zip(axes, [(0.0, 0.0), (1.0, 0.5), (10.0, 2.0)]):
    m_ = XGBoostClassifier(n_estimators=60, max_depth=4, learning_rate=0.3, reg_lambda=lam, gamma=gam, random_state=1); m_.fit(X_tr, y_tr)
    P_ = m_.predict_proba(X_te)
    plot_decision_regions(m_.predict, X_tr, y_tr, ax=ax, grid_n=90, legend=(lam == 0),
                          title=f"λ={lam}, γ={gam}: test log-loss {log_loss(P_, y_te):.3f}, acc {np.mean(P_.argmax(1) == y_te):.3f}")
plt.tight_layout(); plt.show()
