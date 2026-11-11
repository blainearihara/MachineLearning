# -*- coding: utf-8 -*-
"""
GradientBoostClassification_Walkthrough.py -- the code cells of GradientBoostClassification_Walkthrough.ipynb, extracted for step-through debugging.

How to use
- Open this file side by side with GradientBoostClassification_Walkthrough.ipynb. Each "# %%" block below is one code cell of the
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
assert (NB_DIR / "GradientBoost.py").exists(), "Open/run this notebook from inside the GradientBoostClassification/ folder"
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
from GradientBoost import GradientBoostingClassifier
from DecisionTreeGradientBoost import DecisionTreeRegressor as NewtonTree

print("GradientBoostingClassifier methods:", [k for k in vars(GradientBoostingClassifier) if not k.startswith("__")])
print("Newton regression tree methods:   ", [k for k in vars(NewtonTree) if not k.startswith("__")])

def leaf_of(x, node):                       # Illustration helper: the leaf (Node) a row reaches
    while node.value is None:
        node = node.left if x[node.feature_index] <= node.threshold else node.right
    return node

def log_loss(P, y):                         # Illustration helper: mean cross-entropy, y = integer labels
    return -np.mean(np.log(np.clip(P[np.arange(len(y)), y.astype(int)], 1e-15, 1)))


#===================================================================================================
# %% [Code cell 4  |  notebook cell 7]  2. The problem: class probabilities, not just labels
#===================================================================================================
centers = [[0.0, 0.0], [2.2, 0.0], [1.1, 1.9]]
X_tr, y_tr = make_blobs(240, centers=centers, std=0.9)
X_te, y_te = make_blobs(1000, centers=centers, std=0.9)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for k in range(3):
    ax.scatter(X_tr[y_tr == k, 0], X_tr[y_tr == k, 1], s=16, c=CLASS_COLORS[k], edgecolor="white", linewidth=0.4, label=f"class {k}")
ax.set(title="Three overlapping classes (n = 240)", xlabel="$x_0$", ylabel="$x_1$"); ax.legend(); plt.show()
print("class frequencies π_k:", np.round(np.bincount(y_tr.astype(int)) / len(y_tr), 3))


#===================================================================================================
# %% [Code cell 5  |  notebook cell 9]  4. Core concepts, theory and math  >  4.1 Softmax and the log-prior start
#===================================================================================================
gbc0 = GradientBoostingClassifier(n_estimators=0); gbc0.fit(X_tr, y_tr)
pi = np.bincount(y_tr.astype(int)) / len(y_tr)
print("init_raw_scores_ (ln π_k):", np.round(gbc0.init_raw_scores_, 4), " ln π:", np.round(np.log(pi), 4))
P0 = gbc0.predict_proba(X_te[:3])
print("predict_proba with no trees:", np.round(P0, 4))
assert np.allclose(gbc0.init_raw_scores_, np.log(pi)) and np.allclose(P0, pi)
S = rng.normal(size=(5, 3)) * 50
assert np.allclose(gbc0.softmax(S), gbc0.softmax(S + 1000.0)) and np.allclose(gbc0.softmax(S).sum(axis=1), 1)
print("✓ log-prior start reproduces π (Eq. 3); softmax is shift-invariant and stable even for scores ~ ±1000")


#===================================================================================================
# %% [Code cell 6  |  notebook cell 11]  4. Core concepts, theory and math  >  4.2 The gradient of cross-entropy with softmax: $p - y$
#===================================================================================================
# Illustration only — finite-difference check of Eq. (4) for one sample
F_i = rng.normal(size=3); y_i = np.array([0.0, 1.0, 0.0])
Li = lambda F: -np.sum(y_i * np.log(gbc0.softmax(F[None, :])[0]))
p_i = gbc0.softmax(F_i[None, :])[0]
for k in range(3):
    e = np.zeros(3); e[k] = 1e-6
    fd = (Li(F_i + e) - Li(F_i - e)) / 2e-6
    print(f"k = {k}: dL/dF_k ≈ {fd:+.6f}   p_k - y_k = {p_i[k] - y_i[k]:+.6f}")
    assert np.isclose(fd, p_i[k] - y_i[k], atol=1e-6)


#===================================================================================================
# %% [Code cell 7  |  notebook cell 13]  4. Core concepts, theory and math  >  4.4 Newton leaf values and the $(K-1)/K$ factor
#===================================================================================================
nt = NewtonTree(n_classes=3)
r_ = rng.uniform(-1, 1, 12); p_ = rng.uniform(0.05, 0.95, 12)
print(f"calculate_leaf_value = {nt.calculate_leaf_value(r_, p_):.6f}   Eq.(7) = {(2/3) * r_.sum() / np.sum(p_ * (1 - p_)):.6f}")
assert np.isclose(nt.calculate_leaf_value(r_, p_), (2 / 3) * r_.sum() / np.sum(p_ * (1 - p_)))

# Inside a fitted model: round 1 starts from the log-prior probabilities π
gbc1 = GradientBoostingClassifier(n_estimators=1, max_depth=2, learning_rate=0.5); gbc1.fit(X_tr, y_tr)
for k in range(3):
    tree_k = gbc1.trees[k][0]; r_k = (y_tr == k) - pi[k]; p_k = np.full(len(y_tr), pi[k])
    groups = {}
    for x_i, ri, pi_i in zip(X_tr, r_k, p_k):
        groups.setdefault(id(leaf_of(x_i, tree_k.root)), []).append((ri, pi_i))
    for nd in iter_nodes(tree_k.root):
        if nd.value is not None:
            rr, pp = np.array(groups[id(nd)]).T
            assert np.isclose(nd.value, (2 / 3) * rr.sum() / np.sum(pp * (1 - pp)))
print("✓ every leaf of every round-1 tree equals (K-1)/K · Σr / Σp(1-p) over the samples it contains (Eq. 7)")


#===================================================================================================
# %% [Code cell 8  |  notebook cell 15]  4. Core concepts, theory and math  >  4.5 The binary case: two mirror-image trees, one log-odds Newton step
#===================================================================================================
Xb, yb = make_moons(200, noise=0.3)
gbin = GradientBoostingClassifier(n_estimators=3, max_depth=2, learning_rate=0.5); gbin.fit(Xb, yb)
for t in range(3):
    t0, t1 = gbin.trees[0][t], gbin.trees[1][t]
    assert np.allclose(t0.predict(Xb), -t1.predict(Xb))
    assert [(n.feature_index, n.threshold) for n in iter_nodes(t0.root) if n.value is None] == \
           [(n.feature_index, n.threshold) for n in iter_nodes(t1.root) if n.value is None]
# the log-odds update of round 1 equals the binary Newton step Σ(y-p)/Σp(1-p) per leaf
p1 = np.full(len(yb), np.mean(yb)); r = yb - p1; groups = {}
for x_i, ri, pi_i in zip(Xb, r, p1):
    groups.setdefault(id(leaf_of(x_i, gbin.trees[1][0].root)), []).append((ri, pi_i))
for nd in iter_nodes(gbin.trees[1][0].root):
    if nd.value is not None:
        rr, pp = np.array(groups[id(nd)]).T
        assert np.isclose(2 * nd.value, rr.sum() / np.sum(pp * (1 - pp)))
print("✓ binary case: identical splits, mirror-image leaves, and v1 - v0 = Σ(y-p)/Σp(1-p): one log-odds Newton step")


#===================================================================================================
# %% [Code cell 9  |  notebook cell 17]  4. Core concepts, theory and math  >  4.5 The binary case: two mirror-image trees, one log-odds Newton step
#===================================================================================================
gbin.trees[1][0].print_tree()          # used to raise AttributeError: ... no attribute 'root1'


#===================================================================================================
# %% [Code cell 10  |  notebook cell 19]  6. Worked numerical example: two rounds, three classes
#===================================================================================================
x_toy = np.arange(1, 10, dtype=float).reshape(-1, 1)
y_toy = np.array([0, 0, 0, 1, 1, 2, 1, 2, 2], dtype=float)
toy = GradientBoostingClassifier(n_estimators=2, max_depth=1, learning_rate=0.5); toy.fit(x_toy, y_toy)
Y1 = np.eye(3)[y_toy.astype(int)]
raw = np.tile(toy.init_raw_scores_, (9, 1))
print("π =", np.round(np.bincount(y_toy.astype(int)) / 9, 4), " F^(0) = ln π =", np.round(toy.init_raw_scores_, 4))
for t in range(2):
    P = toy.softmax(raw)
    print(f"\n=== round {t + 1}: probabilities p_k(x) at the start of the round")
    print(np.round(P, 3))
    for k in range(3):
        tree = toy.trees[k][t]; r = Y1[:, k] - P[:, k]; left = x_toy.ravel() <= tree.root.threshold
        num_L, den_L = r[left].sum(), np.sum(P[left, k] * (1 - P[left, k]))
        num_R, den_R = r[~left].sum(), np.sum(P[~left, k] * (1 - P[~left, k]))
        print(f" class {k}: residuals {np.round(r, 3)}")
        print(f"          stump x_0 <= {tree.root.threshold}:  left  (2/3)·{num_L:+.3f}/{den_L:.3f} = {tree.root.left.value:+.4f}"
              f"   right (2/3)·{num_R:+.3f}/{den_R:.3f} = {tree.root.right.value:+.4f}")
        assert np.isclose(tree.root.left.value, (2/3) * num_L / den_L) and np.isclose(tree.root.right.value, (2/3) * num_R / den_R)
        raw[:, k] += 0.5 * tree.predict(x_toy)
    P_new = toy.softmax(raw)
    print(f" after round {t + 1}: predicted classes {np.argmax(P_new, axis=1).tolist()}  (true {y_toy.astype(int).tolist()}),"
          f" log-loss {log_loss(P_new, y_toy):.4f}")
assert np.allclose(toy.softmax(raw), toy.predict_proba(x_toy))
print("\n✓ hand-computed probabilities after 2 rounds equal toy.predict_proba()")


#===================================================================================================
# %% [Code cell 11  |  notebook cell 21]  7. Visual experiments  >  7.1 Probability surfaces as rounds are added
#===================================================================================================
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.2); gbc.fit(X_tr, y_tr)
g0_, g1_ = np.meshgrid(np.linspace(-2.5, 4.5, 90), np.linspace(-2.5, 4.3, 90)); grid = np.c_[g0_.ravel(), g1_.ravel()]
rgb = np.array([[0.12, 0.47, 0.71], [0.84, 0.15, 0.16], [0.17, 0.63, 0.17]])

def staged_proba(model, X, T):
    """Illustration only: probabilities after the first T rounds (same arithmetic as predict_proba)."""
    raw_ = np.tile(model.init_raw_scores_, (len(X), 1))
    for k in range(model.n_classes_):
        for tree in model.trees[k][:T]:
            raw_[:, k] += model.learning_rate * tree.predict(X)
    return model.softmax(raw_)
assert np.allclose(staged_proba(gbc, X_te, 100), gbc.predict_proba(X_te))

fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
for ax, T in zip(axes, [0, 1, 10, 100]):
    Pg = staged_proba(gbc, grid, T)
    ax.imshow((Pg @ rgb).reshape(g0_.shape + (3,)), origin="lower", extent=[-2.5, 4.5, -2.5, 4.3], aspect="auto")
    for k in range(3):
        ax.scatter(X_tr[y_tr == k, 0], X_tr[y_tr == k, 1], s=8, c=CLASS_COLORS[k], edgecolor="white", linewidth=0.3)
    Pte = staged_proba(gbc, X_te, T)
    ax.set(title=f"{T} rounds: test log-loss {log_loss(Pte, y_te):.3f}, acc {np.mean(Pte.argmax(1) == y_te):.3f}", xlabel="$x_0$"); ax.grid(False)
axes[0].set_ylabel("$x_1$"); plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 12  |  notebook cell 23]  7. Visual experiments  >  7.2 Learning curves: log-loss and accuracy for several learning rates
#===================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
curves = {}
for eta_, col in zip([1.0, 0.3, 0.1], CLASS_COLORS):
    m_ = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=eta_); m_.fit(X_tr, y_tr)
    Ts = np.arange(0, 121, 4)
    tr_ll = [log_loss(staged_proba(m_, X_tr, T), y_tr) for T in Ts]
    te_ll = [log_loss(staged_proba(m_, X_te, T), y_te) for T in Ts]
    te_acc = [np.mean(staged_proba(m_, X_te, T).argmax(1) == y_te) for T in Ts]
    curves[eta_] = (Ts, te_ll, te_acc)
    axes[0].plot(Ts, te_ll, color=col, label=f"η = {eta_}: test"); axes[0].plot(Ts, tr_ll, color=col, ls=":", label=f"η = {eta_}: train")
    axes[1].plot(Ts, te_acc, color=col, label=f"η = {eta_}")
axes[0].set(title="Log-loss (dotted = train, solid = test)", xlabel="rounds", ylabel="mean cross-entropy", ylim=(0, 1.2)); axes[0].legend(ncol=2, fontsize=8)
axes[1].set(title="Test accuracy", xlabel="rounds", ylabel="accuracy"); axes[1].legend()
plt.tight_layout(); plt.show()
for eta_, (Ts, te_ll, te_acc) in curves.items():
    i = int(np.argmin(te_ll))
    print(f"η = {eta_}: best test log-loss {te_ll[i]:.3f} at {Ts[i]} rounds, final {te_ll[-1]:.3f};  test accuracy best {max(te_acc):.3f}, final {te_acc[-1]:.3f}")


#===================================================================================================
# %% [Code cell 13  |  notebook cell 25]  7. Visual experiments  >  7.3 Are the probabilities trustworthy? A reliability diagram
#===================================================================================================
def reliability(P, y, bins=8):
    conf = P.max(axis=1); correct = P.argmax(axis=1) == y
    edges = np.linspace(1 / 3, 1, bins + 1); idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return [(conf[idx == b].mean(), correct[idx == b].mean(), np.sum(idx == b)) for b in range(bins) if np.any(idx == b)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for T, col in [(15, CLASS_COLORS[0]), (120, CLASS_COLORS[1])]:
    m_ = GradientBoostingClassifier(n_estimators=T, max_depth=2, learning_rate=0.3); m_.fit(X_tr, y_tr)
    P_ = m_.predict_proba(X_te); rel = np.array(reliability(P_, y_te))
    axes[0].plot(rel[:, 0], rel[:, 1], "o-", color=col, label=f"{T} rounds (test log-loss {log_loss(P_, y_te):.3f})")
    axes[1].hist(P_.max(axis=1), bins=20, range=(1/3, 1), alpha=0.5, color=col, label=f"{T} rounds")
axes[0].plot([1/3, 1], [1/3, 1], "k--", label="perfect calibration")
axes[0].set(title="Reliability: confidence vs. accuracy (test)", xlabel="predicted probability of the predicted class", ylabel="observed accuracy"); axes[0].legend()
axes[1].set(title="Distribution of confidence (test)", xlabel="max predicted probability", ylabel="count"); axes[1].legend()
plt.tight_layout(); plt.show()
