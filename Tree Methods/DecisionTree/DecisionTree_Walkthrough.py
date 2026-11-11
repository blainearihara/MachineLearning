# -*- coding: utf-8 -*-
"""
DecisionTree_Walkthrough.py -- the code cells of DecisionTree_Walkthrough.ipynb, extracted for step-through debugging.

How to use
- Open this file side by side with DecisionTree_Walkthrough.ipynb. Each "# %%" block below is one code cell of the
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
# %% [Code cell 1  |  notebook cell 4]  1. Setup
#===================================================================================================
# Setup: imports, reproducibility, plot style, and the import path to the repo modules
import sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

NB_DIR = Path.cwd()                       # Jupyter starts the kernel in the notebook's own folder
assert (NB_DIR / "DecisionTree.py").exists(), "Open/run this notebook from inside the DecisionTree/ folder"
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
# %% [Code cell 2  |  notebook cell 5]  1. Setup
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
# %% [Code cell 3  |  notebook cell 6]  1. Setup
#===================================================================================================
# The implementation under study: imported from DecisionTree.py, never copied into this notebook
import DecisionTree as dt_mod
from DecisionTree import DecisionTreeClassifier, Node

print("module:", Path(dt_mod.__file__).name)
print("DecisionTreeClassifier methods:", [k for k in vars(DecisionTreeClassifier) if not k.startswith("__")])


#===================================================================================================
# %% [Code cell 4  |  notebook cell 8]  2. The problem: supervised classification
#===================================================================================================
# Motivating data set: noisy XOR (class 1 when x0*x1 < 0, then 10% of labels flipped)
X_xor, y_xor = make_xor(200, noise=0.10)
X_xor_test, y_xor_test = make_xor(2000, noise=0.10)      # large held-out set for estimating the risk

fig, ax = plt.subplots(figsize=(5.4, 4.8))
for k in (0, 1):
    m = y_xor == k
    ax.scatter(X_xor[m, 0], X_xor[m, 1], s=18, c=CLASS_COLORS[k], edgecolor="white", linewidth=0.4, label=f"class {k}")
ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
ax.set(title="Noisy XOR training data (n = 200)", xlabel="$x_0$", ylabel="$x_1$")
ax.legend(loc="upper right", frameon=True); plt.show()
print(f"class balance: {np.mean(y_xor == 1):.2f} of the training points are class 1")


#===================================================================================================
# %% [Code cell 5  |  notebook cell 10]  2. The problem: supervised classification
#===================================================================================================
# Illustration only — not the repo implementation.
# Best training accuracy achievable by any of 20,000 random straight lines w·x + b = 0
W = rng.normal(size=(20_000, 2))
b = rng.uniform(-1, 1, size=20_000)
side = (X_xor @ W.T + b) > 0                                   # (n, lines): which side each point falls on
acc = (side == y_xor[:, None]).mean(axis=0)
acc = np.maximum(acc, 1 - acc)                                 # either side may be labelled class 1
print(f"best of 20,000 linear rules: training accuracy = {acc.max():.3f}")
print(f"always predict the majority class:              {max(y_xor.mean(), 1 - y_xor.mean()):.3f}")


#===================================================================================================
# %% [Code cell 6  |  notebook cell 12]  3. Why a decision tree fits this problem
#===================================================================================================
# Preview: the repo's tree on the XOR data
clf_preview = DecisionTreeClassifier(min_samp_split=2, max_depth=3)
clf_preview.fit(X_xor, y_xor.reshape(-1, 1))            # y as a column vector (n, 1); a 1-D vector also works
train_acc = np.mean(np.array(clf_preview.predict(X_xor)) == y_xor)
test_acc = np.mean(np.array(clf_preview.predict(X_xor_test)) == y_xor_test)
print(f"tree depth = {tree_depth(clf_preview.root)}, leaves = {count_leaves(clf_preview.root)}, "
      f"train accuracy = {train_acc:.3f}, test accuracy = {test_acc:.3f}\n")
clf_preview.print_tree()
plot_decision_regions(clf_preview.predict, X_xor, y_xor, title="Repo DecisionTreeClassifier, max_depth=3")
plt.show()


#===================================================================================================
# %% [Code cell 7  |  notebook cell 14]  3. Why a decision tree fits this problem
#===================================================================================================
# Check: strictly increasing transforms of the features leave predictions unchanged
Xm, ym = make_moons(120, noise=0.25)
Xtr, Xte, ytr, yte = train_test_split_np(Xm, ym, test_frac=0.5)
phi = lambda Z: np.c_[np.exp(Z[:, 0]), Z[:, 1] ** 3]          # exp and cube are strictly increasing

tree_raw = DecisionTreeClassifier(min_samp_split=2, max_depth=3); tree_raw.fit(Xtr, ytr.reshape(-1, 1))
tree_phi = DecisionTreeClassifier(min_samp_split=2, max_depth=3); tree_phi.fit(phi(Xtr), ytr.reshape(-1, 1))
pred_raw, pred_phi = np.array(tree_raw.predict(Xte)), np.array(tree_phi.predict(phi(Xte)))
assert np.array_equal(pred_raw, pred_phi)
print(f"✓ identical predictions on all {len(Xte)} test points after transforming the features")


#===================================================================================================
# %% [Code cell 8  |  notebook cell 16]  4. Core concepts, theory and math  >  4.1 Impurity: measuring how mixed a node is
#===================================================================================================
# Illustration only — the impurity measures for K = 2 as functions of p = P(class 1)
p = np.linspace(0, 1, 501)
with np.errstate(divide="ignore", invalid="ignore"):
    H = -(np.where(p > 0, p * np.log2(p), 0.0) + np.where(p < 1, (1 - p) * np.log2(1 - p), 0.0))
G = 1 - p**2 - (1 - p)**2
E = 1 - np.maximum(p, 1 - p)

fig, ax = plt.subplots()
ax.plot(p, H, lw=2, label="entropy $H$ in bits, Eq. (2)")
ax.plot(p, G, lw=2, label="Gini $G$, Eq. (1)")
ax.plot(p, E, "--", lw=1.5, color="gray", label="misclassification rate $1-\\max_k p_k$")
ax.set(title="Impurity of a two-class node", xlabel="$p$ = proportion of class 1 in the node", ylabel="impurity")
ax.legend(); plt.show()


#===================================================================================================
# %% [Code cell 9  |  notebook cell 18]  4. Core concepts, theory and math  >  4.1 Impurity: measuring how mixed a node is
#===================================================================================================
# Illustration only — Gini as "probability that two labels drawn at random from the node disagree"
labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=float)     # a node with K = 3 classes
draws = rng.integers(0, len(labels), size=(200_000, 2))            # two independent draws, with replacement
mc = np.mean(labels[draws[:, 0]] != labels[draws[:, 1]])
p_k = np.bincount(labels.astype(int)) / len(labels)
print("class proportions p_k:", p_k)
print(f"Monte-Carlo P(two draws disagree) = {mc:.4f}   vs   1 - sum_k p_k^2 = {1 - np.sum(p_k**2):.4f}")


#===================================================================================================
# %% [Code cell 10  |  notebook cell 20]  4. Core concepts, theory and math  >  4.1 Impurity: measuring how mixed a node is
#===================================================================================================
clf = DecisionTreeClassifier()          # fresh instance, used only for its helper methods
for y_node in [labels, np.array([1., 1, 1, 1]), np.array([0., 1, 0, 1]), np.array([0., 1, 2, 3])]:
    print(f"y = {str(y_node):<34} gini_index = {clf.gini_index(y_node):.4f}   entropy = {clf.entropy(y_node):.4f}")


#===================================================================================================
# %% [Code cell 11  |  notebook cell 21]  4. Core concepts, theory and math  >  4.1 Impurity: measuring how mixed a node is
#===================================================================================================
# Check — independent formula versions of Eqs. (1) and (2) (illustration code, not the repo's)
def gini_formula(y):
    p = np.unique(y, return_counts=True)[1] / len(y)
    return 1 - np.sum(p**2)

def entropy_formula(y):
    p = np.unique(y, return_counts=True)[1] / len(y)
    return -np.sum(p * np.log2(p))

for _ in range(200):
    y_rand = rng.integers(0, 4, size=rng.integers(1, 30)).astype(float)
    assert np.isclose(clf.gini_index(y_rand), gini_formula(y_rand))
    assert np.isclose(clf.entropy(y_rand), entropy_formula(y_rand))
uniform4 = np.array([0., 1, 2, 3])
assert np.isclose(clf.gini_index(uniform4), 1 - 1/4) and np.isclose(clf.entropy(uniform4), np.log2(4))
print("✓ gini_index and entropy agree with Eqs. (1)-(2) on 200 random label vectors")
print("✓ at the uniform distribution with K = 4 they reach the derived maxima 1 - 1/K = 0.75 and log2 K = 2")


#===================================================================================================
# %% [Code cell 12  |  notebook cell 23]  4. Core concepts, theory and math  >  4.2 Splits and information gain
#===================================================================================================
# Illustration only — Eq. (3) computed by hand
y_parent = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)
y_left, y_right = y_parent[:5], y_parent[5:]                  # left = four 0s and one 1, right = five 1s
wL, wR = len(y_left) / len(y_parent), len(y_right) / len(y_parent)
gP, gL, gR = gini_formula(y_parent), gini_formula(y_left), gini_formula(y_right)
manual = gP - (wL * gL + wR * gR)
print(f"G(S) = {gP:.4f},  G(S_L) = {gL:.4f},  G(S_R) = {gR:.4f}")
print(f"Gini gain = {gP:.4f} - ({wL:.1f}·{gL:.4f} + {wR:.1f}·{gR:.4f}) = {manual:.4f}")


#===================================================================================================
# %% [Code cell 13  |  notebook cell 25]  4. Core concepts, theory and math  >  4.2 Splits and information gain
#===================================================================================================
g_gini = clf.information_gain(y_parent, y_left, y_right, "gini")
g_ent  = clf.information_gain(y_parent, y_left, y_right, "entropy")
print(f"information_gain(..., 'gini')    = {g_gini:.4f}")
print(f"information_gain(..., 'entropy') = {g_ent:.4f}")


#===================================================================================================
# %% [Code cell 14  |  notebook cell 26]  4. Core concepts, theory and math  >  4.2 Splits and information gain
#===================================================================================================
# Check — Eq. (3) for both impurities, and non-negativity (Jensen) over many random splits
assert np.isclose(g_gini, manual)
ent_manual = entropy_formula(y_parent) - (wL * entropy_formula(y_left) + wR * entropy_formula(y_right))
assert np.isclose(g_ent, ent_manual)

gains_random = []
for _ in range(500):
    y_ = rng.integers(0, 3, size=20).astype(float)
    s = rng.integers(1, 20)                                  # split position: first s samples go left
    for mode in ("gini", "entropy"):
        gains_random.append(clf.information_gain(y_, y_[:s], y_[s:], mode))
assert min(gains_random) > -1e-12
print(f"✓ information_gain matches Eq. (3) for both modes")
print(f"✓ smallest gain over {len(gains_random)} random splits = {min(gains_random):.2e}  (zero up to floating-point rounding)")


#===================================================================================================
# %% [Code cell 15  |  notebook cell 28]  4. Core concepts, theory and math  >  4.3 Candidate thresholds and the exhaustive split search
#===================================================================================================
# A simple two-class data set with one informative direction
Xb, yb = make_blobs(40, centers=[[-1.0, 0.0], [1.2, 0.6]], std=0.8)

# Illustration only — exhaustive scan of Eqs. (4)-(5) written independently of the repo
def scan_gains(X, y):
    rows = []
    for j in range(X.shape[1]):
        for tau in np.unique(X[:, j]):                        # candidate thresholds = distinct observed values
            left = X[:, j] <= tau
            if 0 < left.sum() < len(y):                        # both children non-empty
                g = gini_formula(y) - (left.mean() * gini_formula(y[left]) + (1 - left.mean()) * gini_formula(y[~left]))
                rows.append((j, tau, g))
    return rows

scan = scan_gains(Xb, yb)
j_s, tau_s, g_s = max(scan, key=lambda r: r[2])               # max() keeps the FIRST maximum, like the repo

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for k in (0, 1):
    axes[0].scatter(Xb[yb == k, 0], Xb[yb == k, 1], c=CLASS_COLORS[k], s=22, edgecolor="white", label=f"class {k}")
(axes[0].axvline if j_s == 0 else axes[0].axhline)(tau_s, color="k", ls="--", label=f"best split $x_{j_s} \\leq {tau_s:.2f}$")
axes[0].set(title="Data and the winning split", xlabel="$x_0$", ylabel="$x_1$"); axes[0].legend()
for j in (0, 1):
    rows = [(t, g) for jj, t, g in scan if jj == j]
    axes[j + 1].plot(*zip(*rows), "o-", ms=3, color=CLASS_COLORS[j + 2])
    if j == j_s:
        axes[j + 1].plot(tau_s, g_s, "k*", ms=16, label="maximum over all candidates")
        axes[j + 1].legend()
    axes[j + 1].set(title=f"Gini gain of every threshold on $x_{j}$", xlabel=f"threshold $\\tau$ on $x_{j}$", ylabel="gain $\\Delta$")
plt.tight_layout(); plt.show()
print(f"scan winner: feature {j_s}, threshold {tau_s:.4f}, gain {g_s:.4f}")


#===================================================================================================
# %% [Code cell 16  |  notebook cell 30]  4. Core concepts, theory and math  >  4.3 Candidate thresholds and the exhaustive split search
#===================================================================================================
best = clf.get_best_split(np.c_[Xb, yb], *Xb.shape)
print({k: (f"array of shape {v.shape}" if isinstance(v, np.ndarray) else v) for k, v in best.items()})


#===================================================================================================
# %% [Code cell 17  |  notebook cell 31]  4. Core concepts, theory and math  >  4.3 Candidate thresholds and the exhaustive split search
#===================================================================================================
# Check — the repo's search returns the arg-max of the independent scan, Eq. (5)
assert best["feature_index"] == j_s and best["threshold"] == tau_s
assert np.isclose(best["info_gain"], g_s)
assert len(best["dataset_left"]) == np.sum(Xb[:, j_s] <= tau_s) and len(best["dataset_right"]) == np.sum(Xb[:, j_s] > tau_s)
print(f"✓ get_best_split chose x_{j_s} <= {tau_s:.4f} with gain {g_s:.4f}: the maximum over all {len(scan)} candidates")


#===================================================================================================
# %% [Code cell 18  |  notebook cell 34]  4. Core concepts, theory and math  >  4.3 Candidate thresholds and the exhaustive split search
#===================================================================================================
yL_best, yR_best = best["dataset_left"][:, -1], best["dataset_right"][:, -1]
gini_g = clf.information_gain(yb, yL_best, yR_best, "gini")
ent_g  = clf.information_gain(yb, yL_best, yR_best, "entropy")
print(f"default (gini) tree : stored info_gain = {best['info_gain']:.4f} | Gini gain of that split = {gini_g:.4f}")
assert np.isclose(best["info_gain"], gini_g)

clf_ent = DecisionTreeClassifier(criterion="entropy")
best_ent = clf_ent.get_best_split(np.c_[Xb, yb], *Xb.shape)
ent_check = clf_ent.information_gain(yb, best_ent["dataset_left"][:, -1], best_ent["dataset_right"][:, -1], "entropy")
print(f"criterion='entropy': chose x_{best_ent['feature_index']} <= {best_ent['threshold']:.4f}, stored gain {best_ent['info_gain']:.4f} = entropy gain {ent_check:.4f}")
assert np.isclose(best_ent["info_gain"], ent_check)


#===================================================================================================
# %% [Code cell 19  |  notebook cell 36]  4. Core concepts, theory and math  >  4.3 Candidate thresholds and the exhaustive split search
#===================================================================================================
# Two training points: x = 1 (class 0) and x = 3 (class 1). Where does the repo put the boundary?
t2 = DecisionTreeClassifier(min_samp_split=2, max_depth=1)
t2.fit(np.array([[1.0], [3.0]]), np.array([[0.0], [1.0]]))
t2.print_tree()
probe = np.array([[1.0], [1.1], [2.0], [2.9]])
print("x          :", probe.ravel())
print("prediction :", np.array(t2.predict(probe)))


#===================================================================================================
# %% [Code cell 20  |  notebook cell 38]  4. Core concepts, theory and math  >  4.4 Recursion, stopping rules and leaf values
#===================================================================================================
# Illustration only — Eq. (6) via counting
def majority_formula(y):
    return float(np.argmax(np.bincount(y.astype(int))))

leaf_examples = [np.array([2., 0, 2, 1, 2]), np.array([0., 0, 1]), np.array([1., 0, 0, 1]), np.array([0., 1, 1, 0])]
for y_leaf in leaf_examples:
    print(f"y = {str(y_leaf):<22} majority by counting = {majority_formula(y_leaf)}   calculate_leaf_value = {clf.calculate_leaf_value(y_leaf)}")


#===================================================================================================
# %% [Code cell 21  |  notebook cell 39]  4. Core concepts, theory and math  >  4.4 Recursion, stopping rules and leaf values
#===================================================================================================
# Check — agreement on 300 random label vectors, ties included
for _ in range(300):
    y_rand = rng.integers(0, 3, size=rng.integers(1, 15)).astype(float)
    assert clf.calculate_leaf_value(y_rand) == majority_formula(y_rand)
print("✓ calculate_leaf_value returns the majority class of Eq. (6), with ties going to the smaller label")


#===================================================================================================
# %% [Code cell 22  |  notebook cell 42]  4. Core concepts, theory and math  >  4.4 Recursion, stopping rules and leaf values
#===================================================================================================
# Requested max_depth vs. the depth of the tree actually grown on the XOR data
print("max_depth | actual depth | leaves")
for md in [0, 1, 2, 3, 4, 5, None]:
    t = DecisionTreeClassifier(min_samp_split=2, max_depth=md)
    t.fit(X_xor, y_xor.reshape(-1, 1))
    print(f"{str(md):>9} | {tree_depth(t.root):>12} | {count_leaves(t.root):>6}")
    if md is not None:
        assert tree_depth(t.root) == md
print("✓ every tree is exactly as deep as max_depth allows (max_depth=0 is a single leaf)")


#===================================================================================================
# %% [Code cell 23  |  notebook cell 44]  4. Core concepts, theory and math  >  4.4 Recursion, stopping rules and leaf values
#===================================================================================================
cases = [(np.array([[1.0], [1.0], [1.0]]), np.array([[0.0], [1.0], [1.0]]), "identical x, mixed labels"),
         (np.array([[2.0, 5.0], [2.0, 5.0]]), np.array([[1.0], [1.0]]), "identical x, pure node")]
for X_dup, y_dup, desc in cases:
    t = DecisionTreeClassifier(min_samp_split=2, max_depth=3)
    t.fit(X_dup, y_dup)
    print(f"{desc:<26}: fit succeeded -> a single leaf predicting {t.root.value}  (depth {tree_depth(t.root)})")
    assert t.root.value is not None


#===================================================================================================
# %% [Code cell 24  |  notebook cell 46]  4. Core concepts, theory and math  >  4.5 Prediction: routing a point to its leaf
#===================================================================================================
# Illustration only — route a point through the Node objects by hand, recording each comparison
def route(x, node):
    path = []
    while node.value is None:                                  # decision node
        go_left = x[node.feature_index] <= node.threshold
        path.append(f"x_{node.feature_index}={x[node.feature_index]:+.3f} {'<=' if go_left else '> '} {node.threshold:+.3f}")
        node = node.left if go_left else node.right
    return node.value, path

value, path = route(X_xor_test[0], clf_preview.root)
print("point:", X_xor_test[0], "\npath :", "  ->  ".join(path), f"\nleaf value: {value}")


#===================================================================================================
# %% [Code cell 25  |  notebook cell 48]  4. Core concepts, theory and math  >  4.5 Prediction: routing a point to its leaf
#===================================================================================================
# Check — the repo's predict agrees with the hand traversal on all 2,000 test points
pred_repo = np.array(clf_preview.predict(X_xor_test))
pred_hand = np.array([route(x, clf_preview.root)[0] for x in X_xor_test])
assert np.array_equal(pred_repo, pred_hand)
print(f"✓ predict() == manual routing for all {len(X_xor_test)} test points")


#===================================================================================================
# %% [Code cell 26  |  notebook cell 50]  5. Implementation walkthrough, in the order the code runs
#===================================================================================================
t1d = DecisionTreeClassifier(max_depth=3); t1d.fit(X_xor, y_xor)               # 1-D labels
print("1-D labels: fit works; training accuracy =", np.mean(np.array(t1d.predict(X_xor)) == y_xor))

X_s = np.array([[9.0], [10.0], [11.0], [12.0]])           # perfectly separable: 9, 10 -> 'a'; 11, 12 -> 'b'
y_str = np.array(["a", "a", "b", "b"])                    # NumPy string labels (dtype <U1)
t_str = DecisionTreeClassifier(max_depth=2); t_str.fit(X_s, y_str)
t_str.print_tree()
print("predict([9.5, 11.5]) =", t_str.predict(np.array([[9.5], [11.5]])))
assert t_str.predict(np.array([[9.5], [11.5]])) == ["a", "b"] and t_str.root.threshold == 10.0


#===================================================================================================
# %% [Code cell 27  |  notebook cell 52]  6. Worked numerical example
#===================================================================================================
# The worked-example data set (sorted by x0 for readability)
X_toy = np.array([[1, 1], [3, 7], [4, 6], [5, 5], [6, 3], [7, 9], [8, 4], [9, 8]], dtype=float)
y_toy = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=float)
ds_toy = np.c_[X_toy, y_toy]                                  # the format build_tree works with: [x0, x1, y]

print("  i   x0   x1   y")
for i, (a, b_, c) in enumerate(ds_toy):
    print(f"{i:>3} {a:>4.0f} {b_:>4.0f} {c:>3.0f}")

fig, ax = plt.subplots(figsize=(4.8, 4.4))
for k in (0, 1):
    m = y_toy == k
    ax.scatter(X_toy[m, 0], X_toy[m, 1], s=90, c=CLASS_COLORS[k], edgecolor="k", label=f"class {k}")
for i, (a, b_) in enumerate(X_toy):
    ax.annotate(str(i), (a, b_), textcoords="offset points", xytext=(6, 5))
ax.set(title="Worked example: 8 samples", xlabel="$x_0$", ylabel="$x_1$", xlim=(0, 10), ylim=(0, 10))
ax.legend(loc="lower right", frameon=True); plt.show()


#===================================================================================================
# %% [Code cell 28  |  notebook cell 54]  6. Worked numerical example
#===================================================================================================
print(f"G(root) by hand       = 1 - (0.5^2 + 0.5^2) = {1 - 0.5**2 - 0.5**2}")
print(f"G(root) by gini_index = {clf.gini_index(y_toy)}")


#===================================================================================================
# %% [Code cell 29  |  notebook cell 56]  6. Worked numerical example
#===================================================================================================
def candidate_table(ds, title):
    """Print every candidate split of a node, scored with the repo's own split / gini_index / information_gain."""
    y = ds[:, -1]
    rows = []
    for j in range(ds.shape[1] - 1):
        for tau in np.unique(ds[:, j]):
            L, R = clf.split(ds, j, tau)
            if len(L) and len(R):
                rows.append((j, tau, len(L), len(R), clf.gini_index(L[:, -1]), clf.gini_index(R[:, -1]),
                             clf.information_gain(y, L[:, -1], R[:, -1], "gini")))
    best_gain = max(r[-1] for r in rows)
    print(title)
    print("  j   tau | |S_L| |S_R| |  G(S_L)  G(S_R) |   gain")
    for j, tau, nL, nR, gL_, gR_, g in rows:
        mark = "  <- best" if np.isclose(g, best_gain) else ""
        print(f"  {j} {tau:>5.0f} | {nL:>5} {nR:>5} | {gL_:>7.4f} {gR_:>7.4f} | {g:>6.4f}{mark}")

candidate_table(ds_toy, "Root node (8 samples, G = 0.5):")


#===================================================================================================
# %% [Code cell 30  |  notebook cell 58]  6. Worked numerical example
#===================================================================================================
root_split = clf.get_best_split(ds_toy, *X_toy.shape)
print(f"get_best_split: feature {root_split['feature_index']}, threshold {root_split['threshold']}, gain {root_split['info_gain']:.4f}")
print("left child rows :", root_split["dataset_left"][:, :2].tolist(), "labels", root_split["dataset_left"][:, -1])
print("right child rows:", root_split["dataset_right"][:, :2].tolist(), "labels", root_split["dataset_right"][:, -1])


#===================================================================================================
# %% [Code cell 31  |  notebook cell 60]  6. Worked numerical example
#===================================================================================================
candidate_table(root_split["dataset_left"], "Left child (samples 0-3, G = 0.375):")


#===================================================================================================
# %% [Code cell 32  |  notebook cell 62]  6. Worked numerical example
#===================================================================================================
candidate_table(root_split["dataset_right"], "Right child (samples 4-7, G = 0.375):")


#===================================================================================================
# %% [Code cell 33  |  notebook cell 64]  6. Worked numerical example
#===================================================================================================
toy_tree = DecisionTreeClassifier(min_samp_split=2, max_depth=5)
toy_tree.fit(X_toy, y_toy.reshape(-1, 1))
toy_tree.print_tree()

r = toy_tree.root
assert (r.feature_index, r.threshold) == (0, 5.0)
assert (r.left.feature_index, r.left.threshold, r.left.left.value, r.left.right.value) == (1, 6.0, 1.0, 0.0)
assert (r.right.feature_index, r.right.threshold, r.right.left.value, r.right.right.value) == (0, 8.0, 0.0, 1.0)
assert np.array_equal(np.array(toy_tree.predict(X_toy)), y_toy)
print("\n✓ fit() built exactly the tree derived by hand, and it classifies all 8 training samples correctly")


#===================================================================================================
# %% [Code cell 34  |  notebook cell 66]  6. Worked numerical example
#===================================================================================================
for x_new in np.array([[2.0, 2.0], [8.5, 6.0]]):
    value, path = route(x_new, toy_tree.root)
    print(f"x = {x_new}:  " + "  ->  ".join(path) + f"  ->  leaf {value}   (predict: {toy_tree.predict(x_new[None, :])[0]})")

fig, ax = plt.subplots(figsize=(4.8, 4.4))
plot_decision_regions(toy_tree.predict, X_toy, y_toy, ax=ax, title="Regions of the worked-example tree", pad=1.0, legend=False)
ax.scatter([2, 8.5], [2, 6], marker="X", s=140, c="k", label="new points")
ax.legend(loc="lower right", frameon=True); plt.show()


#===================================================================================================
# %% [Code cell 35  |  notebook cell 68]  7. Visual experiments  >  7.1 Decision regions as the tree grows
#===================================================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for ax, md in zip(axes.ravel(), [1, 2, 3, 5, 7, 11]):
    t = DecisionTreeClassifier(min_samp_split=2, max_depth=md); t.fit(X_xor, y_xor.reshape(-1, 1))
    tr = np.mean(np.array(t.predict(X_xor)) == y_xor); te = np.mean(np.array(t.predict(X_xor_test)) == y_xor_test)
    plot_decision_regions(t.predict, X_xor, y_xor, ax=ax, legend=(md == 0),
                          title=f"max_depth={md} ({count_leaves(t.root)} leaves)\ntrain {tr:.2f} / test {te:.2f}")
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 36  |  notebook cell 70]  7. Visual experiments  >  7.2 Training and test accuracy versus depth
#===================================================================================================
depths, tr_acc, te_acc = [], [], []
for md in range(1, 14):
    t = DecisionTreeClassifier(min_samp_split=2, max_depth=md); t.fit(X_xor, y_xor.reshape(-1, 1))
    depths.append(tree_depth(t.root))
    tr_acc.append(np.mean(np.array(t.predict(X_xor)) == y_xor))
    te_acc.append(np.mean(np.array(t.predict(X_xor_test)) == y_xor_test))

fig, ax = plt.subplots()
ax.plot(depths, tr_acc, "o-", label="training accuracy")
ax.plot(depths, te_acc, "s-", label="test accuracy (2,000 fresh points)")
ax.axhline(0.9, color="gray", ls="--", label="Bayes ceiling (10% label noise)")
ax.set(title="Noisy XOR: accuracy vs. tree depth", xlabel="tree depth", ylabel="accuracy")
ax.legend(); plt.show()
print(f"best test accuracy {max(te_acc):.3f} at depth {depths[int(np.argmax(te_acc))]}; "
      f"deepest tree: train {tr_acc[-1]:.3f}, test {te_acc[-1]:.3f}")


#===================================================================================================
# %% [Code cell 37  |  notebook cell 72]  7. Visual experiments  >  7.3 Greedy myopia: why XOR is hard for a one-step-ahead search
#===================================================================================================
X_x0, y_x0 = make_xor(400, noise=0.0)
root_scan = scan_gains(X_x0, y_x0)
left = X_x0[:, 0] <= 0
child_scan = scan_gains(X_x0[left], y_x0[left])

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for j in (0, 1):
    rows = [(t, g) for jj, t, g in root_scan if jj == j]
    axes[0].plot(*zip(*rows), ".", ms=3, color=CLASS_COLORS[j + 2], label=f"feature $x_{j}$")
    rows = [(t, g) for jj, t, g in child_scan if jj == j]
    axes[1].plot(*zip(*rows), ".", ms=3, color=CLASS_COLORS[j + 2], label=f"feature $x_{j}$")
axes[0].set(title="Root of noise-free XOR: gain of every split", xlabel="threshold $\\tau$", ylabel="Gini gain")
axes[1].set(title="Inside the child $x_0 \\leq 0$: gain of every split", xlabel="threshold $\\tau$")
for ax in axes: ax.legend()
plt.tight_layout(); plt.show()
print(f"largest root gain:  {max(g for *_, g in root_scan):.4f}")
print(f"largest child gain: {max(g for *_, g in child_scan):.4f}   (the maximum possible for a two-class node is 0.5)")


#===================================================================================================
# %% [Code cell 38  |  notebook cell 74]  7. Visual experiments  >  7.4 Gini versus entropy: do they choose different splits?
#===================================================================================================
def best_split_by(ds, mode):
    """Arg-max of Eq. (5) using the repo's split + information_gain with the given impurity mode."""
    y = ds[:, -1]; best_c, best_g = None, -np.inf
    for j in range(ds.shape[1] - 1):
        for tau in np.unique(ds[:, j]):
            L, R = clf.split(ds, j, tau)
            if len(L) and len(R):
                g = clf.information_gain(y, L[:, -1], R[:, -1], mode)
                if g > best_g:
                    best_c, best_g = (j, tau), g
    return best_c

agree, trials = 0, 200
for _ in range(trials):
    Xr, yr = make_moons(30, noise=0.35)
    ds = np.c_[Xr, yr]
    agree += best_split_by(ds, "gini") == best_split_by(ds, "entropy")
print(f"Gini and entropy chose the same root split in {agree}/{trials} random 30-point data sets ({agree/trials:.0%})")

# One example: both gain curves along x0, rescaled to a maximum of 1 so their shapes can be compared
Xr, yr = make_moons(60, noise=0.3); ds = np.c_[Xr, yr]
taus = np.unique(Xr[:, 0])[:-1]
curves = {mode: np.array([clf.information_gain(yr, *[part[:, -1] for part in clf.split(ds, 0, t)], mode) for t in taus])
          for mode in ("gini", "entropy")}
fig, ax = plt.subplots()
for mode, c in curves.items():
    ax.plot(taus, c / c.max(), label=f"{mode} gain (rescaled)")
ax.set(title="Gain along $x_0$ under the two criteria (moons, n = 60)", xlabel="threshold $\\tau$ on $x_0$", ylabel="gain / max gain")
ax.legend(); plt.show()


#===================================================================================================
# %% [Code cell 39  |  notebook cell 76]  7. Visual experiments  >  7.5 Instability: small data changes, different trees
#===================================================================================================
X_mo, y_mo = make_moons(150, noise=0.3)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
grid_preds = []
gx = np.c_[np.repeat(np.linspace(-1.5, 2.5, 60), 60), np.tile(np.linspace(-1.2, 1.7, 60), 60)]
for s, ax in enumerate(axes):
    idx = rng.choice(len(X_mo), size=105, replace=False)       # a different 70% subsample each time
    t = DecisionTreeClassifier(min_samp_split=2, max_depth=6); t.fit(X_mo[idx], y_mo[idx].reshape(-1, 1))
    grid_preds.append(np.array(t.predict(gx)))
    plot_decision_regions(t.predict, X_mo[idx], y_mo[idx], ax=ax, legend=(s == 0),
                          title=f"subsample {s+1}: root split X_{t.root.feature_index} <= {t.root.threshold:.2f}")
plt.tight_layout(); plt.show()
dis = np.mean([np.mean(grid_preds[a] != grid_preds[b]) for a, b in [(0, 1), (0, 2), (1, 2)]])
print(f"on average the three trees disagree on {dis:.0%} of a grid covering the data")


#===================================================================================================
# %% [Code cell 40  |  notebook cell 78]  8. Hyperparameters
#===================================================================================================
X_h, y_h = make_moons(200, noise=0.2)
X_h_test, y_h_test = make_moons(2000, noise=0.2)
md_grid, mss_grid = [1, 2, 3, 4, 5, 7, 10], [2, 5, 10, 20, 40, 80]
acc_grid = np.zeros((len(md_grid), len(mss_grid)))
for a, md in enumerate(md_grid):
    for b_, mss in enumerate(mss_grid):
        t = DecisionTreeClassifier(min_samp_split=mss, max_depth=md); t.fit(X_h, y_h.reshape(-1, 1))
        acc_grid[a, b_] = np.mean(np.array(t.predict(X_h_test)) == y_h_test)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(acc_grid, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(len(mss_grid)), mss_grid); ax.set_yticks(range(len(md_grid)), md_grid)
for a in range(len(md_grid)):
    for b_ in range(len(mss_grid)):
        ax.text(b_, a, f"{acc_grid[a, b_]:.2f}", ha="center", va="center", fontsize=8, color="w" if acc_grid[a, b_] < acc_grid.mean() else "k")
ax.set(title="Test accuracy on moons (noise 0.2, n_train = 200)", xlabel="min_samp_split", ylabel="max_depth")
ax.grid(False); fig.colorbar(im, label="test accuracy"); plt.show()
a_best, b_best = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
print(f"best: max_depth={md_grid[a_best]}, min_samp_split={mss_grid[b_best]}, test accuracy {acc_grid[a_best, b_best]:.3f}")


#===================================================================================================
# %% [Code cell 41  |  notebook cell 80]  8. Hyperparameters
#===================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, mss in zip(axes, [2, 20]):
    t = DecisionTreeClassifier(min_samp_split=mss, max_depth=10); t.fit(X_h, y_h.reshape(-1, 1))
    te = np.mean(np.array(t.predict(X_h_test)) == y_h_test)
    plot_decision_regions(t.predict, X_h, y_h, ax=ax, legend=(mss == 2),
                          title=f"max_depth=10, min_samp_split={mss}: {count_leaves(t.root)} leaves, test acc {te:.2f}")
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 42  |  notebook cell 82]  9. Complexity and limitations
#===================================================================================================
ns, secs = [50, 100, 200, 400, 800], []
for n_ in ns:
    Xt_, yt_ = make_xor(n_, noise=0.1)
    t0 = time.perf_counter()
    DecisionTreeClassifier(min_samp_split=2, max_depth=4).fit(Xt_, yt_.reshape(-1, 1))
    secs.append(time.perf_counter() - t0)
slope = np.polyfit(np.log(ns), np.log(secs), 1)[0]

fig, ax = plt.subplots()
ax.loglog(ns, secs, "o-", label=f"measured fit time (log-log slope ≈ {slope:.2f})")
ax.loglog(ns, secs[-1] * (np.array(ns) / ns[-1]) ** 2, "k--", lw=1, label="reference slope 2 ($\\propto n^2$)")
ax.set(title="Fit time vs. training-set size (max_depth=4, m=2)", xlabel="n (training samples)", ylabel="seconds")
ax.legend(); plt.show()
for n_, s_ in zip(ns, secs):
    print(f"n = {n_:>4}: {s_:.3f} s")
print(f"fitted log-log slope over all points: {slope:.2f}; "
      f"local slope over the last doubling (400 -> 800): {np.log2(secs[-1] / secs[-2]):.2f}")
