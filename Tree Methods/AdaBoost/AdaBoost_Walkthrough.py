# -*- coding: utf-8 -*-
"""
AdaBoost_Walkthrough.py -- the code cells of AdaBoost_Walkthrough.ipynb, extracted for step-through debugging.

How to use
- Open this file side by side with AdaBoost_Walkthrough.ipynb. Each "# %%" block below is one code cell of the
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
assert (NB_DIR / "AdaBoost.py").exists(), "Open/run this notebook from inside the AdaBoost/ folder"
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
from AdaBoost import AdaBoostClassifier
from DecisionTreeAdaBoost import DecisionTreeClassifier as WeightedTree

print("AdaBoostClassifier methods:", [k for k in vars(AdaBoostClassifier) if not k.startswith("__")])
print("weighted tree methods:     ", [k for k in vars(WeightedTree) if not k.startswith("__")])


#===================================================================================================
# %% [Code cell 4  |  notebook cell 7]  2. The problem: turning weak rules into a strong classifier
#===================================================================================================
X_tr, y_tr = make_moons(200, noise=0.25)
X_te, y_te = make_moons(1000, noise=0.25)
stump = WeightedTree(max_depth=1); stump.fit(X_tr, y_tr)                       # uniform weights by default
acc = lambda pred, y: np.mean(np.asarray(pred) == y)
fig, ax = plt.subplots(figsize=(6, 4.4))
plot_decision_regions(stump.predict, X_tr, y_tr, ax=ax, title=f"One decision stump: test accuracy {acc(stump.predict(X_te), y_te):.3f}")
plt.show()
stump.print_tree()


#===================================================================================================
# %% [Code cell 5  |  notebook cell 9]  4. Core concepts, theory and math  >  4.1 Weak learners that respect sample weights
#===================================================================================================
x1d = np.arange(1, 11, dtype=float).reshape(-1, 1)
y1d = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1], dtype=float)          # point at x=7 is a class-0 point among class-1
w_uniform = np.full(10, 0.1)
w_heavy = w_uniform.copy(); w_heavy[6] = 0.6; w_heavy /= w_heavy.sum()   # emphasise sample 6 (x=7)
for name, w in [("uniform weights", w_uniform), ("sample 6 (x=7) up-weighted", w_heavy)]:
    t = WeightedTree(max_depth=1); t.fit(x1d, y1d, w)
    print(f"{name:<28}: stump x_0 <= {t.root.threshold}, leaves {t.root.left.value}/{t.root.right.value}, "
          f"weighted error {np.sum(w * (np.array(t.predict(x1d)) != y1d)) / w.sum():.3f}")


#===================================================================================================
# %% [Code cell 6  |  notebook cell 11]  4. Core concepts, theory and math  >  4.1 Weak learners that respect sample weights
#===================================================================================================
wt = WeightedTree()
def gini_w(y, w):                         # Illustration only — weighted Gini formula
    p = np.array([w[y == k].sum() for k in np.unique(y)]) / w.sum()
    return 1 - np.sum(p**2)
for _ in range(300):
    n_ = int(rng.integers(2, 25)); y_ = rng.integers(0, 3, n_).astype(float); w_ = rng.random(n_); s = int(rng.integers(1, n_))
    assert np.isclose(wt.gini_index(y_, w_), gini_w(y_, w_))
    gain = gini_w(y_, w_) - (w_[:s].sum() * gini_w(y_[:s], w_[:s]) + w_[s:].sum() * gini_w(y_[s:], w_[s:])) / w_.sum()
    assert np.isclose(wt.information_gain(y_, y_[:s], y_[s:], w_, w_[:s], w_[s:], "gini"), gain)
    best = max(np.unique(y_), key=lambda k: w_[y_ == k].sum())
    assert wt.calculate_leaf_value(y_, w_) == best
print("✓ weighted gini_index, information_gain and calculate_leaf_value match the weighted formulas (300 random cases)")


#===================================================================================================
# %% [Code cell 7  |  notebook cell 13]  4. Core concepts, theory and math  >  4.2 AdaBoost as greedy minimisation of the exponential loss (binary case)
#===================================================================================================
ada_bin = AdaBoostClassifier(n_estimators=30, max_depth=1, learning_rate=1.0)
ada_bin.fit(X_tr, y_tr)
eps = np.array(ada_bin.estimator_errors_)            # weighted training error of each kept stump
alphas = np.array(ada_bin.get_estimator_weights())
print("first five weighted errors ε_t:", np.round(eps[:5], 4))
print("first five alphas           :", np.round(alphas[:5], 4))
print("Eq.(3) ½·ln((1-ε)/ε)         :", np.round(0.5 * np.log((1 - eps[:5]) / eps[:5]), 4))


#===================================================================================================
# %% [Code cell 8  |  notebook cell 14]  4. Core concepts, theory and math  >  4.2 AdaBoost as greedy minimisation of the exponential loss (binary case)
#===================================================================================================
# Check — Eq. (3) for every round, and the training-error bound of Eq. (5) at every stage
assert np.allclose(alphas, 0.5 * np.log((1 - eps) / eps))
bound = np.cumprod(2 * np.sqrt(eps * (1 - eps)))
staged = ada_bin.get_staged_predictions(X_tr)
train_err = np.array([np.mean(np.array(p) != y_tr) for p in staged])
assert np.all(train_err <= bound + 1e-12)
print(f"✓ alpha_t = ½ ln((1-ε_t)/ε_t) for all {len(alphas)} rounds; training error ≤ Π 2√(ε(1-ε)) at every stage")

fig, ax = plt.subplots()
ax.plot(range(1, len(bound) + 1), train_err, "o-", ms=3, label="training error of the repo's ensemble")
ax.plot(range(1, len(bound) + 1), bound, "--", label="bound Π 2√(ε_t(1−ε_t)), Eq. (5)")
ax.set(title="AdaBoost on moons (stumps, η = 1): training error and its bound", xlabel="round t", ylabel="training error"); ax.legend()
plt.show()


#===================================================================================================
# %% [Code cell 9  |  notebook cell 16]  4. Core concepts, theory and math  >  4.3 Multiclass: SAMME, and why the repo's update is exactly SAMME
#===================================================================================================
# Illustration only — textbook SAMME (Eq. 6), using the repo's WeightedTree as the weak learner
def samme(X, y, T, eta):
    K = len(np.unique(y)); w = np.full(len(y), 1 / len(y)); learners, alphas_s = [], []
    for _ in range(T):
        h = WeightedTree(max_depth=1); h.fit(X, y, w)
        miss = (np.array(h.predict(X)) != y).astype(float)
        e = np.clip(np.sum(w * miss) / w.sum(), 1e-10, 1 - 1e-10)
        if e >= 1 - 1 / K:
            break
        a = eta * (np.log((1 - e) / e) + np.log(K - 1))
        w = w * np.exp(a * miss); w = w / w.sum()                     # SAMME: only misclassified points change
        learners.append(h); alphas_s.append(a)
    return learners, np.array(alphas_s)

X3, y3 = make_blobs(240, centers=[[0, 0], [2.2, 0], [1.1, 1.9]], std=0.9)
X3_te, y3_te = make_blobs(1000, centers=[[0, 0], [2.2, 0], [1.1, 1.9]], std=0.9)
learners_s, alphas_s = samme(X3, y3, T=40, eta=0.5)
ada3 = AdaBoostClassifier(n_estimators=40, max_depth=1, learning_rate=0.5); ada3.fit(X3, y3)
print("SAMME alphas (first 5):      ", np.round(alphas_s[:5], 4))
print("repo alphas × 2 (first 5):   ", np.round(2 * np.array(ada3.alphas[:5]), 4))


#===================================================================================================
# %% [Code cell 10  |  notebook cell 17]  4. Core concepts, theory and math  >  4.3 Multiclass: SAMME, and why the repo's update is exactly SAMME
#===================================================================================================
# Check — same learners, alphas differing by exactly a factor 2, identical predictions
assert len(learners_s) == len(ada3.trees)
for h_s, h_r in zip(learners_s, ada3.trees):
    assert (h_s.root.feature_index, h_s.root.threshold) == (h_r.root.feature_index, h_r.root.threshold)
assert np.allclose(alphas_s, 2 * np.array(ada3.alphas))
scores = np.zeros((len(X3_te), 3))
for h, a in zip(learners_s, alphas_s):
    p = np.array(h.predict(X3_te)); scores[np.arange(len(p)), p.astype(int)] += a
assert np.array_equal(np.argmax(scores, axis=1).astype(float), np.array(ada3.predict(X3_te)))
print(f"✓ all {len(learners_s)} stumps identical, alpha_SAMME = 2·alpha_repo, identical predictions on {len(X3_te)} test points")
print(f"test accuracy {acc(ada3.predict(X3_te), y3_te):.3f}")


#===================================================================================================
# %% [Code cell 11  |  notebook cell 19]  4. Core concepts, theory and math  >  4.4 Stopping rules
#===================================================================================================
# (1)+(2): a data set where no stump beats chance -> no learners kept -> majority-class fallback
X_bad = np.array([[0.], [0.], [1.], [1.]]); y_bad = np.array([0., 1., 0., 1.])
a_bad = AdaBoostClassifier(n_estimators=10); a_bad.fit(X_bad, y_bad)
print(f"balanced, inseparable data: learners kept = {len(a_bad.trees)}, predict -> {a_bad.predict(X_bad)}")
y_bad2 = np.array([0., 1., 1., 1.]); X_bad2 = np.array([[0.], [0.], [0.], [0.]])
a_bad2 = AdaBoostClassifier(n_estimators=10); a_bad2.fit(X_bad2, y_bad2)
print(f"identical inputs, labels 0,1,1,1: learners kept = {len(a_bad2.trees)}, predict -> {a_bad2.predict(X_bad2[:1])} (majority class)")
# (3): a perfectly separable problem -> one perfect stump, then stop
a_sep = AdaBoostClassifier(n_estimators=50); a_sep.fit(x1d, (x1d.ravel() > 5).astype(float))
print(f"separable data: learners kept = {len(a_sep.trees)} (stopped after the perfect stump), alpha = {a_sep.alphas[0]:.2f}")
# (4): one class
a_one = AdaBoostClassifier(n_estimators=5); a_one.fit(x1d, np.ones(10))
print(f"one class: learners kept = {len(a_one.trees)}, predict -> {a_one.predict(x1d[:2])}")
assert len(a_sep.trees) == 1 and a_bad2.predict(X_bad2[:1]) == [1.0]


#===================================================================================================
# %% [Code cell 12  |  notebook cell 21]  6. Worked numerical example: three rounds on ten points
#===================================================================================================
x_toy = np.arange(1, 11, dtype=float).reshape(-1, 1)
y_toy = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0], dtype=float)
toy = AdaBoostClassifier(n_estimators=3, max_depth=1, learning_rate=1.0)
toy.fit(x_toy, y_toy)

# Replay the round-by-round weights with Eq. (4) (illustration), and check them against the repo's stored ε and α
w = np.full(10, 0.1); scores = np.zeros((10, 2))
for t, (tree, a, e) in enumerate(zip(toy.trees, toy.alphas, toy.estimator_errors_)):
    pred = np.array(tree.predict(x_toy)); miss = pred != y_toy
    print(f"--- round {t + 1}: weights before = {np.round(w, 4)}")
    print(f"    stump: x_0 <= {tree.root.threshold}  -> left {tree.root.left.value:.0f}, right {tree.root.right.value:.0f};  "
          f"misclassified points: x = {x_toy[miss].ravel().astype(int).tolist()}")
    eps_t = w[miss].sum()
    print(f"    ε = {eps_t:.4f} (repo: {e:.4f});  α = ½ ln((1-ε)/ε) = {0.5 * np.log((1 - eps_t) / eps_t):.4f} (repo: {a:.4f})")
    assert np.isclose(eps_t, e) and np.isclose(a, 0.5 * np.log((1 - eps_t) / eps_t))
    w = w * np.where(miss, np.exp(a), np.exp(-a)); w = w / w.sum()
    scores[np.arange(10), pred.astype(int)] += a
    print(f"    ensemble scores (class0, class1) per point: {np.round(scores, 3).tolist()}")
    print(f"    ensemble prediction: {np.argmax(scores, axis=1).tolist()}  training accuracy {np.mean(np.argmax(scores, axis=1) == y_toy):.1f}")
print("\nrepo predict():", np.array(toy.predict(x_toy)).astype(int).tolist(), " true labels:", y_toy.astype(int).tolist())
assert np.array_equal(np.array(toy.predict(x_toy)), y_toy)


#===================================================================================================
# %% [Code cell 13  |  notebook cell 23]  6. Worked numerical example: three rounds on ten points
#===================================================================================================
fig, axes = plt.subplots(1, 4, figsize=(17, 3.4), sharey=True)
w = np.full(10, 0.1)
for t in range(3):
    tree, a = toy.trees[t], toy.alphas[t]
    pred = np.array(tree.predict(x_toy)); miss = pred != y_toy
    ax = axes[t]
    ax.scatter(x_toy.ravel(), y_toy, s=2500 * w, c=[CLASS_COLORS[int(k)] for k in y_toy], edgecolor=np.where(miss, "k", "none"), linewidth=2)
    ax.axvline(tree.root.threshold + 0.5, color="gray", ls="--")
    ax.set(title=f"round {t + 1}: stump at x ≤ {tree.root.threshold:.0f}, α = {a:.3f}", xlabel="$x_0$", yticks=[0, 1], ylim=(-0.6, 1.6))
    w = w * np.where(miss, np.exp(a), np.exp(-a)); w = w / w.sum()
axes[0].set_ylabel("class")
xx = np.linspace(0.5, 10.5, 400).reshape(-1, 1)
axes[3].step(xx.ravel(), toy.predict(xx), where="mid", color="k")
axes[3].scatter(x_toy.ravel(), y_toy, c=[CLASS_COLORS[int(k)] for k in y_toy], s=60)
axes[3].set(title="final weighted vote of the 3 stumps", xlabel="$x_0$")
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 14  |  notebook cell 25]  7. Visual experiments  >  7.1 The decision boundary as stumps are added
#===================================================================================================
ada = AdaBoostClassifier(n_estimators=200, max_depth=1, learning_rate=0.5)
ada.fit(X_tr, y_tr)
print(f"stumps kept: {len(ada.trees)}")

class FirstT:                                            # Illustration only: wrap the first T stumps of a fitted model
    def __init__(self, model, T): self.m, self.T = model, T
    def predict(self, X):
        scores = np.zeros((len(X), len(self.m.classes_)))
        for tree, a in zip(self.m.trees[:self.T], self.m.alphas[:self.T]):
            p = np.array(tree.predict(X))
            for j, c in enumerate(self.m.classes_):
                scores[:, j] += a * (p == c)
        return self.m.classes_[np.argmax(scores, axis=1)]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
for ax, T in zip(axes, [1, 5, 25, 200]):
    sub = FirstT(ada, T)
    plot_decision_regions(sub.predict, X_tr, y_tr, ax=ax, grid_n=100, legend=(T == 1),
                          title=f"{T} stump(s): train {acc(sub.predict(X_tr), y_tr):.3f} / test {acc(sub.predict(X_te), y_te):.3f}")
assert np.array_equal(FirstT(ada, len(ada.trees)).predict(X_te), np.array(ada.predict(X_te)))   # wrapper == repo at T_max
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 15  |  notebook cell 27]  7. Visual experiments  >  7.2 Learning curves: training and test error by round
#===================================================================================================
def staged_errors(model, X, y):
    return np.array([np.mean(np.array(p) != y) for p in model.get_staged_predictions(X)])

tr_err, te_err = staged_errors(ada, X_tr, y_tr), staged_errors(ada, X_te, y_te)
fig, ax = plt.subplots()
ax.plot(range(1, len(tr_err) + 1), tr_err, label="training error")
ax.plot(range(1, len(te_err) + 1), te_err, label="test error")
ax.set(title="AdaBoost (stumps, η = 0.5) on moons", xlabel="number of stumps", ylabel="error rate"); ax.legend(); plt.show()
print(f"after 1 stump: train {tr_err[0]:.3f} / test {te_err[0]:.3f};  best test {te_err.min():.3f} at round {te_err.argmin() + 1};"
      f"  after {len(te_err)}: train {tr_err[-1]:.3f} / test {te_err[-1]:.3f}")


#===================================================================================================
# %% [Code cell 16  |  notebook cell 29]  7. Visual experiments  >  7.3 Learning rate and weak-learner depth
#===================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
results_lr = {}
for eta in [1.0, 0.5, 0.1]:
    m_ = AdaBoostClassifier(n_estimators=200, max_depth=1, learning_rate=eta); m_.fit(X_tr, y_tr)
    e_ = staged_errors(m_, X_te, y_te); results_lr[eta] = e_
    axes[0].plot(range(1, len(e_) + 1), e_, label=f"η = {eta}")
axes[0].set(title="Test error: learning rate (stumps)", xlabel="rounds", ylabel="test error"); axes[0].legend()
results_depth = {}
for depth in [1, 2, 4]:
    m_ = AdaBoostClassifier(n_estimators=100, max_depth=depth, learning_rate=0.5); m_.fit(X_tr, y_tr)
    e_ = staged_errors(m_, X_te, y_te); results_depth[depth] = (e_, len(m_.trees), staged_errors(m_, X_tr, y_tr)[-1])
    axes[1].plot(range(1, len(e_) + 1), e_, label=f"max_depth = {depth} ({len(m_.trees)} learners kept)")
axes[1].set(title="Test error: weak-learner depth (η = 0.5)", xlabel="rounds"); axes[1].legend()
plt.tight_layout(); plt.show()
for eta, e_ in results_lr.items():
    print(f"η = {eta}: test error after 10 rounds {e_[min(9, len(e_)-1)]:.3f}, after {len(e_)} rounds {e_[-1]:.3f}, best {e_.min():.3f}")
for d, (e_, kept, tr_last) in results_depth.items():
    print(f"max_depth = {d}: learners kept {kept}, final train error {tr_last:.3f}, final test error {e_[-1]:.3f}, best {e_.min():.3f}")


#===================================================================================================
# %% [Code cell 17  |  notebook cell 31]  7. Visual experiments  >  7.4 Sensitivity to label noise
#===================================================================================================
y_noisy = y_tr.copy()
flipped = rng.choice(len(y_tr), size=20, replace=False); y_noisy[flipped] = 1 - y_noisy[flipped]
ada_n = AdaBoostClassifier(n_estimators=200, max_depth=1, learning_rate=0.5); ada_n.fit(X_tr, y_noisy)

# Illustration only — the weights before each round, replayed with Eq. (4) from the repo's stored stumps and alphas
w = np.full(len(y_noisy), 1 / len(y_noisy)); share = []
for tree, a in zip(ada_n.trees, ada_n.alphas):
    share.append(w[flipped].sum())
    miss = np.array(tree.predict(X_tr)) != y_noisy
    w = w * np.where(miss, np.exp(a), np.exp(-a)); w /= w.sum()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].plot(range(1, len(share) + 1), share, label="weight on the 20 flipped points")
axes[0].axhline(20 / 200, color="gray", ls="--", label="their share at the start (10%)")
axes[0].set(title="Where the weight goes when labels are noisy", xlabel="round", ylabel="share of total weight"); axes[0].legend()
e_clean, e_noisy = staged_errors(ada, X_te, y_te), staged_errors(ada_n, X_te, y_te)
axes[1].plot(e_clean, label="clean training labels"); axes[1].plot(e_noisy, label="10% of training labels flipped")
axes[1].set(title="Test error (clean test labels)", xlabel="round", ylabel="test error"); axes[1].legend()
plt.tight_layout(); plt.show()
print(f"weight share on flipped points: round 1 {share[0]:.2f}, round 50 {share[49]:.2f}, round {len(share)} {share[-1]:.2f}")
print(f"final test error: clean {e_clean[-1]:.3f}, noisy {e_noisy[-1]:.3f}")
