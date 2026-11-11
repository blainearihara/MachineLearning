# -*- coding: utf-8 -*-
"""
RandomForest_Walkthrough.py -- the code cells of RandomForest_Walkthrough.ipynb, extracted for step-through debugging.

How to use
- Open this file side by side with RandomForest_Walkthrough.ipynb. Each "# %%" block below is one code cell of the
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
assert (NB_DIR / "RandomForest.py").exists(), "Open/run this notebook from inside the RandomForest/ folder"
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
import RandomForest as rf_mod
import DecisionTreeRandomForest as dtrf_mod
from RandomForest import RandomForestClassifier
from DecisionTreeRandomForest import DecisionTreeClassifier

print("RandomForestClassifier methods:", [k for k in vars(RandomForestClassifier) if not k.startswith("__")])
print("DecisionTreeClassifier methods:", [k for k in vars(DecisionTreeClassifier) if not k.startswith("__")])


#===================================================================================================
# %% [Code cell 4  |  notebook cell 7]  2. The problem: a flexible classifier that does not memorise noise
#===================================================================================================
# Five independent training sets from the same distribution -> five deep trees
X_te, y_te = make_moons(1000, noise=0.3)                         # a common test set
g0, g1 = np.meshgrid(np.linspace(-1.5, 2.5, 90), np.linspace(-1.2, 1.7, 90))
grid = np.c_[g0.ravel(), g1.ravel()]
votes = np.zeros(len(grid)); accs = []
for s in range(5):
    Xs, ys = make_moons(200, noise=0.3)
    tree = DecisionTreeClassifier(min_samp_split=2, max_depth=None)  # a single fully grown tree
    tree.fit(Xs, ys.reshape(-1, 1))
    votes += np.array(tree.predict(grid))
    accs.append(np.mean(np.array(tree.predict(X_te)) == y_te))

fig, ax = plt.subplots(figsize=(6.5, 4.6))
cf = ax.contourf(g0, g1, votes.reshape(g0.shape), levels=np.arange(-0.5, 6), cmap="RdBu_r")
fig.colorbar(cf, ticks=range(6), label="number of the 5 trees predicting class 1")
ax.set(title="Five deep trees, five training sets: where do they disagree?", xlabel="$x_0$", ylabel="$x_1$"); plt.show()
print("test accuracy of the five trees:", np.round(accs, 3))
print(f"share of the plane where the five trees do not all agree: {np.mean((votes > 0) & (votes < 5)):.0%}")


#===================================================================================================
# %% [Code cell 5  |  notebook cell 9]  4. Core concepts, theory and math  >  4.1 Bootstrap samples
#===================================================================================================
# Illustration only — Eq. (1): expected fraction of distinct samples in a bootstrap sample
ns = np.array([2, 5, 10, 20, 50, 100, 1000])
coverage = 1 - (1 - 1 / ns) ** ns
for n_, c in zip(ns, coverage):
    print(f"n = {n_:>5}: expected distinct fraction = {c:.4f}")
print(f"limit 1 - 1/e = {1 - np.exp(-1):.4f}")


#===================================================================================================
# %% [Code cell 6  |  notebook cell 11]  4. Core concepts, theory and math  >  4.1 Bootstrap samples
#===================================================================================================
n = 200
rf_tmp = RandomForestClassifier(random_state=1)          # the forest's own random generator, seeded with 1
idx_col = np.arange(n).reshape(-1, 1)
fractions, counts = [], []
for _ in range(2000):
    drawn, _ = rf_tmp.bootstrap_sample(idx_col, idx_col)  # rows of the index column = the drawn indices
    c = np.bincount(drawn.ravel(), minlength=n)
    fractions.append(np.mean(c > 0)); counts.append(c)
counts = np.concatenate(counts)
print(f"mean distinct fraction over 2,000 bootstrap samples (n = {n}): {np.mean(fractions):.4f}")
print("empirical P(N_i = k), k = 0..5:", np.round([np.mean(counts == k) for k in range(6)], 4))


#===================================================================================================
# %% [Code cell 7  |  notebook cell 12]  4. Core concepts, theory and math  >  4.1 Bootstrap samples
#===================================================================================================
# Check — Eqs. (1) and (2)
from math import factorial
exact = 1 - (1 - 1 / n) ** n
assert abs(np.mean(fractions) - exact) < 0.002
poisson = np.array([np.exp(-1) / factorial(k) for k in range(6)])
assert np.allclose([np.mean(counts == k) for k in range(6)], poisson, atol=0.005)
print(f"✓ distinct fraction {np.mean(fractions):.4f} vs exact {exact:.4f};  copy counts match Poisson(1): {np.round(poisson, 4)}")


#===================================================================================================
# %% [Code cell 8  |  notebook cell 14]  4. Core concepts, theory and math  >  4.2 Why averaging reduces variance, and why correlation limits it
#===================================================================================================
# Illustration only — Eq. (3) by simulation
Ts = np.array([1, 2, 5, 10, 25, 50, 100])
fig, ax = plt.subplots()
for rho, col in zip([0.0, 0.2, 0.5], CLASS_COLORS):
    S = rng.normal(size=(20_000, 1))
    E = rng.normal(size=(20_000, Ts.max()))
    Z = np.sqrt(rho) * S + np.sqrt(1 - rho) * E                 # 20,000 replications of T_max correlated "trees"
    measured = [Z[:, :T].mean(axis=1).var() for T in Ts]
    theory = rho + (1 - rho) / Ts
    ax.plot(Ts, measured, "o", color=col, label=f"simulated, ρ = {rho}")
    ax.plot(Ts, theory, "-", color=col, alpha=0.7, label=f"Eq. (3), floor ρσ² = {rho}")
    assert np.allclose(measured, theory, rtol=0.05, atol=0.003)
ax.set_xscale("log"); ax.set(title="Variance of an average of T correlated estimators (σ² = 1)", xlabel="T (number of trees)", ylabel="variance of the average")
ax.legend(ncol=2, fontsize=8); plt.show()
print("✓ simulated variances agree with Eq. (3) for every ρ and T")


#===================================================================================================
# %% [Code cell 9  |  notebook cell 16]  4. Core concepts, theory and math  >  4.3 Majority voting
#===================================================================================================
# Illustration only — accuracy of a majority vote of T voters, each correct with probability p = 0.65
from math import comb
p_c, T_odd = 0.65, np.arange(1, 102, 2)
indep = [sum(comb(T, k) * p_c**k * (1 - p_c)**(T - k) for k in range(T // 2 + 1, T + 1)) for T in T_odd]
# Correlated voters: each round, with probability 0.4 all voters copy one shared draw; otherwise they vote independently
corr = []
for T in T_odd:
    shared = rng.random(20_000) < 0.4
    common = rng.random(20_000) < p_c
    own = rng.random((20_000, T)) < p_c
    correct = np.where(shared[:, None], common[:, None], own)
    corr.append(np.mean(correct.sum(axis=1) > T / 2))
fig, ax = plt.subplots()
ax.plot(T_odd, indep, label="independent voters (binomial formula)")
ax.plot(T_odd, corr, label="correlated voters (simulation)")
ax.axhline(p_c, color="gray", ls="--", label=f"a single voter (p = {p_c})")
ax.set(title="Majority vote accuracy vs. number of voters", xlabel="T (number of voters, odd)", ylabel="P(majority is correct)")
ax.legend(); plt.show()


#===================================================================================================
# %% [Code cell 10  |  notebook cell 18]  4. Core concepts, theory and math  >  4.3 Majority voting
#===================================================================================================
rf_demo = RandomForestClassifier()
for votes_ in [[1., 0., 1., 1., 0.], [2., 0., 2., 1., 1., 2.], [0., 1., 1., 0.], [1., 0., 0., 1.]]:
    print(f"votes {votes_} -> majority_vote = {rf_demo.majority_vote(np.array(votes_))}")


#===================================================================================================
# %% [Code cell 11  |  notebook cell 19]  4. Core concepts, theory and math  >  4.3 Majority voting
#===================================================================================================
# Check — agreement with Eq. (4) on 500 random vote vectors, ties included (ties -> smallest label)
for _ in range(500):
    v = rng.integers(0, 3, size=rng.integers(1, 12)).astype(float)
    assert rf_demo.majority_vote(v) == np.argmax(np.bincount(v.astype(int)))
print("✓ majority_vote returns the plurality class of Eq. (4); ties go to the smallest label")


#===================================================================================================
# %% [Code cell 12  |  notebook cell 21]  4. Core concepts, theory and math  >  4.4 Random feature subsets at every node
#===================================================================================================
for mf in [None, "sqrt", "log2", 3, 0.5]:
    t = DecisionTreeClassifier(max_features=mf)
    sizes = {len(list(t.get_random_features(9))) for _ in range(50)}
    print(f"max_features={str(mf):<5} with m = 9: subset size(s) seen = {sizes}")
print("max_features='log2' with m = 1: subset size =", len(DecisionTreeClassifier(max_features="log2").get_random_features(1)))
try:
    DecisionTreeClassifier(max_features="auto").get_random_features(9)
except ValueError as err:
    print("max_features='auto' ->", type(err).__name__ + ":", err)

for _ in range(200):
    m_ = int(rng.integers(2, 40)); t = DecisionTreeClassifier(max_features="sqrt")
    sub = np.asarray(t.get_random_features(m_))
    assert len(sub) == int(np.sqrt(m_)) and len(set(sub)) == len(sub) and sub.max() < m_
print("✓ 'sqrt' gives floor(sqrt(m)) distinct, valid feature indices for every m tested")


#===================================================================================================
# %% [Code cell 13  |  notebook cell 24]  4. Core concepts, theory and math  >  4.5 The forest's tree, and the bootstrap-duplicate crash that was fixed
#===================================================================================================
X_m, y_m = make_moons(200, noise=0.3)
print("max_depth | actual depth")
for md in [1, 2, 3, 5, None]:
    t = DecisionTreeClassifier(min_samp_split=2, max_depth=md); t.fit(X_m, y_m)
    print(f"{str(md):>9} | {tree_depth(t.root):>12}")
    if md is not None:
        assert tree_depth(t.root) == md


#===================================================================================================
# %% [Code cell 14  |  notebook cell 26]  4. Core concepts, theory and math  >  4.5 The forest's tree, and the bootstrap-duplicate crash that was fixed
#===================================================================================================
from math import exp
def p_at_least(k):                                   # Poisson(1) tail, Eq. (2)
    return 1 - sum(exp(-1) / factorial(i) for i in range(k))

print("k | P(N >= k) | expected # rows drawn >= k times in a 20-tree forest on 200 rows")
for k in [2, 3, 5, 8, 10]:
    print(f"{k:>2}| {p_at_least(k):.2e}  | {20 * 200 * p_at_least(k):10.4f}")

X_c, y_c = make_moons(200, noise=0.3)
dup_rows = []
for seed in range(1, 11):
    f = RandomForestClassifier(n_estimators=20, min_samp_split=2, max_depth=None, random_state=seed)
    f.fit(X_c, y_c)                                      # fully grown trees: crashed before the fix
    dup_rows.append(np.mean([np.sum(np.bincount(idx, minlength=200) >= 2) for idx in f.bootstrap_indices_]))
print(f"\n✓ 10/10 fully grown forests fitted; on average {np.mean(dup_rows):.0f} of 200 rows were duplicated in each bootstrap sample")


#===================================================================================================
# %% [Code cell 15  |  notebook cell 28]  5. Implementation walkthrough, in the order the code runs
#===================================================================================================
X_r, y_r = make_moons(200, noise=0.3); X_rt, y_rt = make_moons(500, noise=0.3)
for seed in [3, 0]:
    forest = RandomForestClassifier(n_estimators=5, max_depth=None, random_state=seed)
    forest.fit(X_r, y_r); first = np.array(forest.predict(X_rt))
    forest.fit(X_r, y_r); second = np.array(forest.predict(X_rt))             # same object, refit
    fresh = RandomForestClassifier(n_estimators=5, max_depth=None, random_state=seed)
    fresh.fit(X_r, y_r); third = np.array(fresh.predict(X_rt))                # new object, same seed
    assert np.array_equal(first, second) and np.array_equal(first, third)
    print(f"random_state={seed}: refit and a fresh object both reproduce the forest exactly")
np.random.seed(123); before = np.random.random()
np.random.seed(123); RandomForestClassifier(n_estimators=3, random_state=1).fit(X_r, y_r); after = np.random.random()
assert before == after
print("✓ fitting a forest leaves NumPy's global random state untouched")


#===================================================================================================
# %% [Code cell 16  |  notebook cell 30]  6. Worked numerical example: a 3-tree forest on 10 samples
#===================================================================================================
X_toy = np.array([[1, 2], [2, 5], [3, 1], [4, 7], [5, 3], [6, 8], [7, 4], [8, 6], [3, 8], [9, 2]], dtype=float)
y_toy = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0], dtype=float)
print(" i  x0  x1  y")
for i, ((a, b_), c) in enumerate(zip(X_toy, y_toy)):
    print(f"{i:>2} {a:>3.0f} {b_:>3.0f} {c:>2.0f}")


#===================================================================================================
# %% [Code cell 17  |  notebook cell 31]  6. Worked numerical example: a 3-tree forest on 10 samples
#===================================================================================================
toy_forest = RandomForestClassifier(n_estimators=3, min_samp_split=2, max_depth=2, max_features="sqrt", random_state=5)
toy_forest.fit(X_toy, y_toy)
for i, (tree, idx) in enumerate(zip(toy_forest.trees, toy_forest.bootstrap_indices_)):
    oob = sorted(set(range(len(X_toy))) - set(idx.tolist()))
    print(f"--- tree {i}: bootstrap indices {sorted(idx.tolist())}")
    print(f"    distinct rows used: {len(set(idx.tolist()))}/10, out-of-bag rows: {oob}")
    tree.print_tree()


#===================================================================================================
# %% [Code cell 18  |  notebook cell 32]  6. Worked numerical example: a 3-tree forest on 10 samples
#===================================================================================================
# Check — each tree is exactly the tree you get by fitting the repo's tree to its recorded bootstrap sample
for tree, idx in zip(toy_forest.trees, toy_forest.bootstrap_indices_):
    replica = DecisionTreeClassifier(min_samp_split=2, max_depth=2, max_features="sqrt", random_state=tree.random_state)
    replica.fit(X_toy[idx], y_toy[idx])
    assert np.array_equal(np.array(replica.predict(X_toy)), np.array(tree.predict(X_toy)))
print("✓ refitting each tree on its recorded bootstrap sample (with its recorded seed) reproduces it exactly")


#===================================================================================================
# %% [Code cell 19  |  notebook cell 34]  6. Worked numerical example: a 3-tree forest on 10 samples
#===================================================================================================
Q = np.array([[2.0, 2.0], [5.0, 6.0], [8.0, 3.0]])
P = np.array([t.predict(Q) for t in toy_forest.trees])          # shape (T, n_queries)
print("query       tree0 tree1 tree2 | majority_vote | forest.predict")
for q, final in zip(range(len(Q)), toy_forest.predict(Q)):
    print(f"{str(Q[q]):<11} {P[0, q]:>5.0f} {P[1, q]:>5.0f} {P[2, q]:>5.0f} | {toy_forest.majority_vote(P[:, q]):>13.0f} | {final:>14.0f}")
print("\ntraining accuracy of the forest:", np.mean(np.array(toy_forest.predict(X_toy)) == y_toy))


#===================================================================================================
# %% [Code cell 20  |  notebook cell 36]  7. Visual experiments  >  7.1 One tree versus a forest
#===================================================================================================
X_tr, y_tr = make_moons(200, noise=0.3)
X_test, y_test = make_moons(1000, noise=0.3)
common = dict(min_samp_split=2, max_depth=None)          # fully grown trees

single = DecisionTreeClassifier(**common); single.fit(X_tr, y_tr.reshape(-1, 1))
forest50 = RandomForestClassifier(n_estimators=51, max_features="sqrt", random_state=7, **common)
forest50.fit(X_tr, y_tr.reshape(-1, 1))
acc = lambda model: np.mean(np.array(model.predict(X_test)) == y_test)

fig, axes = plt.subplots(1, 5, figsize=(24, 4.3))
plot_decision_regions(single.predict, X_tr, y_tr, ax=axes[0], grid_n=100, title=f"single tree (all data): test {acc(single):.3f}")
for k in range(3):
    t = forest50.trees[k]
    plot_decision_regions(t.predict, X_tr, y_tr, ax=axes[k + 1], grid_n=100, legend=False, title=f"forest tree #{k}: test {acc(t):.3f}")
plot_decision_regions(forest50.predict, X_tr, y_tr, ax=axes[4], grid_n=100, legend=False, title=f"forest of 51 trees: test {acc(forest50):.3f}")
plt.tight_layout(); plt.show()
print(f"mean test accuracy of the 51 individual forest trees: {np.mean([acc(t) for t in forest50.trees]):.3f}")


#===================================================================================================
# %% [Code cell 21  |  notebook cell 38]  7. Visual experiments  >  7.2 Test accuracy as trees are added
#===================================================================================================
def staged_vote_accuracy(model, X, y):
    P = np.array([tree.predict(X) for tree in model.trees])            # (T, n)
    ts = np.arange(1, len(model.trees) + 1, 2)
    accs = [np.mean(np.array([model.majority_vote(P[:t, i]) for i in range(P.shape[1])]) == y) for t in ts]
    return ts, np.array(accs)

bagging = RandomForestClassifier(n_estimators=51, max_features=None, random_state=7, **common)
bagging.fit(X_tr, y_tr.reshape(-1, 1))
ts, acc_rf = staged_vote_accuracy(forest50, X_test, y_test)
_, acc_bag = staged_vote_accuracy(bagging, X_test, y_test)

fig, ax = plt.subplots()
ax.plot(ts, acc_rf, "o-", ms=3, label="random forest (max_features='sqrt')")
ax.plot(ts, acc_bag, "s-", ms=3, label="bagging (max_features=None)")
ax.axhline(acc(single), color="gray", ls="--", label="single tree on all data")
ax.set(title="Test accuracy vs. number of trees (moons, n = 200)", xlabel="number of trees voting (odd)", ylabel="test accuracy")
ax.legend(); plt.show()
print(f"forest: 1 tree {acc_rf[0]:.3f} -> 51 trees {acc_rf[-1]:.3f};  bagging: 1 tree {acc_bag[0]:.3f} -> 51 trees {acc_bag[-1]:.3f}")


#===================================================================================================
# %% [Code cell 22  |  notebook cell 40]  7. Visual experiments  >  7.3 Decorrelation: the effect of max_features
#===================================================================================================
def add_noise_features(X, k=4):
    return np.c_[X, rng.normal(size=(len(X), k))]
def redundant_copies(X):
    return np.column_stack([X[:, j] + 0.25 * rng.normal(size=len(X)) for j in (0, 1, 0, 1, 0, 1)])

mf_list, mtry, n_reps = [1, "sqrt", None], [1, 2, 6], 3
table = {name: np.zeros((len(mf_list), 3)) for name in ("noise features", "redundant features")}
for rep in range(n_reps):
    Xb_tr, yb_tr = make_moons(200, noise=0.2); Xb_te, yb_te = make_moons(600, noise=0.2)
    for name, make in (("noise features", add_noise_features), ("redundant features", redundant_copies)):
        A_tr, A_te = make(Xb_tr), make(Xb_te)
        for a, mf in enumerate(mf_list):
            f = RandomForestClassifier(n_estimators=21, max_features=mf, random_state=rep + 1, **common)
            f.fit(A_tr, yb_tr.reshape(-1, 1))
            P = np.array([t.predict(A_te) for t in f.trees])           # (21, n_test)
            C = np.corrcoef(P)
            table[name][a] += np.array([np.mean(P == yb_te, axis=1).mean(), C[np.triu_indices_from(C, k=1)].mean(),
                                        np.mean(np.array(f.predict(A_te)) == yb_te)]) / n_reps
X6_tr = add_noise_features(X_tr)                                     # used for feature importance in §7.4

for name, vals in table.items():
    print(f"{name} (means over {n_reps} replications):\n  max_features (m_try) | mean single-tree acc | mean pairwise ρ | forest acc")
    for mf, k, (a1, rho_, af) in zip(mf_list, mtry, vals):
        print(f"  {str(mf):>12} ({k})     | {a1:>20.3f} | {rho_:>15.3f} | {af:>10.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, (name, vals) in zip(axes, table.items()):
    ax.plot(mtry, vals[:, 0], "o-", label="mean single-tree accuracy")
    ax.plot(mtry, vals[:, 2], "s-", label="forest accuracy")
    ax.set(title=name, xlabel="$m_{try}$ (features searched per node, of m = 6)", ylabel="test accuracy", xticks=mtry)
    ax2 = ax.twinx(); ax2.plot(mtry, vals[:, 1], "^--", color=CLASS_COLORS[1], label="mean pairwise ρ")
    ax2.set_ylabel("correlation between trees", color=CLASS_COLORS[1]); ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()


#===================================================================================================
# %% [Code cell 23  |  notebook cell 42]  7. Visual experiments  >  7.4 Feature importance
#===================================================================================================
f_imp = RandomForestClassifier(n_estimators=31, max_features="sqrt", random_state=11, **common)
f_imp.fit(X6_tr, y_tr.reshape(-1, 1))
names = ["x0 (moons)", "x1 (moons)", "x2 (noise)", "x3 (noise)", "x4 (noise)", "x5 (noise)"]
imp = f_imp.get_feature_importance(names)
assert np.allclose(imp, f_imp.get_feature_importance())       # feature_names is optional

# Check — recompute the definition above directly from the trees' nodes
manual_imp = np.zeros(6)
for t in f_imp.trees:
    for node in iter_nodes(t.root):
        if node.value is None and node.info_gain:
            manual_imp[node.feature_index] += node.info_gain
assert np.allclose(imp, manual_imp / len(f_imp.trees))

fig, ax = plt.subplots()
ax.bar(names, imp, color=[CLASS_COLORS[0]] * 2 + ["gray"] * 4)
ax.set(title="Repo feature importance (sum of unweighted Gini gains per tree)", ylabel="importance"); plt.xticks(rotation=20); plt.show()
print("importances:", np.round(imp, 3), " sum =", round(imp.sum(), 3))
print("✓ get_feature_importance equals the per-node sum of info_gain / n_estimators")


#===================================================================================================
# %% [Code cell 24  |  notebook cell 44]  7. Visual experiments  >  7.5 Variance reduction, measured
#===================================================================================================
grid_small = np.c_[np.repeat(np.linspace(-1.5, 2.5, 50), 50), np.tile(np.linspace(-1.2, 1.7, 50), 50)]
H_tree, V_forest, acc_tree, acc_forest = [], [], [], []
for s in range(10):
    Xs, ys = make_moons(200, noise=0.3)
    t = DecisionTreeClassifier(**common); t.fit(Xs, ys.reshape(-1, 1))
    f = RandomForestClassifier(n_estimators=25, max_features="sqrt", random_state=100 + s, **common); f.fit(Xs, ys.reshape(-1, 1))
    H_tree.append(np.array(t.predict(grid_small)))
    V_forest.append(np.mean([tr.predict(grid_small) for tr in f.trees], axis=0))      # vote share per grid point
    acc_tree.append(np.mean(np.array(t.predict(X_test)) == y_test)); acc_forest.append(np.mean(np.array(f.predict(X_test)) == y_test))
var_tree, var_forest = np.var(H_tree, axis=0), np.var(V_forest, axis=0)
print(f"average variance across training sets: single-tree output {var_tree.mean():.4f}   forest vote share {var_forest.mean():.4f}"
      f"   (ratio {var_forest.mean() / var_tree.mean():.2f})")
print(f"mean test accuracy over the 10 training sets: single tree {np.mean(acc_tree):.3f}, forest {np.mean(acc_forest):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, V_, ttl in zip(axes, [var_tree, var_forest], ["single deep tree: Var of output", "forest of 25: Var of vote share"]):
    im = ax.imshow(V_.reshape(50, 50).T, origin="lower", extent=[-1.5, 2.5, -1.2, 1.7], aspect="auto", cmap="magma", vmin=0, vmax=0.25)
    ax.set(title=ttl, xlabel="$x_0$"); ax.grid(False)
axes[0].set_ylabel("$x_1$"); fig.colorbar(im, ax=axes, label="variance across 10 training sets"); plt.show()


#===================================================================================================
# %% [Code cell 25  |  notebook cell 46]  8. Hyperparameters
#===================================================================================================
same = RandomForestClassifier(n_estimators=4, min_samp_split=2, max_depth=8, max_features=None, bootstrap=False, random_state=1)
same.fit(X_tr, y_tr.reshape(-1, 1))
P_same = np.array([t.predict(X_test) for t in same.trees])
print("bootstrap=False and max_features=None -> all trees identical:", all(np.array_equal(P_same[0], p) for p in P_same))

depth_rows = []
for md in [1, 2, 3, 5, 8, 12, 20]:
    f = RandomForestClassifier(n_estimators=25, min_samp_split=2, max_depth=md, max_features="sqrt", random_state=5)
    f.fit(X_tr, y_tr.reshape(-1, 1))
    t = DecisionTreeClassifier(min_samp_split=2, max_depth=md); t.fit(X_tr, y_tr.reshape(-1, 1))
    depth_rows.append((md, acc(t), acc(f)))
fig, ax = plt.subplots()
ax.plot([r_[0] for r_ in depth_rows], [r_[1] for r_ in depth_rows], "o-", label="single tree")
ax.plot([r_[0] for r_ in depth_rows], [r_[2] for r_ in depth_rows], "s-", label="forest of 25")
ax.set(title="Test accuracy vs. max_depth (moons)", xlabel="max_depth", ylabel="test accuracy"); ax.legend(); plt.show()
for r_ in depth_rows:
    print(f"max_depth={r_[0]:>2}: single tree {r_[1]:.3f}   forest {r_[2]:.3f}")
