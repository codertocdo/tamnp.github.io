"""figures.py — regenerates every figure in the post into assets/img/overfitting/.

Run after ./poly_overfit (needs data.csv). matplotlib 3.11.2.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import lasso_path, Ridge
from sklearn.preprocessing import StandardScaler

from experiments import (true_f, poly_features, load_cpp_data, bias_variance,
                         learning_curves, early_stopping, leak_feature_selection,
                         leak_target_encoding, leak_groups, leak_temporal,
                         winners_curse, NOISE_SD, SEED)

OUT = "assets/img/overfitting"
os.makedirs(OUT, exist_ok=True)

# ---- palette (validated reference categorical slots 1-3) + recessive ink ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150,
})

DATA = load_cpp_data()
XTR, YTR = DATA["train"]
XVA, YVA = DATA["val"]
XTE, YTE = DATA["test"]
GRIDX = np.linspace(-1, 1, 600)


def fit(x, y, d, lam=0.0):
    """Same solver as the C++ program: least squares on [A; sqrt(lam) I'] (intercept free)."""
    A = poly_features(x, d)
    if lam > 0:
        P = np.zeros((d, d + 1)); P[:, 1:] = np.sqrt(lam) * np.eye(d)
        A = np.vstack([A, P]); y = np.concatenate([y, np.zeros(d)])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def mse(w, x, y):
    return np.mean((poly_features(x, len(w) - 1) @ w - y) ** 2)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# 1. ------------------------------------------------------------------ poly fits
def fig_poly_fits():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True, sharey=True)
    labels = {1: "Underfitting", 3: "Good fit", 9: "Overfitting", 15: "Severe overfitting"}
    rng = np.random.default_rng(0)
    te_idx = rng.choice(len(XTE), 80, replace=False)
    for ax, d in zip(axes.ravel(), (1, 3, 9, 15)):
        w = fit(XTR, YTR, d)
        ax.scatter(XTE[te_idx], YTE[te_idx], s=12, color=MUTED, alpha=0.5, label="Test samples", zorder=1)
        ax.plot(GRIDX, true_f(GRIDX), "--", color=INK2, lw=1.5, label="True model", zorder=2)
        ax.plot(GRIDX, poly_features(GRIDX, d) @ w, color=ORANGE, label="Predicted model", zorder=3)
        ax.scatter(XTR, YTR, s=34, color=BLUE, edgecolor=SURF, linewidth=1.2, label="Training samples", zorder=4)
        ax.set_title(f"Degree {d}: {labels[d]}\ntrain MSE = {mse(w, XTR, YTR):.4f}   "
                     f"test MSE = {mse(w, XTE, YTE):.4g}", loc="left")
        ax.set_ylim(-2.2, 2.0)
    for ax in axes[1]: ax.set_xlabel("x")
    for ax in axes[:, 0]: ax.set_ylabel("y")
    axes[0, 0].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "of-poly-fits.png")


# 2. ------------------------------------------------------------- error vs degree
def fig_error_vs_degree():
    ds = np.arange(0, 16)
    tr, va, te = [], [], []
    for d in ds:
        w = fit(XTR, YTR, d)
        tr.append(mse(w, XTR, YTR)); va.append(mse(w, XVA, YVA)); te.append(mse(w, XTE, YTE))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(-0.5, 2.5, color=BLUE, alpha=0.06, lw=0)
    ax.axvspan(6.5, 15.5, color=ORANGE, alpha=0.06, lw=0)
    ax.axhline(NOISE_SD**2, color=INK2, lw=1, ls=":")
    ax.text(15.3, NOISE_SD**2 * 0.82, "noise floor σ² = 0.0625", ha="right", va="top", color=INK2, fontsize=8.5)
    ax.plot(ds, tr, "-o", color=BLUE, ms=5, label="Training error")
    ax.plot(ds, va, "-s", color=ORANGE, ms=5, label="Validation error (15 pts)")
    ax.plot(ds, te, "-^", color=AQUA, ms=5, label="Test error (1000 pts)")
    best = int(np.argmin(va))
    ax.annotate(f"min validation error\n→ choose degree {best}", xy=(best, va[best]),
                xytext=(best + 1.6, 0.5), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.text(1, 30, "underfitting\n(high bias)", ha="center", color=BLUE, fontsize=9.5)
    ax.text(11, 30, "overfitting\n(high variance)", ha="center", color=ORANGE, fontsize=9.5)
    ax.set_yscale("log"); ax.set_xlim(-0.5, 15.5); ax.set_xticks(ds)
    ax.set_xlabel("Polynomial degree d (model complexity)"); ax.set_ylabel("MSE (log scale)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.18, 1.0))
    fig.tight_layout()
    save(fig, "of-error-vs-degree.png")


# 3. ----------------------------------------------------------- bias / variance
def fig_bias_variance():
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(10, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    for i, d in enumerate((1, 3, 9)):
        ax = fig.add_subplot(gs[0, i])
        preds = []
        for _ in range(40):
            x = rng.uniform(-1, 1, 30)
            y = true_f(x) + NOISE_SD * rng.standard_normal(30)
            p = poly_features(GRIDX, d) @ fit(x, y, d)
            preds.append(p)
            ax.plot(GRIDX, p, color=ORANGE, lw=0.8, alpha=0.35)
        ax.plot(GRIDX, np.mean(preds, 0), color=BLUE, lw=2.2, label="Average model")
        ax.plot(GRIDX, true_f(GRIDX), "--", color=INK, lw=1.5, label="True model")
        ax.set_ylim(-2.2, 2.0); ax.set_title(f"Degree {d}: 40 models, 40 training sets", loc="left", fontsize=10)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8)
    bv = bias_variance(degrees=range(0, 11))
    ax = fig.add_subplot(gs[1, :])
    ax.plot(bv.degree, bv.bias2 + 1e-5, "-o", color=BLUE, ms=5, label="Bias²")
    ax.plot(bv.degree, bv.variance, "-s", color=ORANGE, ms=5, label="Variance")
    ax.plot(bv.degree, bv.expected_mse, "-^", color=AQUA, ms=5, label="Expected test MSE = Bias² + Var + σ²")
    ax.axhline(NOISE_SD**2, color=INK2, lw=1, ls=":")
    ax.text(9.6, NOISE_SD**2 * 0.45, "σ² (irreducible)", ha="right", color=INK2, fontsize=8.5)
    ax.set_yscale("log"); ax.set_xticks(range(0, 11))
    ax.set_xlabel("Polynomial degree"); ax.set_ylabel("Error (log scale)")
    ax.set_title("Monte Carlo decomposition over 500 training sets of n = 30 (bias² at degree 3–5 is ≈ 0; plotted +1e-5)",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "of-bias-variance.png")


# 4. ------------------------------------------------------------- learning curves
def fig_learning_curves():
    lc = learning_curves()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, d in zip(axes, (3, 12)):
        sub = lc[lc.degree == d]
        ax.plot(sub.n_train, sub.train_mse, "-o", color=BLUE, ms=5, label="Training error")
        ax.plot(sub.n_train, sub.test_mse, "-^", color=AQUA, ms=5, label="Test error")
        ax.axhline(NOISE_SD**2, color=INK2, lw=1, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([15, 20, 30, 50, 100, 200, 500], ["15", "20", "30", "50", "100", "200", "500"])
        ax.minorticks_off()
        ax.set_xlabel("Training-set size n (log scale)")
        ax.set_title(f"Degree {d}" + (" (right capacity)" if d == 3 else " (too much capacity)"), loc="left")
    axes[0].set_ylabel("Median MSE over 200 runs (log)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "of-learning-curves.png")


# 5. ------------------------------------------------------------- early stopping
def fig_early_stopping():
    es = early_stopping(DATA)
    best = es.loc[es.val_mse.idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(es.iteration, es.train_mse, color=BLUE, label="Training error")
    ax.plot(es.iteration, es.val_mse, color=ORANGE, label="Validation error")
    ax.plot(es.iteration, es.test_mse, color=AQUA, lw=1.5, label="Test error")
    ax.axvline(best.iteration, color=INK2, lw=1, ls="--")
    ax.text(best.iteration * 1.25, 0.5, f"stop here\n(iteration {int(best.iteration):,})", color=INK, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.01, 1.5)
    ax.set_xlabel("Gradient-descent iteration (log scale)"); ax.set_ylabel("MSE (log scale)")
    ax.set_title("Degree-15 model trained by gradient descent", loc="left")
    ax.legend(loc="lower left")
    # right panel: the model at three moments of training
    ax = axes[1]
    d = 15
    X = poly_features(XTR, d); mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0); X[:, 1:] = (X[:, 1:] - mu) / sd
    G = poly_features(GRIDX, d); G[:, 1:] = (G[:, 1:] - mu) / sd
    lr = 1.0 / np.linalg.eigvalsh(2 / len(YTR) * X.T @ X).max()
    w, t = np.zeros(d + 1), 0
    snaps = {100: AQUA, int(best.iteration): BLUE, 10_000_000: ORANGE}
    ax.plot(GRIDX, true_f(GRIDX), "--", color=INK2, lw=1.5, label="True model")
    for stop, c in snaps.items():
        while t < stop:
            w -= lr * (2 / len(YTR)) * X.T @ (X @ w - YTR); t += 1
        ax.plot(GRIDX, G @ w, color=c, label=f"after {stop:,} iterations")
    ax.scatter(XTR, YTR, s=22, color=INK2, zorder=5)
    ax.set_ylim(-2.2, 2.0); ax.set_xlabel("x")
    ax.set_title("The fitted curve as training goes on", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "of-early-stopping.png")


# 6. ------------------------------------------------------------------- ridge
def fig_ridge():
    d = 15
    lams = np.logspace(-9, 2, 60)
    tr, va, te = [], [], []
    for lam in lams:
        w = fit(XTR, YTR, d, lam)
        tr.append(mse(w, XTR, YTR)); va.append(mse(w, XVA, YVA)); te.append(mse(w, XTE, YTE))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    ax.plot(GRIDX, true_f(GRIDX), "--", color=INK2, lw=1.5, label="True model")
    for lam, c in ((0.0, ORANGE), (1e-2, BLUE), (10.0, AQUA)):
        ax.plot(GRIDX, poly_features(GRIDX, d) @ fit(XTR, YTR, d, lam), color=c,
                label=f"λ = {lam:g}" + ("  (no penalty)" if lam == 0 else ""))
    ax.scatter(XTR, YTR, s=22, color=INK2, zorder=5)
    ax.set_ylim(-2.2, 2.0); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Same degree-15 features, different λ", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    ax.plot(lams, tr, color=BLUE, label="Training error")
    ax.plot(lams, va, color=ORANGE, label="Validation error")
    ax.plot(lams, te, color=AQUA, lw=1.5, label="Test error")
    ax.axhline(NOISE_SD**2, color=INK2, lw=1, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Regularization strength λ (log scale)"); ax.set_ylabel("MSE (log scale)")
    ax.text(3e-9, 2, "← overfitting", color=ORANGE, fontsize=9)
    ax.text(3, 0.25, "underfitting →", color=BLUE, fontsize=9, ha="right")
    ax.set_title("Error as a function of λ", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "of-ridge.png")


# 7. ------------------------------------------------------ L1 vs L2 geometry
def fig_l1_l2_geometry():
    w_hat = np.array([1.6, 0.55])                     # unregularized optimum
    Hm = np.array([[1.0, 0.45], [0.45, 0.6]])          # loss curvature (elliptical contours)
    loss = lambda W: np.einsum("...i,ij,...j->...", W - w_hat, Hm, W - w_hat)
    t = 1.0
    th = np.linspace(0, 2 * np.pi, 20001)
    circ = np.c_[np.cos(th), np.sin(th)] * t
    diam = np.c_[np.cos(th), np.sin(th)]
    diam = diam / np.abs(diam).sum(1, keepdims=True) * t
    g = np.linspace(-1.7, 2.6, 400)
    W1, W2 = np.meshgrid(g, g)
    L = loss(np.stack([W1, W2], -1))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    for ax, shape, name, col in ((axes[0], diam, "ℓ1 (LASSO):  |w₁| + |w₂| ≤ t", BLUE),
                                 (axes[1], circ, "ℓ2 (Ridge):  w₁² + w₂² ≤ t²", ORANGE)):
        opt = shape[np.argmin(loss(shape))]
        level = loss(opt)
        ax.contour(W1, W2, L, levels=[level * k for k in (0.25, 1, 2.2, 4)], colors=MUTED, linewidths=1)
        ax.contour(W1, W2, L, levels=[level], colors=INK, linewidths=1.6)
        ax.fill(shape[:, 0], shape[:, 1], color=col, alpha=0.18, lw=0)
        ax.plot(shape[:, 0], shape[:, 1], color=col, lw=2)
        ax.plot(*w_hat, "o", color=INK, ms=6)
        ax.annotate("ŵ (no penalty)", w_hat, xytext=(w_hat[0] - 0.15, w_hat[1] + 0.45), fontsize=9)
        ax.plot(*opt, "o", color=col, ms=9, mec=SURF, mew=1.5, zorder=5)
        ax.annotate(f"solution\n({opt[0]:.2f}, {opt[1]:.2f})", opt, xytext=(opt[0] - 0.2, opt[1] - 0.85), ha="right",
                    fontsize=9, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
        ax.set_aspect("equal"); ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.5, 1.8)
        ax.set_xlabel("w₁"); ax.set_ylabel("w₂"); ax.set_title(name, loc="left")
        ax.grid(False)
        print(name, "→", np.round(opt, 3))
    fig.tight_layout()
    save(fig, "of-l1-l2-geometry.png")


# 8. --------------------------------------------------- lasso / ridge paths
def fig_paths():
    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(-2, 1.6, 200))
    r_alphas = np.logspace(-2, 4.5, 200)
    r_coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in r_alphas]).T
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for c in coefs:
        axes[0].plot(alphas, c, color=BLUE, lw=1.4, alpha=0.85)
    for c in r_coefs:
        axes[1].plot(r_alphas, c, color=ORANGE, lw=1.4, alpha=0.85)
    for ax, title in ((axes[0], "LASSO (ℓ1): coefficients hit exactly 0, one by one"),
                      (axes[1], "Ridge (ℓ2): coefficients shrink but never reach 0")):
        ax.set_xscale("log"); ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Penalty strength α (log scale)"); ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("Coefficient value\n(one line per feature, diabetes data)")
    fig.tight_layout()
    save(fig, "of-lasso-ridge-paths.png")


# 9. ------------------------------------------------------ k-fold schematic
def fig_kfold():
    k = 5
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axis("off")
    for r in range(k):
        y0 = k - 1 - r
        for c in range(k):
            is_val = (c == r)
            ax.add_patch(Rectangle((c * 1.0 + 0.04, y0 + 0.1), 0.92, 0.8,
                                   color=ORANGE if is_val else BLUE, alpha=0.9 if is_val else 0.35, lw=0))
            ax.text(c + 0.5, y0 + 0.5, "validate" if is_val else "train", ha="center", va="center",
                    color=SURF if is_val else INK, fontsize=8.5)
        ax.text(-0.15, y0 + 0.5, f"Run {r + 1}", ha="right", va="center", color=INK2)
        ax.text(5.2, y0 + 0.5, f"→ error$_{r + 1}$", ha="left", va="center", color=INK2)
    ax.add_patch(Rectangle((6.6, 0.1), 1.3, k - 0.2, color=MUTED, alpha=0.35, lw=0))
    ax.text(7.25, k / 2, "TEST SET\n\nlocked away\nuntil the\nvery end", ha="center", va="center", color=INK, fontsize=9)
    ax.text(2.5, -0.35, "Training set (the only data the model-selection process may touch)",
            ha="center", color=INK2, fontsize=9)
    ax.text(5.2, -0.35, "CV error = mean of the 5", ha="left", color=INK, fontsize=9)
    ax.set_xlim(-1, 8.1); ax.set_ylim(-0.6, k + 0.1)
    save(fig, "of-kfold.png")


# 10. ---------------------------------------------- leakage: reported vs real
def fig_leakage_results():
    w, r = leak_feature_selection()
    te_w, te_r = leak_target_encoding()
    g_w, g_r = leak_groups()
    t_r, t_w = leak_temporal()
    bv, bt = winners_curse()
    rows = [("Feature selection before CV\n(50 samples, 5000 noise features)", w.mean(), r.mean()),
            ("Target encoding fitted on all rows\n(random target, 1000 categories)", te_w, te_r),
            ("Same patient in train and test\n(record-level vs patient-level split)", g_w, g_r),
            ("Feature that peeks at the future\n(centered rolling mean, random walk)", t_w, t_r),
            ("Best of 1000 models picked on\na 100-sample validation set", bv.mean(), bt.mean())]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ys = np.arange(len(rows))[::-1]
    h = 0.36
    for y0, (name, leaky, honest) in zip(ys, rows):
        ax.barh(y0 + h / 2 + 0.01, leaky, height=h, color=ORANGE)
        ax.barh(y0 - h / 2 - 0.01, honest, height=h, color=BLUE)
        ax.text(leaky + 0.01, y0 + h / 2, f"{leaky:.3f}", va="center", fontsize=8.5, color=INK)
        ax.text(honest + 0.01, y0 - h / 2, f"{honest:.3f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0.5, color=INK, lw=1.2, ls="--")
    ax.text(0.505, len(rows) - 0.45, "chance = 0.5 (the truth in every case)", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.08); ax.set_xlabel("Accuracy")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="Reported (leaky) estimate"),
                       Patch(color=BLUE, label="Honest estimate")],
              loc="lower right", bbox_to_anchor=(1.0, -0.02))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "of-leakage-results.png")


# 11. ------------------------------------------- leakage geometry (selection)
def fig_leak_geometry():
    rng = np.random.default_rng(7)
    n, p = 50, 5000
    X = rng.standard_normal((n, p)); y = np.repeat([0, 1], n // 2)
    top2 = np.argsort(-SelectKBest(f_classif, k=2).fit(X, y).scores_)[:2]
    Xnew = rng.standard_normal((n, p))                     # fresh patients, same features
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, M, title in ((axes[0], X, "The 50 samples used to pick the features"),
                         (axes[1], Xnew, "50 new samples, same two features")):
        for cls, col, mk in ((0, BLUE, "o"), (1, ORANGE, "s")):
            ax.scatter(M[y == cls, top2[0]], M[y == cls, top2[1]], s=40, color=col, marker=mk,
                       edgecolor=SURF, linewidth=1, label=f"class {cls}")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(f"noise feature #{top2[0]}")
    axes[0].set_ylabel(f"noise feature #{top2[1]}")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Out of 5000 pure-noise features, the 2 'most predictive' ones look separable — "
                 "but only on the data that chose them", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "of-leak-geometry.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]: globals()[name]()
        raise SystemExit
    fig_poly_fits(); fig_error_vs_degree(); fig_bias_variance(); fig_learning_curves()
    fig_early_stopping(); fig_ridge(); fig_l1_l2_geometry(); fig_paths(); fig_kfold()
    fig_leakage_results(); fig_leak_geometry()
