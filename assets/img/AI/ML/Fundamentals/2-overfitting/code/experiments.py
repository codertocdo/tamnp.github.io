"""experiments.py — every number quoted in the Overfitting & Data Leakage post.

Python 3.13, NumPy 2.5.3, scikit-learn 1.9.1, pandas 3.0.5. All randomness is seeded.
Run: python experiments.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_wine
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold,
                                     TimeSeriesSplit, cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

SEED = 42
NOISE_SD = 0.25


def true_f(x):
    return 1.5 * x**3 - x**2 - x + 0.5


def poly_features(x, d):
    return np.vander(x, d + 1, increasing=True)


# ---------------------------------------------------------------------------
# E1. Bias-variance decomposition, estimated by Monte Carlo over 500 training sets
# ---------------------------------------------------------------------------
def bias_variance(degrees=range(0, 13), n_train=30, n_sets=500, seed=SEED):
    rng = np.random.default_rng(seed)
    x_eval = rng.uniform(-1, 1, 2000)            # evaluation points ~ same p(x)
    f_eval = true_f(x_eval)
    rows = []
    for d in degrees:
        preds = np.empty((n_sets, x_eval.size))
        for s in range(n_sets):
            x = rng.uniform(-1, 1, n_train)
            y = true_f(x) + NOISE_SD * rng.standard_normal(n_train)
            w, *_ = np.linalg.lstsq(poly_features(x, d), y, rcond=None)
            preds[s] = poly_features(x_eval, d) @ w
        mean_pred = preds.mean(axis=0)
        bias2 = np.mean((mean_pred - f_eval) ** 2)
        var = np.mean(preds.var(axis=0))
        rows.append((d, bias2, var, bias2 + var + NOISE_SD**2))
    return pd.DataFrame(rows, columns=["degree", "bias2", "variance", "expected_mse"])


# ---------------------------------------------------------------------------
# E2. Learning curves: median train / test MSE vs. training-set size
# ---------------------------------------------------------------------------
def learning_curves(degrees=(3, 12), sizes=(15, 20, 30, 50, 100, 200, 500),
                    repeats=200, seed=SEED):
    rng = np.random.default_rng(seed)
    x_te = rng.uniform(-1, 1, 2000)
    y_te = true_f(x_te) + NOISE_SD * rng.standard_normal(x_te.size)
    rows = []
    for d in degrees:
        for n in sizes:
            tr, te = [], []
            for _ in range(repeats):
                x = rng.uniform(-1, 1, n)
                y = true_f(x) + NOISE_SD * rng.standard_normal(n)
                w, *_ = np.linalg.lstsq(poly_features(x, d), y, rcond=None)
                tr.append(np.mean((poly_features(x, d) @ w - y) ** 2))
                te.append(np.mean((poly_features(x_te, d) @ w - y_te) ** 2))
            rows.append((d, n, np.median(tr), np.median(te)))
    return pd.DataFrame(rows, columns=["degree", "n_train", "train_mse", "test_mse"])


# ---------------------------------------------------------------------------
# E3. Early stopping: plain gradient descent on degree-15 features
# ---------------------------------------------------------------------------
def load_cpp_data(path="data.csv"):
    df = pd.read_csv(path)
    return {s: (g.x.to_numpy(), g.y.to_numpy()) for s, g in df.groupby("split")}


def early_stopping(data, d=15, checkpoints=None):
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    Xtr, Xva, Xte = (poly_features(x, d) for x in (xtr, xva, xte))
    mu, sd = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0)          # scaler fitted on TRAIN only
    for X in (Xtr, Xva, Xte):
        X[:, 1:] = (X[:, 1:] - mu) / sd
    n = len(ytr)
    H = 2.0 / n * Xtr.T @ Xtr                                # Hessian of the MSE
    lr = 1.0 / np.linalg.eigvalsh(H).max()                   # safe step size
    if checkpoints is None:
        checkpoints = np.unique(np.logspace(0, 7, 400).astype(int))
    w = np.zeros(d + 1)
    rows, t = [], 0
    for c in checkpoints:
        while t < c:                                         # explicit GD loop
            w -= lr * (2.0 / n) * Xtr.T @ (Xtr @ w - ytr)
            t += 1
        rows.append((t, *(np.mean((X @ w - y) ** 2)
                          for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte)))))
    return pd.DataFrame(rows, columns=["iteration", "train_mse", "val_mse", "test_mse"])


# ---------------------------------------------------------------------------
# E4. Lasso vs. ridge on the diabetes dataset: sparsity
# ---------------------------------------------------------------------------
def lasso_vs_ridge():
    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    rows = []
    for a in (0.01, 0.1, 1, 5, 10, 20):
        l1 = Lasso(alpha=a, max_iter=100_000).fit(X, y).coef_
        l2 = Ridge(alpha=a * 100).fit(X, y).coef_
        rows.append((a, int(np.sum(np.abs(l1) > 1e-10)), int(np.sum(np.abs(l2) > 1e-10))))
    return pd.DataFrame(rows, columns=["lasso_alpha", "lasso_nonzero", "ridge_nonzero(alpha*100)"])


# ---------------------------------------------------------------------------
# E5. Leakage A — feature selection before CV (replicates ESL §7.10.2)
# ---------------------------------------------------------------------------
def leak_feature_selection(n=50, p=5000, k=100, seeds=range(20)):
    wrong, right = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        X = rng.standard_normal((n, p))                       # pure noise
        y = np.repeat([0, 1], n // 2)                          # labels unrelated to X
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        # WRONG: select the k "best" features using ALL samples, then cross-validate
        X_sel = SelectKBest(f_classif, k=k).fit_transform(X, y)
        wrong.append(cross_val_score(KNeighborsClassifier(1), X_sel, y, cv=cv).mean())
        # RIGHT: selection is a step of the model, refitted inside every training fold
        pipe = make_pipeline(SelectKBest(f_classif, k=k), KNeighborsClassifier(1))
        right.append(cross_val_score(pipe, X, y, cv=cv).mean())
    return np.array(wrong), np.array(right)


# ---------------------------------------------------------------------------
# E6. Leakage B — target encoding fitted on the whole dataset
# ---------------------------------------------------------------------------
def leak_target_encoding(n=2000, n_cat=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, n_cat, n)
    y = rng.integers(0, 2, n)                                 # random target
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    # WRONG: replace each category by the mean target over ALL rows (incl. test rows)
    means = pd.Series(y).groupby(cat).mean()
    X_wrong = means.loc[cat].to_numpy().reshape(-1, 1)
    wrong = cross_val_score(LogisticRegression(), X_wrong, y, cv=cv).mean()
    # RIGHT: TargetEncoder inside the pipeline (fitted per training fold, with
    # internal cross-fitting so a row never sees its own label)
    pipe = make_pipeline(TargetEncoder(cv=KFold(5, shuffle=True, random_state=seed)), LogisticRegression())
    right = cross_val_score(pipe, cat.reshape(-1, 1), y, cv=cv).mean()
    return wrong, right


# ---------------------------------------------------------------------------
# E7. Leakage C — the same patient in train and test (non-independence)
# ---------------------------------------------------------------------------
def leak_groups(n_patients=100, per_patient=10, dim=20, seed=SEED):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_patients, dim))
    labels = rng.integers(0, 2, n_patients)                   # diagnosis: random per patient
    groups = np.repeat(np.arange(n_patients), per_patient)
    X = centers[groups] + 0.1 * rng.standard_normal((groups.size, dim))
    y = labels[groups]
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    record_cv = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=seed)).mean()
    patient_cv = cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups).mean()
    return record_cv, patient_cv


# ---------------------------------------------------------------------------
# E8. Leakage D — a "rolling mean" feature that peeks into the future
# ---------------------------------------------------------------------------
def leak_temporal(n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    price = np.cumsum(rng.standard_normal(n))                 # random walk: unpredictable
    s = pd.Series(price)
    y = (s.shift(-1) > s).astype(int)                          # target: does it go up tomorrow?
    past = pd.concat({f"ret_{k}": s.diff(k) for k in (1, 2, 5)}, axis=1)   # legal features
    leaky = past.copy()
    leaky["centered_ma5_gap"] = s.rolling(5, center=True).mean() - s      # uses t+1, t+2 !
    ok = past.notna().all(axis=1) & leaky.notna().all(axis=1) & s.shift(-1).notna()
    cv = TimeSeriesSplit(5)
    model = LogisticRegression()
    honest = cross_val_score(model, past[ok], y[ok], cv=cv).mean()
    leaked = cross_val_score(model, leaky[ok], y[ok], cv=cv).mean()
    return honest, leaked


# ---------------------------------------------------------------------------
# E9. Overfitting the validation set — the winner's curse
# ---------------------------------------------------------------------------
def winners_curse(n_models=1000, n_val=100, n_test=10_000, seeds=range(20)):
    best_val, its_test = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        # Each "model" is a coin-flipper; its true accuracy is exactly 0.5.
        val_acc = rng.binomial(n_val, 0.5, n_models) / n_val
        best = val_acc.max()
        best_val.append(best)
        its_test.append(rng.binomial(n_test, 0.5) / n_test)    # the winner, on fresh data
    return np.array(best_val), np.array(its_test)


# ---------------------------------------------------------------------------
# E10. Scaler leakage: real, but usually tiny
# ---------------------------------------------------------------------------
def leak_scaler(seeds=range(20)):
    X, y = load_wine(return_X_y=True)
    diffs = []
    for s in seeds:
        cv = StratifiedKFold(10, shuffle=True, random_state=s)
        leaked = cross_val_score(KNeighborsClassifier(5), StandardScaler().fit_transform(X), y, cv=cv).mean()
        clean = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X, y, cv=cv).mean()
        diffs.append(leaked - clean)
    return np.array(diffs)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("== E1 bias-variance (Monte Carlo, 500 training sets of n=30) ==")
    print(bias_variance().round(4).to_string(index=False))

    print("\n== E2 learning curves (median over 200 repeats) ==")
    print(learning_curves().round(4).to_string(index=False))

    print("\n== E3 early stopping (degree 15, gradient descent) ==")
    es = early_stopping(load_cpp_data())
    best = es.loc[es.val_mse.idxmin()]
    print(f"best val at iteration {int(best.iteration)}: train={best.train_mse:.4f} "
          f"val={best.val_mse:.4f} test={best.test_mse:.4f}")
    last = es.iloc[-1]
    print(f"after {int(last.iteration)} iterations: train={last.train_mse:.4f} "
          f"val={last.val_mse:.4f} test={last.test_mse:.4f}")
    print(f"test MSE min over run: {es.test_mse.min():.4f} at iteration "
          f"{int(es.loc[es.test_mse.idxmin()].iteration)}")

    print("\n== E4 lasso vs ridge non-zero coefficients (diabetes, 10 features) ==")
    print(lasso_vs_ridge().to_string(index=False))

    print("\n== E5 feature selection leakage (ESL 7.10.2 replication, 20 seeds) ==")
    w, r = leak_feature_selection()
    print(f"wrong CV accuracy: {w.mean():.3f} ± {w.std():.3f}  (min {w.min():.2f}, max {w.max():.2f})")
    print(f"right CV accuracy: {r.mean():.3f} ± {r.std():.3f}")

    print("\n== E6 target-encoding leakage ==")
    print("wrong = %.3f   right = %.3f" % leak_target_encoding())

    print("\n== E7 group leakage (100 patients x 10 records) ==")
    print("record-level KFold = %.3f   patient-level GroupKFold = %.3f" % leak_groups())

    print("\n== E8 temporal leakage (random walk) ==")
    print("past-only features = %.3f   with centered rolling mean = %.3f" % leak_temporal())

    print("\n== E9 winner's curse (1000 coin-flip models, 100 val samples) ==")
    bv, bt = winners_curse()
    print(f"best validation accuracy: {bv.mean():.3f} ± {bv.std():.3f};  same model on test: {bt.mean():.3f}")

    print("\n== E10 scaler leakage on wine, k-NN (leaked - clean), 20 seeds ==")
    dd = leak_scaler()
    print(f"mean diff = {dd.mean():+.4f}, min = {dd.min():+.4f}, max = {dd.max():+.4f}")
