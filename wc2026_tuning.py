"""
wc2026_tuning.py — hyperparameter tuning + feature ablation (read-only analysis).

Answers two questions on the held-out-validated part of the model
(the logistic regression trained on historical + live matches):

  1. HYPERPARAMETERS — best regularisation strength C, penalty type
     (L2 / L1 / elastic-net) by 5-fold CV log-loss + held-out WC 2022.
  2. FEATURE ABLATION — drop each of [elo_diff, ped_diff, host] and measure
     the impact, so we can see what information is (un)necessary.

Metrics: CV accuracy, test accuracy, test log-loss and Brier (calibration).
For probability forecasts, log-loss / Brier matter more than top-1 accuracy.

This script does NOT modify the model or dashboard — analysis only.
Run:  python3 wc2026_tuning.py
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss

warnings.filterwarnings("ignore")

from wc2026_history import get_dataset, MATCHES_TEST
from wc2026_model import _load_live

FEATS = ["elo_diff", "ped_diff", "host"]
CLASSES = ["W", "D", "L"]


def load():
    """Replicate the production training set: historical + live 2026 matches."""
    Xall, y, sw, _, _ = get_dataset()
    X = [[r[0], r[1], r[2]] for r in Xall]
    y = list(y); sw = list(sw)
    live, _ = _load_live()
    for m in live:
        X.append([m["elo_diff"], m["ped_diff"], m.get("host", 0)])
        y.append(m["outcome"]); sw.append(1.0)
    Xte = [[m["elo_diff"], m["ped_diff"], m.get("host", 0)] for m in MATCHES_TEST]
    yte = [m["outcome"] for m in MATCHES_TEST]
    return np.array(X, float), y, np.array(sw), np.array(Xte, float), yte


def make_clf(C=5, penalty="l2", l1_ratio=None):
    if penalty == "l2":
        return LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=6000)
    if penalty == "l1":
        return LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=8000)
    if penalty == "elasticnet":
        return LogisticRegression(penalty="elasticnet", solver="saga", C=C,
                                  l1_ratio=l1_ratio, max_iter=8000)


def evaluate(X, y, sw, Xte, yte, cols, C=5, penalty="l2", l1_ratio=None):
    Xc, Xtec = X[:, cols], Xte[:, cols]
    # 5-fold stratified CV (sample-weighted), validate on original rows
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, lls = [], []
    for tr, va in skf.split(Xc, y):
        sc = StandardScaler().fit(Xc[tr])
        clf = make_clf(C, penalty, l1_ratio).fit(sc.transform(Xc[tr]),
                                                  [y[i] for i in tr], sample_weight=sw[tr])
        yv = [y[i] for i in va]
        accs.append(clf.score(sc.transform(Xc[va]), yv))
        lls.append(log_loss(yv, clf.predict_proba(sc.transform(Xc[va])), labels=clf.classes_.tolist()))
    # final fit + held-out test
    sc = StandardScaler().fit(Xc)
    clf = make_clf(C, penalty, l1_ratio).fit(sc.transform(Xc), y, sample_weight=sw)
    proba = clf.predict_proba(sc.transform(Xtec))
    test_acc = clf.score(sc.transform(Xtec), yte)
    test_ll = log_loss(yte, proba, labels=clf.classes_.tolist())
    brier = np.mean([brier_score_loss([1 if v == c else 0 for v in yte], proba[:, i])
                     for i, c in enumerate(clf.classes_)])
    # coefficient magnitudes (Win class) for surviving-feature view
    wi = list(clf.classes_).index("W")
    coefs = dict(zip([FEATS[c] for c in cols], np.round(clf.coef_[wi], 3)))
    return (100*np.mean(accs), 100*np.std(accs), 100*test_acc, test_ll, brier, coefs)


if __name__ == "__main__":
    X, y, sw, Xte, yte = load()
    print(f"Training rows: {len(X)}  (historical + live)   Test rows: {len(Xte)} (WC2022)\n")

    print("═══ 1. HYPERPARAMETER GRID (all 3 features) ═══")
    print(f"{'penalty':<12}{'C':>6}{'CV acc':>14}{'test acc':>10}{'test LL':>10}{'Brier':>9}")
    print("-"*64)
    best = None
    for penalty, l1r in [("l2", None), ("l1", None), ("elasticnet", 0.5)]:
        for C in [0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10]:
            cv, sd, te, ll, br, _ = evaluate(X, y, sw, Xte, yte, [0,1,2], C, penalty, l1r)
            tag = f"{penalty:<12}{C:>6}{cv:>7.1f}±{sd:<4.1f}{te:>9.1f}%{ll:>10.4f}{br:>9.4f}"
            print(tag)
            score = (ll, -cv)  # prefer lower test log-loss, then higher CV
            if best is None or score < best[0]:
                best = (score, penalty, C, l1r, cv, te, ll, br)
    print(f"\n  → best by test log-loss: {best[1]} C={best[2]}  "
          f"(CV {best[4]:.1f}%, test {best[5]:.1f}%, LL {best[6]:.4f}, Brier {best[7]:.4f})")
    print(f"  → current production: l2 C=5\n")

    print("═══ 2. FEATURE ABLATION (l2, C=5) ═══")
    print(f"{'features':<26}{'CV acc':>14}{'test acc':>10}{'test LL':>10}{'Brier':>9}")
    print("-"*70)
    subsets = [
        ("elo only",            [0]),
        ("ped only",            [1]),
        ("host only",           [2]),
        ("elo + ped",           [0,1]),
        ("elo + host",          [0,2]),
        ("ped + host",          [1,2]),
        ("elo + ped + host",    [0,1,2]),
    ]
    for name, cols in subsets:
        cv, sd, te, ll, br, _ = evaluate(X, y, sw, Xte, yte, cols, C=5)
        print(f"{name:<26}{cv:>7.1f}±{sd:<4.1f}{te:>9.1f}%{ll:>10.4f}{br:>9.4f}")

    print("\n═══ 3. L1 FEATURE SELECTION (does any feature get zeroed?) ═══")
    for C in [0.1, 0.3, 1.0]:
        _,_,_,_,_, coefs = evaluate(X, y, sw, Xte, yte, [0,1,2], C, "l1")
        print(f"  C={C}: Win-class coefficients {coefs}")
    print("\n(Brier/LL lower = better. CV on {} rows; test = held-out WC2022.)".format(len(X)))
