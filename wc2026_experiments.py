"""
wc2026_experiments.py — Resampling / class-balance experiments.

Tests whether handling the W/D/L class imbalance improves accuracy and
probability calibration (Brier score). Compares four configurations under
identical 5-fold stratified CV + held-out WC 2022 evaluation:

  1. baseline            — current model
  2. class_weight        — sklearn class_weight='balanced'
  3. mirror              — mirror augmentation (each A-vs-B also added as
                           B-vs-A with flipped outcome → symmetric, balanced)
  4. mirror + balanced   — both

Mirror augmentation is the principled fix for this structured problem: it
removes the "team_a is usually the favourite" perspective bias and makes the
model symmetric (P(A beats B) is the mirror of P(B beats A)).

Run:  python3 wc2026_experiments.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

from wc2026_history import get_dataset, MATCHES_TEST

CLASSES = ['W', 'D', 'L']
FLIP = {'W': 'L', 'L': 'W', 'D': 'D'}


def _build_clf(class_weight=None):
    return LogisticRegression(solver='lbfgs', C=5, max_iter=2000,
                              random_state=42, class_weight=class_weight)


def _features(matches):
    X = [[m['elo_diff'], m['ped_diff'], m.get('host', 0)] for m in matches]
    y = [m['outcome'] for m in matches]
    return np.array(X, dtype=float), y


def _mirror(X, y, w):
    """Return X,y,w with each row's mirror appended (features negated, W<->L)."""
    Xm = np.vstack([X, -X])
    ym = list(y) + [FLIP[t] for t in y]
    wm = list(w) + list(w)
    return Xm, ym, np.array(wm)


def _brier(clf, scaler, X, y):
    proba = clf.predict_proba(scaler.transform(X))
    scores = []
    for c, idx in zip(clf.classes_, range(len(clf.classes_))):
        yb = [1 if yi == c else 0 for yi in y]
        scores.append(brier_score_loss(yb, proba[:, idx]))
    return float(np.mean(scores))


def evaluate(mirror=False, balanced=False):
    # Training pool (3 features) + sample weights (comp_weight × recency)
    Xall, y, sw, _, _ = get_dataset()
    X = np.array([[r[0], r[1], r[2]] for r in Xall], dtype=float)
    w = np.array(sw)
    cw = 'balanced' if balanced else None

    # ── 5-fold stratified CV (validate on ORIGINAL rows only; mirror only the
    #    training portion of each fold to avoid leakage) ─────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, va in skf.split(X, y):
        Xtr, ytr, wtr = X[tr], [y[i] for i in tr], w[tr]
        if mirror:
            Xtr, ytr, wtr = _mirror(Xtr, ytr, wtr)
        sc = StandardScaler().fit(Xtr)
        clf = _build_clf(cw).fit(sc.transform(Xtr), ytr, sample_weight=wtr)
        Xva, yva = X[va], [y[i] for i in va]
        accs.append(clf.score(sc.transform(Xva), yva))
    cv_mean, cv_std = 100 * np.mean(accs), 100 * np.std(accs)

    # ── Final fit on all training data, eval on held-out WC 2022 ────────────
    Xf, yf, wf = X, list(y), w
    if mirror:
        Xf, yf, wf = _mirror(Xf, yf, wf)
    scaler = StandardScaler().fit(Xf)
    clf = _build_clf(cw).fit(scaler.transform(Xf), yf, sample_weight=wf)

    Xte, yte = _features(MATCHES_TEST)
    test_acc = 100 * clf.score(scaler.transform(Xte), yte)
    brier = _brier(clf, scaler, Xte, yte)

    # Symmetry check: does the model give consistent mirror predictions?
    # P(A beats B) from row r should ≈ P(B beats A as a Loss) from -r
    sample = scaler.transform(Xte[:1])
    p_fwd = clf.predict_proba(sample)[0]
    p_rev = clf.predict_proba(scaler.transform(-Xte[:1]))[0]
    wi, li = list(clf.classes_).index('W'), list(clf.classes_).index('L')
    symmetry_gap = abs(p_fwd[wi] - p_rev[li])  # 0 = perfectly symmetric

    return cv_mean, cv_std, test_acc, brier, symmetry_gap


if __name__ == "__main__":
    configs = [
        ("baseline",          dict(mirror=False, balanced=False)),
        ("class_weight=bal",  dict(mirror=False, balanced=True)),
        ("mirror augment",    dict(mirror=True,  balanced=False)),
        ("mirror + balanced", dict(mirror=True,  balanced=True)),
    ]
    print(f"{'config':<20}{'CV5 acc':>14}{'test(WC22)':>12}{'Brier':>9}{'symGap':>9}")
    print("-" * 64)
    for name, kw in configs:
        cv, std, te, br, sym = evaluate(**kw)
        print(f"{name:<20}{cv:>7.1f}±{std:<4.1f}%{te:>11.1f}%{br:>9.4f}{sym:>9.4f}")
    print()
    print("Brier: lower = better calibrated.  symGap: 0 = perfectly symmetric model.")
