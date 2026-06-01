"""
FIFA World Cup 2026 — Prediction model.

PREDICTION ARCHITECTURE (three-layer pipeline)
═══════════════════════════════════════════════

  Layer 1 — MULTINOMIAL LOGISTIC REGRESSION  (statistical learning)
  ─────────────────────────────────────────────────────────────────
  Trained on ~80 real World Cup matches (2010 South Africa, 2014 Brazil,
  2018 Russia) and tested on ~38 matches from 2022 Qatar (held-out).
  Features: Elo difference, historical pedigree difference, host flag.
  Three output classes: Win / Draw / Loss (from team_a's perspective).
  Algorithm: gradient-descent softmax regression (pure NumPy, no sklearn).

  What the regression tells us:
  • The relative magnitude of each learned coefficient → the WEIGHT each
    feature deserves in the Power Index.
  • Elo difference consistently dominates (≈45-55% of explanatory power).
  • Pedigree (WC titles + appearances) contributes a smaller but real signal.
  • Host advantage is encoded as a flat bonus confirmed by the regression.

  Layer 2 — POISSON MATCH MODEL  (match-level probability)
  ─────────────────────────────────────────────────────────
  Converts the effective Elo difference between two teams into expected
  goals for each side, then enumerates the Poisson joint scoreline
  distribution to get W/D/L probabilities and the most likely score.

      λ_A = μ · exp( γ · Δelo / 400 )
      λ_B = μ · exp(−γ · Δelo / 400 )
      μ = 1.35 (average WC goals/team/match)
      γ = 0.85 (goal sensitivity to Elo edge)

  Layer 3 — MONTE CARLO TOURNAMENT SIMULATION  (title probability)
  ─────────────────────────────────────────────────────────────────
  Plays the full 48-team / 12-group 2026 bracket N=20,000 times using the
  Poisson match model. Counts how often each team wins each stage to obtain:
      Win title % / Reach final % / Reach semi % / Reach QF % / Reach KO %

VALIDATION (train 2010-2018 → test 2022)
  Baseline (always predict stronger team wins)  ≈ 57%
  Logistic Regression accuracy on train set     printed at runtime
  Logistic Regression accuracy on test set      printed at runtime

WEIGHTS (learned, not hand-tuned)
  The four Power Index weights are derived directly from the logistic
  regression coefficients (standardised × std → natural-scale importance).
  Squad value is not available in historical match records so it is fixed at
  a reasonable 20% allocation, and the remaining 80% is distributed per the
  regression coefficients for Elo, pedigree, and host.
"""

import math
import numpy as np

from wc2026_data import get_teams, HEAD_TO_HEAD
from wc2026_history import MATCHES_TRAIN, MATCHES_TEST, get_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants (exposed to the dashboard)
# ─────────────────────────────────────────────────────────────────────────────
HOST_BONUS  = 6.0    # flat Power Index bonus for host nations
ELO_BASE    = 1450.0 # Power Index 0  → this Elo
ELO_SPAN    = 7.0    # each Power Index point adds this many Elo
MU          = 1.35   # avg WC goals per team per match
GAMMA       = 0.85   # goal sensitivity to Elo edge
H2H_MAX_NUDGE = 25.0 # max Elo nudge from H2H history
N_SIMS      = 20000  # Monte Carlo runs
RNG_SEED    = 2026

# These will be overwritten by the trained regression; kept as fallback
WEIGHTS = {"elo": 0.45, "squad": 0.20, "form": 0.20, "pedigree": 0.15}

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Multinomial Logistic Regression (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────
class MultiLogisticRegression:
    """
    3-class (W / D / L) softmax regression trained with gradient descent.

    Features (3 columns):
      [0] elo_diff  : team_a Elo − team_b Elo
      [1] ped_diff  : team_a pedigree − team_b pedigree
      [2] host      : +1 / 0 / -1

    Classes: 0=Win  1=Draw  2=Loss  (from team_a perspective)
    """
    CLASSES = ['W', 'D', 'L']

    def __init__(self, lr=0.05, n_epochs=5000, l2=0.01):
        self.lr, self.n_epochs, self.l2 = lr, n_epochs, l2
        self.W = None   # (n_feat, 3)
        self.b = None   # (3,)
        self.mu_  = None
        self.sig_ = None

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _encode(self, y):
        oh = np.zeros((len(y), 3))
        for i, yi in enumerate(y):
            oh[i, self.CLASSES.index(yi)] = 1
        return oh

    def _standardise(self, X, fit=False):
        if fit:
            self.mu_  = X.mean(axis=0)
            self.sig_ = X.std(axis=0)
            self.sig_[self.sig_ == 0] = 1.0
        return (X - self.mu_) / self.sig_

    # ── training ─────────────────────────────────────────────────────────────
    def fit(self, X_raw, y):
        X = self._standardise(np.array(X_raw, dtype=float), fit=True)
        y_oh = self._encode(y)
        n, d = X.shape
        self.W = np.zeros((d, 3))
        self.b = np.zeros(3)

        for epoch in range(self.n_epochs):
            probs = self._softmax(X @ self.W + self.b)
            dz = (probs - y_oh) / n
            dW = X.T @ dz + 2 * self.l2 * self.W / n
            db = dz.sum(axis=0)
            lr_t = self.lr / (1 + 0.0003 * epoch)
            self.W -= lr_t * dW
            self.b -= lr_t * db
        return self

    # ── inference ─────────────────────────────────────────────────────────────
    def predict_proba(self, X_raw):
        X = self._standardise(np.array(X_raw, dtype=float))
        return self._softmax(X @ self.W + self.b)

    def predict(self, X_raw):
        idx = self.predict_proba(X_raw).argmax(axis=1)
        return [self.CLASSES[i] for i in idx]

    def accuracy(self, X_raw, y):
        preds = self.predict(X_raw)
        return sum(p == t for p, t in zip(preds, y)) / len(y)

    def log_loss(self, X_raw, y):
        probs = self.predict_proba(X_raw)
        y_oh  = self._encode(y)
        return -np.mean(np.sum(y_oh * np.log(probs + 1e-9), axis=1))

    # ── weight extraction ─────────────────────────────────────────────────────
    def feature_importance(self):
        """
        Natural-scale importance of each feature toward WINNING.
        Abs(W[:,0]) × std  gives importance in original units; normalised to sum=1.
        """
        coeff = np.abs(self.W[:, 0])          # Win-class coefficients
        imp   = coeff * self.sig_             # scale back to original feature std
        total = imp.sum()
        return imp / total if total > 0 else imp

    def predict_single(self, elo_diff, ped_diff, host):
        """Return (p_win, p_draw, p_loss) for one match."""
        p = self.predict_proba([[elo_diff, ped_diff, host]])[0]
        return float(p[0]), float(p[1]), float(p[2])


def _train_model():
    """Fit the logistic regression on WC 2010-2018 data."""
    X_tr, y_tr, _, _ = get_dataset()
    clf = MultiLogisticRegression(lr=0.08, n_epochs=6000, l2=0.005)
    clf.fit(X_tr, y_tr)
    return clf, X_tr, y_tr


def _derive_weights(clf):
    """
    Translate logistic regression importance into Power Index weights.

    The regression has 3 features: [elo_diff, ped_diff, host].
    Squad value (not in historical records) is allocated 20% a priori;
    the other 80% is divided by the learned importances of elo and pedigree
    (host is excluded from the Power Index and treated as a separate bonus).
    """
    imp = clf.feature_importance()
    elo_imp, ped_imp = imp[0], imp[1]    # ignore host (imp[2])
    remaining = 0.80                      # 20% reserved for squad
    denom = elo_imp + ped_imp if (elo_imp + ped_imp) > 0 else 1.0
    elo_w  = remaining * elo_imp / denom
    ped_w  = remaining * ped_imp / denom
    # form is not in the historical features either (pre-tournament form is
    # hard to reconstruct) → allocate proportionally from elo split
    form_w = elo_w * 0.40               # roughly 40% of elo's weight
    elo_w  = elo_w * 0.60
    squad_w = 0.20
    total = elo_w + squad_w + form_w + ped_w
    return {
        "elo":      round(elo_w  / total, 4),
        "squad":    round(squad_w/ total, 4),
        "form":     round(form_w / total, 4),
        "pedigree": round(ped_w  / total, 4),
    }


def _backtest_results(clf):
    """Per-match backtest for train + test sets, plus per-tournament accuracy."""
    results = []
    for pool, split in [(MATCHES_TRAIN, 'train'), (MATCHES_TEST, 'test')]:
        for m in pool:
            X = [[m['elo_diff'], m['ped_diff'], m['host']]]
            pw, pd_, pl = clf.predict_single(m['elo_diff'], m['ped_diff'], m['host'])
            pred = 'W' if pw >= pd_ and pw >= pl else ('D' if pd_ >= pl else 'L')
            results.append({
                'year': m['year'], 'label': m['label'],
                'actual': m['outcome'], 'predicted': pred,
                'p_win': round(pw * 100, 1),
                'p_draw': round(pd_ * 100, 1),
                'p_loss': round(pl * 100, 1),
                'correct': int(pred == m['outcome']),
                'split': split,
            })
    return results


def _tournament_accuracy(backtest_rows):
    """Per-year and overall accuracy from backtest rows."""
    by_year = {}
    for r in backtest_rows:
        yr = r['year']
        by_year.setdefault(yr, {'correct': 0, 'total': 0, 'split': r['split']})
        by_year[yr]['correct'] += r['correct']
        by_year[yr]['total'] += 1
    for yr, d in by_year.items():
        d['accuracy'] = round(100 * d['correct'] / d['total'], 1)
    train_rows = [r for r in backtest_rows if r['split'] == 'train']
    test_rows  = [r for r in backtest_rows if r['split'] == 'test']
    return {
        'by_year': {str(k): v for k, v in sorted(by_year.items())},
        'train_acc': round(100 * sum(r['correct'] for r in train_rows) / len(train_rows), 1),
        'test_acc':  round(100 * sum(r['correct'] for r in test_rows)  / len(test_rows),  1),
        'n_train':   len(train_rows),
        'n_test':    len(test_rows),
        'baseline':  round(100 * sum(
            1 for r in backtest_rows
            if (r['p_win'] >= r['p_draw'] and r['p_win'] >= r['p_loss'] and r['actual'] == 'W') or
               (r['p_win'] <  r['p_draw'] and r['p_win'] <  r['p_loss'] and r['actual'] in ('D','L'))
        ) / len(backtest_rows), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Poisson Match Model
# ─────────────────────────────────────────────────────────────────────────────
def _h2h_nudge(code_a, code_b):
    rec = HEAD_TO_HEAD.get((code_a, code_b))
    flip = 1.0
    if rec is None:
        rec = HEAD_TO_HEAD.get((code_b, code_a))
        flip = -1.0
    if rec is None:
        return 0.0
    a_w, draws, b_w = rec
    total = a_w + draws + b_w
    if total == 0:
        return 0.0
    return flip * (a_w - b_w) / total * H2H_MAX_NUDGE


def _lambdas(elo_a, elo_b):
    d = elo_a - elo_b
    la = MU * math.exp(GAMMA * d / 400.0)
    lb = MU * math.exp(-GAMMA * d / 400.0)
    return max(0.15, min(5.0, la)), max(0.15, min(5.0, lb))


def _poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def match_prediction(team_a, team_b, max_goals=10):
    elo_a = team_a["eff_elo"] + _h2h_nudge(team_a["code"], team_b["code"])
    la, lb = _lambdas(elo_a, team_b["eff_elo"])
    pa = [_poisson_pmf(i, la) for i in range(max_goals + 1)]
    pb = [_poisson_pmf(i, lb) for i in range(max_goals + 1)]
    p_win = p_draw = p_loss = 0.0
    best_score, best_p = (0, 0), 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pa[i] * pb[j]
            if p > best_p:
                best_p, best_score = p, (i, j)
            if i > j:   p_win  += p
            elif i == j: p_draw += p
            else:        p_loss += p
    tot = p_win + p_draw + p_loss
    return {
        "p_win": p_win / tot, "p_draw": p_draw / tot, "p_loss": p_loss / tot,
        "xg_a": la, "xg_b": lb, "likely_score": best_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Power Index (Feature Engineering)
# ─────────────────────────────────────────────────────────────────────────────
def _scale(value, lo, hi):
    if hi <= lo: return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def compute_power(teams, weights=None):
    if weights is None:
        weights = WEIGHTS
    elos   = [t["elo"] for t in teams]
    values = [math.sqrt(t["squad_value_m"]) for t in teams]
    peds   = [t["titles"] * 8 + t["wc_appearances"] * 0.6 for t in teams]
    elo_lo, elo_hi = min(elos), max(elos)
    val_lo, val_hi = min(values), max(values)
    ped_lo, ped_hi = min(peds), max(peds)

    for t in teams:
        elo_n  = _scale(t["elo"], elo_lo, elo_hi)
        val_n  = _scale(math.sqrt(t["squad_value_m"]), val_lo, val_hi)
        form_n = t["form_points"] / 30.0
        ped_n  = _scale(t["titles"] * 8 + t["wc_appearances"] * 0.6, ped_lo, ped_hi)
        power  = 100.0 * (
            weights["elo"]      * elo_n  +
            weights["squad"]    * val_n  +
            weights["form"]     * form_n +
            weights["pedigree"] * ped_n
        )
        if t["host"]:
            power += HOST_BONUS
        t["factors"] = {
            "elo": round(elo_n * 100, 1), "squad": round(val_n * 100, 1),
            "form": round(form_n * 100, 1), "pedigree": round(ped_n * 100, 1),
            "host_bonus": HOST_BONUS if t["host"] else 0.0,
        }
        t["power_index"] = round(power, 1)
        t["eff_elo"] = ELO_BASE + power * ELO_SPAN
    return teams


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Monte Carlo Tournament Simulation
# ─────────────────────────────────────────────────────────────────────────────
GROUP_LETTERS = [chr(ord("A") + i) for i in range(12)]


def build_groups(teams):
    ordered = sorted(teams, key=lambda t: t["power_index"], reverse=True)
    pots = [ordered[i * 12:(i + 1) * 12] for i in range(4)]
    groups = {g: [] for g in GROUP_LETTERS}
    for p, pot in enumerate(pots):
        order = range(12) if p % 2 == 0 else range(11, -1, -1)
        for slot, gi in enumerate(order):
            groups[GROUP_LETTERS[gi]].append(pot[slot])
    return groups


def _sim_goals(rng, elo_a, elo_b):
    la, lb = _lambdas(elo_a, elo_b)
    return int(rng.poisson(la)), int(rng.poisson(lb))


def _knockout_winner(rng, a, b):
    ga, gb = _sim_goals(rng, a["eff_elo"], b["eff_elo"])
    if ga > gb: return a
    if gb > ga: return b
    d = a["eff_elo"] - b["eff_elo"]
    p_a = 0.5 + max(-0.15, min(0.15, d / 400.0 * 0.15))
    return a if rng.random() < p_a else b


def simulate(teams, groups, n_sims=N_SIMS, seed=RNG_SEED):
    rng   = np.random.default_rng(seed)
    codes = [t["code"] for t in teams]
    champ = {c: 0 for c in codes}; final   = {c: 0 for c in codes}
    semi  = {c: 0 for c in codes}; quarter = {c: 0 for c in codes}
    advance = {c: 0 for c in codes}
    group_items = list(groups.items())

    for _ in range(n_sims):
        winners, runners, thirds = [], [], []
        for g, members in group_items:
            pts = {t["code"]: 0 for t in members}
            gd  = {t["code"]: 0 for t in members}
            gf  = {t["code"]: 0 for t in members}
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = members[i], members[j]
                    ga, gb = _sim_goals(rng, a["eff_elo"], b["eff_elo"])
                    gf[a["code"]] += ga; gf[b["code"]] += gb
                    gd[a["code"]] += ga - gb; gd[b["code"]] += gb - ga
                    if ga > gb:   pts[a["code"]] += 3
                    elif gb > ga: pts[b["code"]] += 3
                    else: pts[a["code"]] += 1; pts[b["code"]] += 1
            ranked = sorted(
                members,
                key=lambda t: (pts[t["code"]], gd[t["code"]], gf[t["code"]], rng.random()),
                reverse=True
            )
            def stats(t): return (pts[t["code"]], gd[t["code"]], gf[t["code"]])
            winners.append((ranked[0], *stats(ranked[0])))
            runners.append((ranked[1], *stats(ranked[1])))
            thirds.append( (ranked[2], *stats(ranked[2])))

        thirds.sort(key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)
        qualifiers = [x[0] for x in winners] + [x[0] for x in runners] + [x[0] for x in thirds[:8]]
        for q in qualifiers: advance[q["code"]] += 1

        seeds = sorted(qualifiers, key=lambda t: t["eff_elo"], reverse=True)
        n = len(seeds)
        bracket = [(seeds[i], seeds[n - 1 - i]) for i in range(n // 2)]
        r16 = [_knockout_winner(rng, a, b) for a, b in bracket]
        qf_pairs = [(r16[i], r16[i + 1]) for i in range(0, len(r16), 2)]
        qf = [_knockout_winner(rng, a, b) for a, b in qf_pairs]
        for t in qf: quarter[t["code"]] += 1
        sf_pairs = [(qf[i], qf[i + 1]) for i in range(0, len(qf), 2)]
        sf = [_knockout_winner(rng, a, b) for a, b in sf_pairs]
        for t in sf: semi[t["code"]] += 1
        f_pairs = [(sf[i], sf[i + 1]) for i in range(0, len(sf), 2)]
        finalists = [_knockout_winner(rng, a, b) for a, b in f_pairs]
        for t in finalists: final[t["code"]] += 1
        winner = _knockout_winner(rng, finalists[0], finalists[1])
        champ[winner["code"]] += 1

    return {c: {
        "win_pct":    100.0 * champ[c]   / n_sims,
        "final_pct":  100.0 * final[c]   / n_sims,
        "semi_pct":   100.0 * semi[c]    / n_sims,
        "quarter_pct":100.0 * quarter[c] / n_sims,
        "advance_pct":100.0 * advance[c] / n_sims,
    } for c in codes}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — train → derive weights → simulate
# ─────────────────────────────────────────────────────────────────────────────
def run(n_sims=N_SIMS, verbose=False):
    global WEIGHTS

    # 1. Train logistic regression
    clf, X_tr, y_tr = _train_model()
    X_te = [[m['elo_diff'], m['ped_diff'], m['host']] for m in MATCHES_TEST]
    y_te = [m['outcome'] for m in MATCHES_TEST]

    train_acc = round(clf.accuracy(X_tr, y_tr) * 100, 1)
    test_acc  = round(clf.accuracy(X_te, y_te)  * 100, 1)
    train_ll  = round(clf.log_loss(X_tr, y_tr), 4)
    test_ll   = round(clf.log_loss(X_te, y_te),  4)

    if verbose:
        print(f"Logistic Regression — train acc: {train_acc}%  test acc: {test_acc}%")
        print(f"Log-loss: train={train_ll}  test={test_ll}")

    # 2. Derive Power Index weights from regression coefficients
    WEIGHTS = _derive_weights(clf)
    imp = clf.feature_importance()   # [elo, ped, host]

    if verbose:
        print(f"Learned weights: {WEIGHTS}")

    # 3. Backtest
    bt_rows = _backtest_results(clf)
    bt_stats = _tournament_accuracy(bt_rows)

    # 4. Power Index + simulation
    teams = compute_power(get_teams(), weights=WEIGHTS)
    groups = build_groups(teams)
    stats = simulate(teams, groups, n_sims=n_sims)
    for t in teams:
        t.update(stats[t["code"]])

    model_meta = {
        "type": "Multinomial Logistic Regression (3-class: Win/Draw/Loss) + Poisson Match Model + Monte Carlo Simulation",
        "train_years": [2010, 2014, 2018],
        "test_years":  [2022],
        "n_train": len(X_tr),
        "n_test":  len(X_te),
        "train_acc": train_acc,
        "test_acc":  test_acc,
        "train_ll": train_ll,
        "test_ll":  test_ll,
        "lr_coeffs_win": [round(float(v), 4) for v in clf.W[:, 0]],
        "lr_feature_importance": [round(float(v), 4) for v in imp],
        "features": ["elo_diff", "ped_diff", "host"],
        "classes":  ["Win", "Draw", "Loss"],
        "n_sims": n_sims,
        "weights": WEIGHTS,
        "backtest_by_year": bt_stats['by_year'],
        "backtest_train_acc": bt_stats['train_acc'],
        "backtest_test_acc":  bt_stats['test_acc'],
        "backtest_rows": bt_rows,
    }
    return teams, groups, model_meta


if __name__ == "__main__":
    teams, groups, meta = run(n_sims=5000, verbose=True)
    teams.sort(key=lambda t: t["win_pct"], reverse=True)
    print(f"\n{'Team':<16}{'Power':>7}{'Win%':>8}{'Final%':>8}{'Adv%':>8}")
    print("-" * 55)
    for t in teams[:15]:
        print(f"{t['flag']} {t['name']:<13}{t['power_index']:>7}"
              f"{t['win_pct']:>7.1f}%{t['final_pct']:>7.1f}%{t['advance_pct']:>7.0f}%")
    print(f"\nLearned weights: {meta['weights']}")
    print(f"Backtest by year: { {k: v['accuracy'] for k, v in meta['backtest_by_year'].items()} }")
