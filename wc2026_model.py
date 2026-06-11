"""
FIFA World Cup 2026 — Prediction model  (v2 — sklearn upgrade)

PREDICTION ARCHITECTURE
═══════════════════════

  LAYER 1 · MULTINOMIAL LOGISTIC REGRESSION  (sklearn, sample-weighted)
  ──────────────────────────────────────────────────────────────────────
  Trained on 330 real international matches (WC 2010/14/18,
  UEFA Euro 2012/16/20, Copa América 2015/19/21/24, Nations League
  Finals 2020-21 / 2022-23). Competition importance weights (0.6–1.0)
  down-weight friendlier contests during training.

  Features per match (team_a vs team_b):
    elo_diff    Elo(A) − Elo(B)       primary strength signal
    ped_diff    Pedigree(A) − Ped(B)  WC titles×8 + WC apps×0.6
    host        +1/0/−1               home-field flag
    comp_weight competition weight     passthrough for sample weighting

  Model: sklearn LogisticRegression (multi_class='multinomial',
         solver='lbfgs', C=5, max_iter=2000).
  Evaluation: 5-fold stratified cross-validation + held-out WC 2022.
  Calibration: Brier score (lower = better probability calibration).

  LAYER 2 · POISSON GOALS MODEL  (match-level probability)
  ─────────────────────────────────────────────────────────
      λ_A = μ · exp(+γ · Δelo / 400)
      λ_B = μ · exp(−γ · Δelo / 400)
  Enumerating the joint scoreline distribution gives W/D/L %
  and the most likely score.

  LAYER 3 · MONTE CARLO TOURNAMENT SIMULATION  (title probability)
  ────────────────────────────────────────────────────────────────
  N_SIMS full 48-team / 12-group 2026 brackets.
  Bootstrap CI: repeat N_BOOT=50 mini-sims (2000 runs each) to get
  the 90% confidence interval on every team's title %.

  LAYER 4 · ONLINE LEARNING  (wc2026_live.py)
  ──────────────────────────────────────────────
  As 2026 matches happen, use wc2026_live.py to record results.
  Each result triggers an Elo update (TD-learning rule) and a model
  retrain, so predictions improve throughout the tournament.
"""

import math
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from wc2026_data import get_teams, HEAD_TO_HEAD
from wc2026_history import MATCHES_TRAIN, MATCHES_TEST, get_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
HOST_BONUS    = 6.0
ELO_BASE      = 1450.0
ELO_SPAN      = 7.0
MU            = 1.35
GAMMA         = 0.85
H2H_MAX_NUDGE = 25.0
N_SIMS        = 20000
N_BOOT        = 50       # bootstrap mini-sims for CI  (×2000 runs each)
BOOT_SIMS     = 2000
RNG_SEED      = 2026
CLASSES       = ['W', 'D', 'L']

# Fallback weights (overwritten by the trained regression at run time)
WEIGHTS = {"elo": 0.40, "squad": 0.18, "form": 0.20, "pedigree": 0.12,
           "availability": 0.10}

# Balance regularisation: learned weights are clipped to these bands so no
# single factor dominates ("fine balance"). Excess pedigree weight is
# redistributed to recent form.
PEDIGREE_CAP      = 0.12   # max share for historical pedigree
AVAILABILITY_WGT  = 0.10   # fixed share for injury/availability signal

# Live results file (written by wc2026_live.py)
LIVE_FILE = os.path.join(os.path.dirname(__file__), "live_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — sklearn Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
def _build_clf():
    # multi_class='multinomial' deprecated in sklearn 1.5+; it's now the default
    return LogisticRegression(
        solver='lbfgs',
        C=5,
        max_iter=2000,
        random_state=42,
    )


def _train_model(live_matches=None):
    """Fit on historical + optional live 2026 data.

    Returns: (scaler, clf, X_tr_raw, y_tr, sw_tr,
               X_te_raw, y_te, cv_scores, feat_names)
    """
    # comp_weight is used ONLY as sample_weight (not as a feature)
    # Features are: elo_diff, ped_diff, host  (3 cols)
    X_all, y_tr, sw_tr, _, _ = get_dataset()
    X_tr = [[x[0], x[1], x[2]] for x in X_all]   # drop comp_weight column

    # Append live 2026 matches (WC weight = 1.0)
    if live_matches:
        for m in live_matches:
            X_tr.append([m['elo_diff'], m['ped_diff'], m.get('host', 0)])
            y_tr.append(m['outcome'])
            sw_tr.append(1.0)

    # Test set (WC 2022 — held-out, always, 3 features)
    X_te = [[m['elo_diff'], m['ped_diff'], m.get('host', 0)]
            for m in MATCHES_TEST]
    y_te = [m['outcome'] for m in MATCHES_TEST]

    # Scale features (fit on train only)
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(np.array(X_tr, dtype=float))
    Xs_te = scaler.transform(np.array(X_te, dtype=float))

    # Stratified 5-fold CV on training data
    clf_cv = _build_clf()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # sklearn 1.4+ uses params= dict; older uses fit_params= dict
    # Use manual CV loop to pass sample_weight reliably across versions
    from sklearn.model_selection import StratifiedKFold as _SKF
    cv_raw = []
    for train_idx, val_idx in _SKF(n_splits=5, shuffle=True, random_state=42).split(Xs_tr, y_tr):
        Xf, Xv = Xs_tr[train_idx], Xs_tr[val_idx]
        yf = [y_tr[i] for i in train_idx]
        yv = [y_tr[i] for i in val_idx]
        swf = np.array(sw_tr)[train_idx]
        clf_fold = _build_clf()
        clf_fold.fit(Xf, yf, sample_weight=swf)
        cv_raw.append(clf_fold.score(Xv, yv))
    cv_raw = np.array(cv_raw)

    # Final fit on all training data
    clf = _build_clf()
    clf.fit(Xs_tr, y_tr, sample_weight=np.array(sw_tr))

    feat_names = ['elo_diff', 'ped_diff', 'host']
    return scaler, clf, X_tr, y_tr, sw_tr, X_te, y_te, cv_raw, feat_names


def _derive_weights(clf, scaler, feat_names):
    """
    Convert logistic regression Win-class coefficients into Power Index weights.

    The coefficient for comp_weight is excluded (it's a training signal, not a
    team attribute).  Squad value (not in historical records) is allocated 20%.
    Form is split from the Elo signal.
    """
    # Win-class coefficients × feature std → natural-scale importance
    win_idx = CLASSES.index('W')
    coef = clf.coef_[win_idx]             # shape (n_feat,)
    std  = scaler.scale_                  # feature std from training data
    imp  = np.abs(coef) * std
    imp_named = dict(zip(feat_names, imp))

    # Use only elo_diff and ped_diff for the weight split
    elo_raw = imp_named.get('elo_diff', 1.0)
    ped_raw = imp_named.get('ped_diff', 0.1)
    denom   = elo_raw + ped_raw if (elo_raw + ped_raw) > 0 else 1.0

    remaining = 0.80                  # 20% fixed for squad
    elo_full  = remaining * elo_raw / denom
    ped_w     = remaining * ped_raw / denom
    form_w    = elo_full * 0.40       # form carved from elo's share
    elo_w     = elo_full * 0.60

    total = elo_w + 0.20 + form_w + ped_w
    elo_w, squad_w = elo_w / total, 0.20 / total
    form_w, ped_w  = form_w / total, ped_w / total

    # Balance regularisation: cap pedigree, excess flows to recent form.
    if ped_w > PEDIGREE_CAP:
        form_w += ped_w - PEDIGREE_CAP
        ped_w   = PEDIGREE_CAP

    # Carve a fixed share for injury/availability from all factors pro-rata.
    scale = 1.0 - AVAILABILITY_WGT
    return {
        "elo":          round(elo_w   * scale, 4),
        "squad":        round(squad_w * scale, 4),
        "form":         round(form_w  * scale, 4),
        "pedigree":     round(ped_w   * scale, 4),
        "availability": round(AVAILABILITY_WGT, 4),
    }, dict(zip(feat_names, [round(float(v), 4) for v in imp]))


def _match_explanation(scaler, clf, elo_diff, ped_diff, host):
    """Per-feature log-odds attribution for one match (team_a vs team_b).

    Returns dict: feature → contribution toward P(Win) in log-odds units.
    """
    feat_names = ['elo_diff', 'ped_diff', 'host']
    raw = np.array([[elo_diff, ped_diff, host]])
    z   = scaler.transform(raw)[0]              # standardised feature vector
    win_idx = CLASSES.index('W')
    coef    = clf.coef_[win_idx]
    contribs = {f: round(float(coef[i] * z[i]), 4) for i, f in enumerate(feat_names)}
    contribs['bias'] = round(float(clf.intercept_[win_idx]), 4)
    return contribs


def _backtest_rows(scaler, clf):
    """Per-match backtest for all train + test matches."""
    feat_names = ['elo_diff', 'ped_diff', 'host', 'comp_weight']
    rows = []
    for pool, split in [(MATCHES_TRAIN, 'train'), (MATCHES_TEST, 'test')]:
        for m in pool:
            x = [[m['elo_diff'], m['ped_diff'], m.get('host', 0)]]
            xs = scaler.transform(np.array(x, dtype=float))
            proba = clf.predict_proba(xs)[0]
            proba_dict = dict(zip(clf.classes_, proba))
            pw = proba_dict.get('W', 0)
            pd_ = proba_dict.get('D', 0)
            pl  = proba_dict.get('L', 0)
            pred = max(proba_dict, key=proba_dict.get)
            rows.append({
                'year': m['year'], 'tournament': m.get('tournament', 'WC'),
                'label': m['label'], 'split': split,
                'actual': m['outcome'], 'predicted': pred,
                'p_win':  round(pw  * 100, 1),
                'p_draw': round(pd_ * 100, 1),
                'p_loss': round(pl  * 100, 1),
                'correct': int(pred == m['outcome']),
            })
    return rows


def _brier(scaler, clf, X_raw, y):
    xs = scaler.transform(np.array([[r[0], r[1], r[2]] for r in X_raw], dtype=float))
    proba = clf.predict_proba(xs)
    scores = []
    for c, idx in zip(clf.classes_, range(len(clf.classes_))):
        y_bin = [1 if yi == c else 0 for yi in y]
        scores.append(brier_score_loss(y_bin, proba[:, idx]))
    return round(float(np.mean(scores)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Poisson Match Model
# ─────────────────────────────────────────────────────────────────────────────
def _h2h_nudge(code_a, code_b):
    rec  = HEAD_TO_HEAD.get((code_a, code_b))
    flip = 1.0
    if rec is None:
        rec  = HEAD_TO_HEAD.get((code_b, code_a))
        flip = -1.0
    if rec is None:
        return 0.0
    a_w, draws, b_w = rec
    total = a_w + draws + b_w
    return 0.0 if total == 0 else flip * (a_w - b_w) / total * H2H_MAX_NUDGE


def _lambdas(elo_a, elo_b):
    d = elo_a - elo_b
    return (max(0.15, min(5.0, MU * math.exp(GAMMA * d / 400.0))),
            max(0.15, min(5.0, MU * math.exp(-GAMMA * d / 400.0))))


def _pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def match_prediction(team_a, team_b, max_goals=10):
    elo_a  = team_a["eff_elo"] + _h2h_nudge(team_a["code"], team_b["code"])
    la, lb = _lambdas(elo_a, team_b["eff_elo"])
    pa = [_pmf(i, la) for i in range(max_goals + 1)]
    pb = [_pmf(i, lb) for i in range(max_goals + 1)]
    p_win = p_draw = p_loss = 0.0
    best_score, best_p = (0, 0), 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pa[i] * pb[j]
            if p > best_p: best_p, best_score = p, (i, j)
            if i > j:    p_win  += p
            elif i == j: p_draw += p
            else:        p_loss += p
    tot = p_win + p_draw + p_loss
    return {"p_win": p_win/tot, "p_draw": p_draw/tot, "p_loss": p_loss/tot,
            "xg_a": la, "xg_b": lb, "likely_score": best_score}


# ─────────────────────────────────────────────────────────────────────────────
# Power Index (Feature Engineering)
# ─────────────────────────────────────────────────────────────────────────────
def _scale(v, lo, hi):
    return 0.5 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))


def compute_power(teams, weights=None):
    if weights is None:
        weights = WEIGHTS
    elos   = [t["elo"] for t in teams]
    vals   = [math.sqrt(t["squad_value_m"]) for t in teams]
    peds   = [t["titles"] * 8 + t["wc_appearances"] * 0.6 for t in teams]
    elo_lo, elo_hi = min(elos), max(elos)
    val_lo, val_hi = min(vals), max(vals)
    ped_lo, ped_hi = min(peds), max(peds)
    for t in teams:
        elo_n   = _scale(t["elo"], elo_lo, elo_hi)
        val_n   = _scale(math.sqrt(t["squad_value_m"]), val_lo, val_hi)
        form_n  = t["form_points"] / 30.0
        ped_n   = _scale(t["titles"] * 8 + t["wc_appearances"] * 0.6, ped_lo, ped_hi)
        # Availability uses an absolute scale (not field-relative): a team
        # missing 15% of its first-choice XI scores 0.85 here regardless of
        # how injured the rest of the field is.
        avail_n = t.get("availability", 100) / 100.0
        power   = 100.0 * (
            weights["elo"]      * elo_n  +
            weights["squad"]    * val_n  +
            weights["form"]     * form_n +
            weights["pedigree"] * ped_n  +
            weights.get("availability", 0.0) * avail_n
        )
        if t["host"]:
            power += HOST_BONUS
        t["factors"] = {
            "elo": round(elo_n*100, 1), "squad": round(val_n*100, 1),
            "form": round(form_n*100, 1), "pedigree": round(ped_n*100, 1),
            "availability": round(avail_n*100, 1),
            "host_bonus": HOST_BONUS if t["host"] else 0.0,
        }
        t["power_index"] = round(power, 1)
        t["eff_elo"]     = ELO_BASE + power * ELO_SPAN
    return teams


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Monte Carlo Tournament Simulation
# ─────────────────────────────────────────────────────────────────────────────
GROUP_LETTERS = [chr(ord("A") + i) for i in range(12)]


def build_groups(teams):
    ordered = sorted(teams, key=lambda t: t["power_index"], reverse=True)
    pots    = [ordered[i*12:(i+1)*12] for i in range(4)]
    groups  = {g: [] for g in GROUP_LETTERS}
    for p, pot in enumerate(pots):
        order = range(12) if p % 2 == 0 else range(11, -1, -1)
        for slot, gi in enumerate(order):
            groups[GROUP_LETTERS[gi]].append(pot[slot])
    return groups


def _sim_goals(rng, elo_a, elo_b):
    la, lb = _lambdas(elo_a, elo_b)
    return int(rng.poisson(la)), int(rng.poisson(lb))


def _ko_winner(rng, a, b):
    ga, gb = _sim_goals(rng, a["eff_elo"], b["eff_elo"])
    if ga > gb: return a
    if gb > ga: return b
    d   = a["eff_elo"] - b["eff_elo"]
    p_a = 0.5 + max(-0.15, min(0.15, d / 400.0 * 0.15))
    return a if rng.random() < p_a else b


def _simulate_once(rng, group_items, n_sims):
    codes   = [t["code"] for gm in group_items for t in gm[1]]
    champ   = {c: 0 for c in codes};  final   = {c: 0 for c in codes}
    semi    = {c: 0 for c in codes};  quarter = {c: 0 for c in codes}
    advance = {c: 0 for c in codes}
    for _ in range(n_sims):
        winners, runners, thirds = [], [], []
        for g, members in group_items:
            pts = {t["code"]: 0 for t in members}
            gd  = {t["code"]: 0 for t in members}
            gf  = {t["code"]: 0 for t in members}
            for i in range(4):
                for j in range(i+1, 4):
                    a, b = members[i], members[j]
                    ga, gb = _sim_goals(rng, a["eff_elo"], b["eff_elo"])
                    gf[a["code"]] += ga; gf[b["code"]] += gb
                    gd[a["code"]] += ga-gb; gd[b["code"]] += gb-ga
                    if ga > gb:   pts[a["code"]] += 3
                    elif gb > ga: pts[b["code"]] += 3
                    else:         pts[a["code"]] += 1; pts[b["code"]] += 1
            ranked = sorted(members,
                key=lambda t: (pts[t["code"]], gd[t["code"]], gf[t["code"]], rng.random()),
                reverse=True)
            def s(t): return (pts[t["code"]], gd[t["code"]], gf[t["code"]])
            winners.append((ranked[0], *s(ranked[0])))
            runners.append((ranked[1], *s(ranked[1])))
            thirds.append( (ranked[2], *s(ranked[2])))
        thirds.sort(key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)
        quals = [x[0] for x in winners] + [x[0] for x in runners] + [x[0] for x in thirds[:8]]
        for q in quals: advance[q["code"]] += 1
        seeds = sorted(quals, key=lambda t: t["eff_elo"], reverse=True)
        n = len(seeds)
        bracket = [(seeds[i], seeds[n-1-i]) for i in range(n//2)]
        r16 = [_ko_winner(rng, a, b) for a, b in bracket]
        qf  = [_ko_winner(rng, r16[i], r16[i+1]) for i in range(0, len(r16), 2)]
        for t in qf: quarter[t["code"]] += 1
        sf  = [_ko_winner(rng, qf[i], qf[i+1]) for i in range(0, len(qf), 2)]
        for t in sf: semi[t["code"]] += 1
        fi  = [_ko_winner(rng, sf[i], sf[i+1]) for i in range(0, len(sf), 2)]
        for t in fi: final[t["code"]] += 1
        winner = _ko_winner(rng, fi[0], fi[1])
        champ[winner["code"]] += 1
    return champ, final, semi, quarter, advance


def simulate(teams, groups, n_sims=N_SIMS, seed=RNG_SEED, n_boot=N_BOOT, boot_sims=BOOT_SIMS):
    """Run main simulation + bootstrap for confidence intervals."""
    rng         = np.random.default_rng(seed)
    codes       = [t["code"] for t in teams]
    group_items = list(groups.items())

    # Main run
    champ, final, semi, quarter, advance = _simulate_once(rng, group_items, n_sims)

    # Bootstrap CI (50 × 2000 runs with different seeds)
    boot_champ = {c: [] for c in codes}
    for b in range(n_boot):
        brng = np.random.default_rng(seed + b + 1)
        bc, *_ = _simulate_once(brng, group_items, boot_sims)
        for c in codes:
            boot_champ[c].append(100.0 * bc[c] / boot_sims)

    stats = {}
    for c in codes:
        mean  = 100.0 * champ[c] / n_sims
        boots = sorted(boot_champ[c])
        lo    = boots[int(0.05 * n_boot)]
        hi    = boots[int(0.95 * n_boot)]
        stats[c] = {
            "win_pct":    round(mean, 2),
            "win_ci_lo":  round(lo, 2),
            "win_ci_hi":  round(hi, 2),
            "final_pct":  round(100.0 * final[c]   / n_sims, 2),
            "semi_pct":   round(100.0 * semi[c]    / n_sims, 2),
            "quarter_pct":round(100.0 * quarter[c] / n_sims, 2),
            "advance_pct":round(100.0 * advance[c] / n_sims, 2),
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Live match Elo updates (online TD-learning)
# ─────────────────────────────────────────────────────────────────────────────
def _load_live():
    """Load recorded 2026 match results from live_results.json."""
    if not os.path.exists(LIVE_FILE):
        return [], {}
    with open(LIVE_FILE) as f:
        data = json.load(f)
    return data.get("matches", []), data.get("elo_updates", {})


def apply_live_elo(teams, elo_updates):
    """Apply accumulated Elo updates from live 2026 matches."""
    by_code = {t["code"]: t for t in teams}
    for code, delta in elo_updates.items():
        if code in by_code:
            by_code[code]["elo"] = by_code[code]["elo"] + delta
    return teams


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run(n_sims=N_SIMS, verbose=False):
    global WEIGHTS

    # Load live 2026 data
    live_matches, elo_updates = _load_live()

    # 1. Train logistic regression
    scaler, clf, X_tr, y_tr, sw_tr, X_te, y_te, cv_scores, feat_names = \
        _train_model(live_matches=live_matches)

    train_acc  = round(clf.score(
        scaler.transform(np.array(X_tr, dtype=float)), y_tr,
        sample_weight=np.array(sw_tr)) * 100, 1)
    test_acc   = round(clf.score(
        scaler.transform(np.array(X_te, dtype=float)), y_te) * 100, 1)
    cv_mean    = round(float(cv_scores.mean()) * 100, 1)
    cv_std     = round(float(cv_scores.std())  * 100, 1)
    brier_tr   = _brier(scaler, clf, X_tr, y_tr)
    brier_te   = _brier(scaler, clf, X_te, y_te)

    if verbose:
        print(f"LR  train={train_acc}%  cv5={cv_mean}±{cv_std}%  test(WC22)={test_acc}%")
        print(f"Brier  train={brier_tr}  test={brier_te}  (lower=better)")

    # 2. Derive weights from coefficients
    WEIGHTS, feat_imp = _derive_weights(clf, scaler, feat_names)
    if verbose:
        print(f"Learned weights: {WEIGHTS}")

    # 3. Backtest
    bt_rows  = _backtest_rows(scaler, clf)
    bt_by_yr = {}
    for r in bt_rows:
        yr = str(r['year'])+r.get('tournament','')[:2]
        bt_by_yr.setdefault(yr, {'correct':0,'total':0,'split':r['split'],'tournament':r.get('tournament','WC')})
        bt_by_yr[yr]['correct'] += r['correct']
        bt_by_yr[yr]['total']   += 1
    for k,v in bt_by_yr.items():
        v['accuracy'] = round(100*v['correct']/v['total'], 1)

    # 4. Teams — apply live Elo updates, compute Power Index, simulate
    teams = get_teams()
    if elo_updates:
        teams = apply_live_elo(teams, elo_updates)
    teams  = compute_power(teams, weights=WEIGHTS)
    groups = build_groups(teams)
    stats  = simulate(teams, groups, n_sims=n_sims)
    for t in teams:
        t.update(stats[t["code"]])

    model_meta = {
        "type": "Multinomial Logistic Regression (sklearn) + Poisson Goals Model + Monte Carlo (Bootstrap CI)",
        "train_tournaments": ["WC 2010","WC 2014","WC 2018","Euro 2012","Euro 2016",
                              "Euro 2020","Copa 2015","Copa 2019","Copa 2021","Copa 2024",
                              "NLF 2020-21","NLF 2022-23"],
        "test_tournaments": ["WC 2022"],
        "n_train": len(X_tr), "n_test": len(X_te),
        "train_acc": train_acc, "test_acc": test_acc,
        "cv_mean": cv_mean, "cv_std": cv_std,
        "brier_train": brier_tr, "brier_test": brier_te,
        "features": feat_names, "classes": CLASSES,
        "feat_importance": feat_imp,
        "weights": WEIGHTS,
        "n_sims": n_sims, "n_boot": N_BOOT, "boot_sims": BOOT_SIMS,
        "backtest_by_year": bt_by_yr,
        "backtest_rows": bt_rows,
        "live_matches_count": len(live_matches),
        "live_elo_updates": elo_updates,
    }

    # Store scaler+clf on meta for explainability endpoint in dashboard builder
    model_meta["_scaler"] = scaler
    model_meta["_clf"]    = clf

    return teams, groups, model_meta


# ─────────────────────────────────────────────────────────────────────────────
# Public helper — per-matchup log-odds explanation
# ─────────────────────────────────────────────────────────────────────────────
def explain_match(code_a, code_b, teams_by_code, model_meta):
    """Return feature attributions for A vs B (log-odds toward Win for A)."""
    a, b = teams_by_code[code_a], teams_by_code[code_b]
    elo_d = a["elo"] - b["elo"]
    ped_d = (a["titles"]*8 + a["wc_appearances"]*0.6) - (b["titles"]*8 + b["wc_appearances"]*0.6)
    host  = (1 if a.get("host") else -1 if b.get("host") else 0)
    scaler, clf = model_meta["_scaler"], model_meta["_clf"]
    return _match_explanation(scaler, clf, elo_d, ped_d, host)


if __name__ == "__main__":
    teams, groups, meta = run(n_sims=5000, verbose=True)
    teams.sort(key=lambda t: t["win_pct"], reverse=True)
    print(f"\n{'Team':<16}{'Power':>7}{'Win%':>8}{'CI 90%':>14}{'Final%':>8}")
    print("-" * 60)
    for t in teams[:15]:
        ci = f"[{t['win_ci_lo']:.1f}–{t['win_ci_hi']:.1f}]"
        print(f"{t['flag']} {t['name']:<13}{t['power_index']:>7}"
              f"{t['win_pct']:>7.1f}%{ci:>14}{t['final_pct']:>7.1f}%")
    print(f"\nCV 5-fold: {meta['cv_mean']}% ± {meta['cv_std']}%")
    print(f"Test (WC22): {meta['test_acc']}%  Brier test: {meta['brier_test']}")
