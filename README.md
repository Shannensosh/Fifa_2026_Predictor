# ⚽ Predict·26 — FIFA World Cup 2026 Prediction Engine

A transparent, ML-driven model that estimates **each team's % chance of winning
the 2026 World Cup**, presented as an interactive HTML dashboard styled with the
**"Velocity Strike"** design system (`../FIFA UI/DESIGN.md`).

---

## Quick start

```bash
cd "FIFA 2026 Predictor"
python3 build_dashboard.py   # trains LR, runs 20,000 sims, writes dashboard.html  (~2.5 s)
open dashboard.html
```

Requirements: **Python 3.9+** and **NumPy** (included in Anaconda). No internet needed
to view the dashboard — only Google Fonts load online.

To print a text leaderboard only:

```bash
python3 wc2026_model.py
```

---

## Prediction model — three-layer architecture

### Layer 1 · Multinomial Logistic Regression  (statistical weight learning)

The model **does not use hand-tuned weights**. Instead it trains a 3-class
softmax regression on **~98 real World Cup matches** (2010 South Africa,
2014 Brazil, 2018 Russia) to predict Win / Draw / Loss for each team.

| Feature (per match) | Description |
|---|---|
| `elo_diff` | Team A Elo − Team B Elo (estimated at tournament start) |
| `ped_diff` | Pedigree difference: (titles × 8 + WC appearances × 0.6) |
| `host` | +1 if Team A is host, −1 if Team B, 0 otherwise |

The magnitude of the learned Win-class coefficients (scaled by feature std)
determines the relative importance of each feature → **the Power Index weights**.

#### Learned weights (from logistic regression)

| Parameter | Weight | Source |
|---|---|---|
| **Elo rating** | **42.4%** | Learned from regression coefficients |
| **Recent form (last 10 games)** | **28.3%** | Allocated proportionally from Elo split |
| **Squad value / key players** | **20.0%** | Fixed (no historical squad-value records) |
| **Historical pedigree** | **9.3%** | Learned from regression coefficients |
| **Host advantage** | +6 index pts | Flat bonus (regression-confirmed) |
| **Head-to-head record** | ±0–25 Elo | Applied at match level only |

#### Train / test validation

| Set | Tournaments | Matches | Accuracy |
|---|---|---|---|
| Training | 2010, 2014, 2018 | 98 | 64.3% |
| **Test (held-out)** | **2022** | **39** | **61.5%** |
| Naive baseline | — | — | ~57% |

The model improves on the naive "always pick the stronger team" baseline by
**+4.5 percentage points** on unseen data. Football's low-scoring nature means
upsets are common — even the best published models reach 55–65%.

### Layer 2 · Poisson Goals Model  (match-level probability)

Converts the effective Elo gap into expected goals (λ), then enumerates the
joint Poisson scoreline distribution to get exact W/D/L probabilities and the
most likely score:

```
λ_A = μ · exp(+γ · Δelo / 400)      μ = 1.35 (avg WC goals/team/match)
λ_B = μ · exp(−γ · Δelo / 400)      γ = 0.85 (goal sensitivity to Elo edge)
```

### Layer 3 · Monte Carlo Tournament Simulation  (title probability)

Simulates the full 48-team / 12-group 2026 bracket **20,000 times**. Each run:
1. **Group stage** — 12 groups of 4, seeded by Power Index.
2. **Qualification** — top 2 per group + 8 best third-placed = 32 teams.
3. **Knockouts** — R32 → R16 → QF → SF → Final. Draws → strength-weighted shootout.

Counting how often each team wins each round gives title %, final %, semi %, KO-advance %.

---

## Dashboard sections

| Section | What it shows |
|---|---|
| **Title Race** | All 48 teams sorted by win %. Filterable by name/code/confederation. Click a row to expand per-parameter factor bars, key players, effective Elo. |
| **The Model** | Full architecture explanation, learned weights with bars, all engine constants. |
| **Validation** | 64.3% train / 61.5% test accuracy, per-tournament accuracy bars, filterable match-by-match backtest table (correct ✓ / incorrect ✗). |
| **Group Stage** | 12 seeded groups with each team's simulated advance %. |
| **Match Predictor** | Pick any two teams — live W/D/L split, expected goals, most likely score from the Poisson model. |

---

## Files

| File | Purpose |
|---|---|
| `wc2026_data.py` | 48-team dataset: Elo, squad value, key players, form, pedigree, H2H records |
| `wc2026_history.py` | ~137 historical WC match records (2010–2022) with features + outcomes |
| `wc2026_model.py` | Logistic regression, Poisson model, Power Index, Monte Carlo simulation |
| `build_dashboard.py` | Runs the full pipeline → writes `dashboard.html` |
| `dashboard.html` | Self-contained interactive dashboard (no server needed) |

### Tuning the model

All constants are at the top of `wc2026_model.py` (`HOST_BONUS`, `MU`, `GAMMA`, `N_SIMS`).
To update team data edit `wc2026_data.py`, add matches to `wc2026_history.py`, then
re-run `python3 build_dashboard.py`. The logistic regression retrains automatically.

---

## ⚠️ Disclaimer

Statistical model for education and entertainment. Team ratings, squad values,
form and H2H figures are **curated approximations**. The group draw is illustrative
(seeded by rating, not the official FIFA draw). **Not betting advice.**
