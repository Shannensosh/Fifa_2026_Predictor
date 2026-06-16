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
softmax regression (sklearn) on **411 real international matches**:

| Source | Matches | Competition weight |
|---|---|---|
| World Cup 2010 / 2014 / 2018 | 95 | 1.00 |
| UEFA Euro 2012 / 2016 / 2020 / **2024** | 165 | 0.88 |
| Copa América 2015 / 2019 / 2021 / 2024 | 99 | 0.85 |
| Nations League Finals 2021 / 2023 / **2025** | 12 | 0.78 |
| WC 2026 European qualifiers & playoffs (2025-26) | 17 | 0.60 |
| **June 2026 pre-tournament friendlies** | 23 | 0.55 |

Two weighting schemes apply on top of competition importance:
- **Recency decay** (7-year half-life): a Euro 2024 match counts ~3.3× more
  than a WC 2010 match, so the model is anchored to current football.
- **Balance regularisation**: the learned pedigree weight is capped at 12%
  (excess flows to recent form) so historical glory can't outvote current quality.

| Feature (per match) | Description |
|---|---|
| `elo_diff` | Team A Elo − Team B Elo (estimated at tournament start) |
| `ped_diff` | Pedigree difference: (titles × 8 + WC appearances × 0.6) |
| `host` | +1 if Team A is host, −1 if Team B, 0 otherwise |

#### Learned weights (current build)

| Parameter | Weight | Source |
|---|---|---|
| **Elo rating** | **34.3%** | Learned from regression coefficients |
| **Recent form (last 10 games)** | **22.9%** | Carved from Elo share + pedigree overflow |
| **Squad value / key players** | **16.0%** | Fixed allocation |
| **Fitness / availability (injuries)** | **10.0%** | June 2026 squad news, % of first XI fit |
| **Squad trajectory / age** | **10.0%** | Avg starting-XI age — rewards rising, penalises ageing squads |
| **Historical pedigree** | **6.8%** | Learned, under the 12% balance cap |
| **Host advantage** | +6 index pts | Flat bonus (regression-confirmed) |
| **Head-to-head record** | ±0–25 Elo | Applied at match level only |

#### Squad trajectory factor (new)

Average starting-XI age maps to a 0–100 trajectory score: **age 24.5 → 100**
(youngest, most upside), **age 30.5 → 0** (ageing). This shifts the forecast
toward squad quality/momentum rather than pure historical rating — e.g. it moved
**Argentina** (avg age 29.6, the ageing 2022 champions) from #1 down to #3, and
lifted **Spain** (avg age 25.3, Euro 2024 champions) and **France** (prime window)
to the top — closer to bookmaker/pundit consensus.

#### Injury / availability signal (new)

Each team carries an `availability` score = % of its first-choice XI fully fit,
from June 2026 squad reporting (e.g. Brazil 72% — Rodrygo, Militão, Estêvão out;
Netherlands 68% — Timber, Simons, Schouten, de Ligt out; Portugal 100%).
Players out/doubtful are listed per team in the dashboard's expandable rows.

#### Real field + official group draw (verified June 2026)

The 48-team field and the full **official group draw (groups A–L, drawn 5 Dec 2025)**
are the real ones, cross-verified against Wikipedia and NBC Sports. This replaced
an earlier synthetic seeded draw, so the simulation now plays the *actual* group
fixtures — e.g. Spain drew a tough Group H with Uruguay, while Argentina got a
kinder Group J. Eight teams that missed qualification (Italy, Denmark, Serbia,
Poland, Nigeria, Cameroon, Costa Rica, Jamaica) were replaced by the real
qualifiers, including debutants **Curaçao, Cape Verde, Jordan** and returnees
**Haiti, DR Congo, Iraq, South Africa, Czechia**. Italy lost the UEFA playoff
final to Bosnia on penalties — a third straight missed World Cup.

*(The knockout bracket is still strength-seeded rather than the exact official
R32 crossings — a known remaining simplification.)*

#### Train / test validation

| Set | Tournaments | Matches | Accuracy |
|---|---|---|---|
| Training | 2010–2026 (6 competition types) | 411 | 56.2% |
| 5-fold CV | stratified | 411 | 56.0% ± 2.7% |
| **Test (held-out)** | **WC 2022** | **39** | **59.0%** |
| Naive baseline | — | — | ~57% |

Adding the June 2026 friendlies lifted CV accuracy 53.4% → 56.0% and improved
calibration (Brier 0.177 → 0.172).

#### Tuning & methodology improvements (see `wc2026_tuning.py`)

- **Dropped `host` from the regression** — ablation showed it was near-dead-weight
  and slightly *hurt* calibration (most WC matches are neutral-venue). Removing it
  improved Brier 0.173 → 0.170 at the same 64.1% accuracy. The 2026 host edge is
  still applied as the separate flat +6 Power-Index bonus.
- **Margin-of-victory Elo** (`wc2026_live.py`) — the live update now multiplies by a
  goal-difference factor (×1 / ×1.5 / ×(11+GD)/8), so a 7-1 win shifts ratings far
  more than a 1-0. Standard World-Football-Elo practice.
- **Dixon-Coles correction** (ρ = −0.08) on the Poisson match model — lifts the
  low-score cells so the simulated draw rate (≈26%) matches reality (25.6% in the
  test set), vs 24.1% for plain independent Poisson. Fixes the favourite-vs-minnow
  draw under-prediction.

#### What does *not* help (tested, see `wc2026_experiments.py` / `wc2026_tuning.py`)

- **Hyperparameter tuning** — the model is insensitive to `C` (0.05→10) and to
  L1/L2/elastic-net; no meaningful gain available.

- **Up/down-sampling or class balancing** — *hurts*. Brier degrades 0.177 → 0.202
  and CV accuracy drops ~5 pts. The W/D/L class frequencies are real signal
  (favourites genuinely win ~48%), so forcing balance makes the model over-predict
  upsets. **Conclusion: do not resample.**
- **Richer engineered features** (Elo², interactions, win-expectancy) — no gain;
  test accuracy unchanged and Brier slightly worse (overfitting). `elo_diff`
  already captures the strength signal near-optimally. Real accuracy gains would
  need genuinely new data (xG, lineups, rest days), not feature tweaks.

Accuracy is lower than the WC-only v1 because the expanded dataset contains many
more evenly-matched fixtures (Euro/Copa knockouts), which are intrinsically harder
to call — but the probabilities are better calibrated (Brier 0.177 on test) and
generalise beyond World Cup conditions.

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

**Confidence intervals** on each title % use the analytic **Wilson score interval**
on the championship proportion — the exact sampling-uncertainty interval for a
Monte-Carlo estimate, computed in O(1). This replaced an earlier 100k-run
bootstrap, cutting the full build from **~13s to ~3s** with no loss of accuracy.

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
