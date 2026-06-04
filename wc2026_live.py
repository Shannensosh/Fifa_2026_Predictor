"""
wc2026_live.py — Online Learning / Adaptive Model for WC 2026

As matches happen during the 2026 tournament, record each result here.
The model automatically:
  1. Updates team Elo ratings using the TD-learning Elo rule
  2. Retrains the logistic regression on historical + 2026 data
  3. Rebuilds the full dashboard with updated title probabilities

WHY THIS IS REINFORCEMENT LEARNING
────────────────────────────────────
The Elo update rule is equivalent to TD(0) learning:

    Elo_A  +=  K · (S − E)
    Elo_B  +=  K · ((1−S) − (1−E))

where:
    S  = actual outcome  (1.0 = A wins, 0.5 = draw, 0.0 = B wins)
    E  = expected outcome  1 / (1 + 10^(−ΔElo/400))
    K  = learning rate   (30 knockout / 25 group stage)

After each match, the "reward signal" (actual vs expected) drives a gradient
update on every team's strength estimate — exactly how Q-learning / TD-learning
works. The logistic regression is then retrained on the growing dataset to
keep the Win/Draw/Loss probabilities well-calibrated.

USAGE
─────
  # Record a result and rebuild the dashboard
  python3 wc2026_live.py add ARG FRA 3 3 final

  # Show all recorded results
  python3 wc2026_live.py list

  # Undo the last result
  python3 wc2026_live.py undo

  # Just rebuild the dashboard (no new result)
  python3 wc2026_live.py rebuild

  # Clear all 2026 results (reset to pre-tournament model)
  python3 wc2026_live.py reset

STAGE VALUES
────────────
  group | r32 | r16 | qf | sf | final
  (affects K-factor: group=20, r32/r16=25, qf/sf/final=30)
"""

import json
import os
import sys
import math
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DIR       = os.path.dirname(os.path.abspath(__file__))
LIVE_FILE = os.path.join(DIR, "live_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# Elo update constants (TD-learning K-factors by stage)
# ─────────────────────────────────────────────────────────────────────────────
K_FACTORS = {
    'group': 20,
    'r32':   25, 'r16': 25,
    'qf':    30, 'sf':  30, 'final': 30,
}
DEFAULT_K = 22


def _expected(elo_a, elo_b):
    """Expected score for team_a  (Elo formula)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _actual(goals_a, goals_b):
    """Actual score for team_a  (1.0 / 0.5 / 0.0)."""
    if goals_a > goals_b: return 1.0
    if goals_a < goals_b: return 0.0
    return 0.5


def _outcome_str(goals_a, goals_b):
    if goals_a > goals_b: return 'W'
    if goals_a < goals_b: return 'L'
    return 'D'


# ─────────────────────────────────────────────────────────────────────────────
# Live store — read / write  live_results.json
# ─────────────────────────────────────────────────────────────────────────────
def _load():
    if not os.path.exists(LIVE_FILE):
        return {"matches": [], "elo_updates": {}}
    with open(LIVE_FILE) as f:
        return json.load(f)


def _save(data):
    with open(LIVE_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Add a match result
# ─────────────────────────────────────────────────────────────────────────────
def add_result(code_a, code_b, goals_a, goals_b, stage, date=None):
    """
    Record one WC 2026 match and apply Elo TD-update.

    Parameters
    ──────────
    code_a, code_b : team codes  (e.g. 'ARG', 'FRA')
    goals_a, goals_b : integer goals scored at 90 min
                       (for AET/pens, use the 90-min score as draw)
    stage  : 'group' | 'r32' | 'r16' | 'qf' | 'sf' | 'final'
    date   : ISO date string (defaults to today)
    """
    # Load current Elo overrides and compute current effective Elos
    from wc2026_data import get_teams
    teams_list = get_teams()
    base_elo   = {t["code"]: t["elo"] for t in teams_list}

    data       = _load()
    elo_upd    = data["elo_updates"]

    elo_a = base_elo.get(code_a, 1850) + elo_upd.get(code_a, 0)
    elo_b = base_elo.get(code_b, 1850) + elo_upd.get(code_b, 0)

    # TD-learning Elo update
    K = K_FACTORS.get(stage.lower(), DEFAULT_K)
    E = _expected(elo_a, elo_b)
    S = _actual(goals_a, goals_b)

    delta_a =  K * (S     - E)
    delta_b =  K * ((1-S) - (1-E))

    elo_upd[code_a] = round(elo_upd.get(code_a, 0) + delta_a, 2)
    elo_upd[code_b] = round(elo_upd.get(code_b, 0) + delta_b, 2)

    # Build match feature row for retraining
    from wc2026_data import TEAMS
    ped_a = TEAMS.get(code_a, (None,)*10)[4]*8 + TEAMS.get(code_a, (None,)*10)[5]*0.6 \
            if code_a in TEAMS else 0.0
    ped_b = TEAMS.get(code_b, (None,)*10)[4]*8 + TEAMS.get(code_b, (None,)*10)[5]*0.6 \
            if code_b in TEAMS else 0.0

    match_record = {
        "team_a":    code_a,
        "team_b":    code_b,
        "goals_a":   goals_a,
        "goals_b":   goals_b,
        "stage":     stage,
        "date":      date or datetime.date.today().isoformat(),
        "elo_diff":  round(elo_a - elo_b, 1),
        "ped_diff":  round(ped_a - ped_b, 2),
        "host":      0,
        "comp_weight": 1.0,
        "outcome":   _outcome_str(goals_a, goals_b),
        "elo_delta_a": round(delta_a, 2),
        "elo_delta_b": round(delta_b, 2),
        "elo_a_before": round(elo_a, 1),
        "elo_b_before": round(elo_b, 1),
        "label":     f"{code_a} vs {code_b}",
        "year":      2026,
        "tournament": "WC26",
    }

    data["matches"].append(match_record)
    data["elo_updates"] = elo_upd
    _save(data)

    print(f"✓  Recorded: {code_a} {goals_a}–{goals_b} {code_b}  [{stage.upper()}]")
    print(f"   Elo change: {code_a} {delta_a:+.1f} → {elo_a+delta_a:.0f}  |  "
          f"{code_b} {delta_b:+.1f} → {elo_b+delta_b:.0f}")
    return match_record


def undo_last():
    data = _load()
    if not data["matches"]:
        print("No matches to undo.")
        return
    last = data["matches"].pop()
    # Reverse Elo deltas
    code_a, code_b = last["team_a"], last["team_b"]
    data["elo_updates"][code_a] = round(data["elo_updates"].get(code_a, 0) - last["elo_delta_a"], 2)
    data["elo_updates"][code_b] = round(data["elo_updates"].get(code_b, 0) - last["elo_delta_b"], 2)
    _save(data)
    print(f"Undone: {last['label']}  {last['goals_a']}–{last['goals_b']}")


def list_results():
    data = _load()
    if not data["matches"]:
        print("No 2026 results recorded yet.")
        return
    print(f"{'Date':<12}{'Match':<22}{'Score':<8}{'Stage':<8}{'ΔElo A':>8}{'ΔElo B':>8}")
    print("-" * 68)
    for m in data["matches"]:
        score = f"{m['goals_a']}–{m['goals_b']}"
        print(f"{m['date']:<12}{m['label']:<22}{score:<8}{m['stage']:<8}"
              f"{m['elo_delta_a']:>+8.1f}{m['elo_delta_b']:>+8.1f}")
    print(f"\nCumulative Elo updates: {data['elo_updates']}")


def reset():
    if os.path.exists(LIVE_FILE):
        os.remove(LIVE_FILE)
    print("Live results cleared. Model reset to pre-tournament state.")


def rebuild():
    """Retrain model and rebuild dashboard.html + index.html."""
    import subprocess, sys
    print("Rebuilding dashboard…")
    result = subprocess.run([sys.executable, "build_dashboard.py"],
                            capture_output=False, cwd=DIR)
    if result.returncode == 0:
        print("✓ Dashboard rebuilt successfully.")
    else:
        print("✗ Build failed — check the output above.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _usage():
    print(__doc__)
    sys.exit(0)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('help', '--help', '-h'):
        _usage()

    cmd = args[0].lower()

    if cmd == 'add':
        # add CODE_A CODE_B GOALS_A GOALS_B STAGE [DATE]
        if len(args) < 6:
            print("Usage: wc2026_live.py add CODE_A CODE_B GOALS_A GOALS_B STAGE [DATE]")
            print("Example: python3 wc2026_live.py add ARG FRA 3 3 final 2026-07-19")
            sys.exit(1)
        code_a  = args[1].upper()
        code_b  = args[2].upper()
        goals_a = int(args[3])
        goals_b = int(args[4])
        stage   = args[5].lower()
        date    = args[6] if len(args) > 6 else None
        add_result(code_a, code_b, goals_a, goals_b, stage, date)
        print("\nRebuilding dashboard with updated model…")
        rebuild()

    elif cmd == 'list':
        list_results()

    elif cmd == 'undo':
        undo_last()
        print("Rebuilding dashboard…")
        rebuild()

    elif cmd == 'reset':
        confirm = input("Reset ALL 2026 results? (yes/no): ")
        if confirm.lower() == 'yes':
            reset()
            rebuild()

    elif cmd == 'rebuild':
        rebuild()

    else:
        print(f"Unknown command: {cmd}")
        _usage()


if __name__ == "__main__":
    main()
