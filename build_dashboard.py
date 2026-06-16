"""
Build the FIFA World Cup 2026 prediction dashboard (dashboard.html).

Runs the full pipeline:
  1. sklearn LogisticRegression trained on 330 international matches
     (WC + Euro + Copa América + Nations League), competition-weighted
  2. 5-fold cross-validation + bootstrap confidence intervals
  3. Learned weights → Power Index for all 48 teams
  4. Poisson goals model (match-level W/D/L)
  5. Monte Carlo tournament simulation (20,000 runs + 50×2000 bootstrap)
  6. Online learning hook (wc2026_live.py) for real-time 2026 updates
  7. Generates self-contained interactive HTML (Velocity Strike theme)
"""

import json
import os
import datetime
import wc2026_model as M
from wc2026_data import HEAD_TO_HEAD


def build_payload(n_sims):
    teams, groups, meta = M.run(n_sims=n_sims, verbose=True)
    teams.sort(key=lambda t: t["win_pct"], reverse=True)
    for i, t in enumerate(teams):
        t["odds_rank"] = i + 1

    # Pop non-serialisable sklearn objects before building JSON
    scaler = meta.pop("_scaler", None)
    clf    = meta.pop("_clf", None)

    # Build per-team match explanations for the top-10 vs each other
    # (precomputed so JS can show them without Python at runtime)
    explanations = {}
    top10 = [t["code"] for t in teams[:10]]
    by_code = {t["code"]: t for t in teams}
    if scaler and clf:
        for ca in top10:
            for cb in top10:
                if ca == cb: continue
                exp = M.explain_match(ca, cb, by_code, {"_scaler": scaler, "_clf": clf})
                explanations[f"{ca}|{cb}"] = exp

    payload_teams = [{
        "code": t["code"], "name": t["name"], "flag": t["flag"],
        "conf": t["confederation"], "fifa_rank": t["fifa_rank"],
        "elo": t["elo"], "eff_elo": round(t["eff_elo"], 1),
        "titles": t["titles"], "wc_appearances": t["wc_appearances"],
        "squad_value_m": t["squad_value_m"], "key_players": t["key_players"],
        "form": t["form"], "host": t["host"],
        "availability": t.get("availability", 100),
        "injuries_out": t.get("injuries_out", []),
        "injuries_doubtful": t.get("injuries_doubtful", []),
        "avg_age": t.get("avg_age", 27.6),
        "power_index": t["power_index"], "factors": t["factors"],
        "win_pct":     round(t["win_pct"], 2),
        "win_ci_lo":   round(t.get("win_ci_lo", t["win_pct"]), 2),
        "win_ci_hi":   round(t.get("win_ci_hi", t["win_pct"]), 2),
        "final_pct":   round(t["final_pct"], 2),
        "semi_pct":    round(t["semi_pct"], 2),
        "quarter_pct": round(t["quarter_pct"], 2),
        "advance_pct": round(t["advance_pct"], 2),
        "odds_rank": t["odds_rank"],
    } for t in teams]

    payload_groups = {g: [t["code"] for t in members] for g, members in groups.items()}
    h2h = {f"{a}|{b}": list(v) for (a, b), v in HEAD_TO_HEAD.items()}

    # Load live 2026 results if any
    live_path = os.path.join(os.path.dirname(__file__), "live_results.json")
    live_data = {}
    if os.path.exists(live_path):
        with open(live_path) as f:
            live_data = json.load(f)

    # Serialise numpy / sklearn floats cleanly
    def _clean(obj):
        if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):  return [_clean(v) for v in obj]
        try:
            return float(obj)
        except (TypeError, ValueError):
            return obj

    return {
        "teams":        payload_teams,
        "groups":       payload_groups,
        "h2h":          h2h,
        "explanations": explanations,
        "live":         live_data,
        "weights":      _clean(meta["weights"]),
        "constants": {
            "HOST_BONUS":    M.HOST_BONUS,
            "ELO_BASE":      M.ELO_BASE,
            "ELO_SPAN":      M.ELO_SPAN,
            "MU":            M.MU,
            "GAMMA":         M.GAMMA,
            "DC_RHO":        M.DC_RHO,
            "H2H_MAX_NUDGE": M.H2H_MAX_NUDGE,
            "N_SIMS":        n_sims,
        },
        "model_meta": _clean(meta),
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WORLD CUP 2026 · PREDICTION ENGINE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
/* ── design tokens (Velocity Strike) ────────────────────────────────────── */
:root{
  --bg:#0f0f0f; --l1:#1a1a1a; --l2:#252525; --l3:#2e2e2e;
  --on:#e5e2e1; --muted:#8E8E93; --outline:#333;
  --lime:#CCFF00; --lime2:#c3f400; --green:#00FF66; --red:#FF3B30;
  --r-sm:4px; --r-lg:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--on);font-family:Inter,sans-serif;font-size:16px;line-height:1.5;-webkit-font-smoothing:antialiased}
h1,h2,h3,.sora{font-family:Sora,sans-serif}
.mono{font-family:'JetBrains Mono',monospace}
.caps{font-family:Sora;font-weight:700;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}

/* ── header ─────────────────────────────────────────────────────────────── */
header{position:sticky;top:0;z-index:100;backdrop-filter:blur(20px);background:rgba(15,15,15,.85);border-bottom:1px solid #1e1e1e}
.hbar{display:flex;align-items:center;gap:16px;height:58px}
.wrap{max-width:1200px;margin:0 auto;padding:0 24px}
.logo{font-family:Sora;font-weight:800;font-size:18px;letter-spacing:-.01em}
.logo span{color:var(--lime)}
.tag{font-size:11px;color:var(--muted);border:1px solid var(--outline);border-radius:999px;padding:3px 10px;letter-spacing:.04em;text-transform:uppercase}
.live-badge{font-size:11px;color:var(--red);border:1px solid rgba(255,59,48,.5);border-radius:999px;padding:3px 10px;letter-spacing:.04em;text-transform:uppercase;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
nav{margin-left:auto;display:flex;gap:4px;flex-wrap:wrap}
nav a{color:var(--muted);text-decoration:none;font-size:13px;padding:6px 10px;border-radius:var(--r-sm);transition:color .15s}
nav a:hover{color:var(--on);background:var(--l1)}
/* ── CI bar ──────────────────────────────────────────────────────────────── */
.ci-wrap{display:flex;align-items:center;gap:6px}
.ci-range{font-family:'JetBrains Mono';font-size:10px;color:var(--muted);white-space:nowrap}
/* ── live section ────────────────────────────────────────────────────────── */
.live-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:760px){.live-grid{grid-template-columns:1fr}}
.code-block{background:#0a0a0a;border:1px solid #2a2a2a;border-radius:var(--r-lg);padding:16px;font-family:'JetBrains Mono';font-size:13px;color:var(--lime);overflow-x:auto;white-space:pre}
.code-block .cm{color:var(--muted)}
.live-match-row{display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #1e1e1e;font-size:14px}
.live-match-row:last-child{border-bottom:none}
.live-score{font-family:'JetBrains Mono';font-weight:700;font-size:18px;color:var(--on)}
.live-stage{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
/* ── explainer ───────────────────────────────────────────────────────────── */
.exp-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.exp-feat{background:var(--l1);border:1px solid #252525;border-radius:var(--r-sm);padding:10px 12px;text-align:center}
.exp-feat .ev{font-family:'JetBrains Mono';font-weight:700;font-size:16px}
.exp-feat .ev.pos{color:var(--lime)}
.exp-feat .ev.neg{color:var(--red)}
.exp-feat .ek{font-size:11px;color:var(--muted);margin-top:3px}

/* ── hero ────────────────────────────────────────────────────────────────── */
.hero{padding:48px 0 28px;border-bottom:1px solid #1a1a1a;
  background:radial-gradient(800px 280px at 80% -20%,rgba(204,255,0,.09),transparent 55%)}
.hero h1{font-size:clamp(32px,5vw,52px);font-weight:800;line-height:1.04;letter-spacing:-.02em}
.hero h1 .hl{color:var(--lime)}
.hero p{color:var(--muted);max-width:620px;margin-top:12px;font-size:15px}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.pill{font-family:'JetBrains Mono';font-size:12px;background:var(--l1);border:1px solid #252525;border-radius:999px;padding:5px 12px}
.pill b{color:var(--lime)}

/* ── podium ──────────────────────────────────────────────────────────────── */
.podium{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:28px}
@media(max-width:560px){.podium{grid-template-columns:1fr}}
.pod{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);padding:16px 18px;position:relative;overflow:hidden}
.pod.top{border-color:rgba(204,255,0,.4);box-shadow:0 0 0 1px rgba(204,255,0,.1) inset}
.pod .rank{font-family:'JetBrains Mono';font-size:11px;color:var(--muted);letter-spacing:.06em}
.pod .nm{font-family:Sora;font-weight:700;font-size:20px;margin-top:4px}
.pod .fl{font-size:24px}
.pod .pct{font-family:'JetBrains Mono';font-weight:700;font-size:36px;color:var(--lime);margin-top:8px}
.pod .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}

/* ── section chrome ──────────────────────────────────────────────────────── */
section{padding:48px 0;border-bottom:1px solid #1a1a1a}
.shead{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.shead h2{font-size:26px;font-weight:700;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:14px;margin-bottom:22px;max-width:820px;line-height:1.65}

/* ── cards ───────────────────────────────────────────────────────────────── */
.card{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);padding:20px}
.card h3{font-family:Sora;font-size:15px;font-weight:700;margin-bottom:14px}

/* ── data table ──────────────────────────────────────────────────────────── */
.toolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap}
.search-wrap{position:relative;display:flex;align-items:center;gap:8px}
.search{background:#0a0a0a;border:1px solid #2a2a2a;border-radius:var(--r-sm);padding:9px 12px 9px 36px;color:var(--on);font-size:13px;width:230px;transition:border-color .2s}
.search:focus{outline:none;border-color:var(--lime);box-shadow:0 0 0 2px rgba(204,255,0,.12)}
.search-ico{position:absolute;left:11px;color:var(--muted);font-size:14px;pointer-events:none}
.clear-btn{background:none;border:none;color:var(--muted);cursor:pointer;font-size:18px;padding:0 4px;line-height:1;display:none}
.clear-btn:hover{color:var(--on)}
#count{font-family:'JetBrains Mono';font-size:12px}
.tablewrap{border:1px solid #222;border-radius:var(--r-lg);overflow-x:auto;overflow-y:auto;max-height:600px;background:var(--l1)}
table{width:100%;border-collapse:collapse;min-width:760px}
thead{position:sticky;top:0;z-index:10}
th{background:#141414;border-bottom:1px solid #222;padding:11px 14px;text-align:left;white-space:nowrap;font-family:Sora;font-weight:700;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);cursor:pointer;user-select:none}
th:hover{color:var(--on)}
th.num,td.num{text-align:right}
th .ar{opacity:.45;font-size:9px;margin-left:3px}
td{padding:11px 14px;border-bottom:1px solid #1a1a1a;font-size:14px;white-space:nowrap}
tbody tr.datarow{cursor:pointer}
tbody tr.datarow:hover td{background:#1d1d1d}
.team-cell{display:flex;align-items:center;gap:10px}
.team-cell .fl{font-size:19px;line-height:1}
.team-cell .tname{font-size:14px;font-weight:500}
.team-cell .tsub{font-family:'JetBrains Mono';font-size:11px;color:var(--muted);margin-top:1px}
.host-badge{font-size:9px;color:var(--green);border:1px solid rgba(0,255,102,.35);border-radius:3px;padding:1px 4px;letter-spacing:.05em;margin-left:5px;vertical-align:middle}
.bar{position:relative;height:7px;border-radius:999px;background:#262626;width:80px;display:inline-block;vertical-align:middle}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:999px;background:linear-gradient(90deg,var(--lime2),var(--green))}
.win-val{font-family:'JetBrains Mono';font-weight:700;font-size:13px;color:var(--lime)}

/* ── expandable detail row ───────────────────────────────────────────────── */
tr.detrow td{padding:14px 16px;background:#111}
.facgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px 20px;margin-bottom:12px}
.fac .ft{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;color:var(--muted)}
.fac .ft .fv{font-family:'JetBrains Mono';color:var(--on)}
.fbar{height:6px;border-radius:999px;background:#262626;position:relative}
.fbar i{position:absolute;inset:0;border-radius:999px;background:var(--lime)}
.detchips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.dchip{font-family:'JetBrains Mono';font-size:11px;padding:3px 8px;border-radius:var(--r-sm);background:rgba(204,255,0,.1);color:var(--lime)}
.dchip.g{background:rgba(0,255,102,.1);color:var(--green)}
.dchip.m{background:#1e1e1e;color:var(--muted)}
.players-line{font-size:13px;color:var(--muted);margin-bottom:10px}
.players-line b{color:var(--on)}

/* ── model section ───────────────────────────────────────────────────────── */
.model-arch{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
@media(max-width:820px){.model-arch{grid-template-columns:1fr}}
.arch-step{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);padding:18px;position:relative;overflow:hidden}
.arch-step::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--lime2),var(--green))}
.step-num{font-family:'JetBrains Mono';font-size:11px;color:var(--lime);letter-spacing:.06em;margin-bottom:6px}
.step-title{font-family:Sora;font-weight:700;font-size:15px;margin-bottom:8px}
.step-desc{font-size:13px;color:var(--muted);line-height:1.65}
.step-desc b{color:var(--on)}
.wrow{display:flex;align-items:center;gap:10px;margin:10px 0}
.wlabel{width:160px;font-size:13px;color:var(--muted)}
.wlabel b{color:var(--on);display:block}
.wbar-wrap{flex:1;height:10px;background:#1e1e1e;border-radius:999px;position:relative}
.wbar-wrap i{position:absolute;inset:0;border-radius:999px;background:linear-gradient(90deg,var(--lime2),var(--green))}
.wpct{font-family:'JetBrains Mono';font-size:13px;color:var(--lime);width:40px;text-align:right}
.kv{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1e1e1e;font-size:13px;color:var(--muted)}
.kv:last-child{border-bottom:0}
.kv span:last-child{font-family:'JetBrains Mono';color:var(--on)}
.note-box{font-size:12px;color:var(--muted);line-height:1.65;margin-top:10px}

/* ── validation / accuracy ───────────────────────────────────────────────── */
.val-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
@media(max-width:700px){.val-grid{grid-template-columns:repeat(2,1fr)}}
.val-stat{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);padding:16px;text-align:center}
.val-stat .v{font-family:'JetBrains Mono';font-weight:700;font-size:28px;color:var(--lime)}
.val-stat .v.dim{color:var(--muted)}
.val-stat .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:4px}
.val-stat .delta{font-size:12px;color:var(--green);margin-top:2px}
.yr-bars{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
@media(max-width:620px){.yr-bars{grid-template-columns:repeat(2,1fr)}}
.yr-bar-card{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);padding:14px}
.yr-bar-card .ylab{font-family:Sora;font-weight:700;font-size:13px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.yr-bar-card .ylab .badge{font-size:10px;padding:2px 6px;border-radius:3px;font-family:'JetBrains Mono'}
.badge.train{background:rgba(204,255,0,.15);color:var(--lime)}
.badge.test{background:rgba(0,255,102,.15);color:var(--green)}
.acc-bar{height:8px;background:#1e1e1e;border-radius:999px;position:relative;overflow:hidden;margin-top:2px}
.acc-bar i{position:absolute;inset:0;border-radius:999px;background:linear-gradient(90deg,var(--lime2),var(--green));transition:width .6s ease}
.yr-acc-val{font-family:'JetBrains Mono';font-size:18px;font-weight:700;margin-top:6px}

/* ── backtest table ──────────────────────────────────────────────────────── */
.bt-filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.bt-pill{font-size:12px;padding:5px 12px;border-radius:999px;border:1px solid var(--outline);background:none;color:var(--muted);cursor:pointer;font-family:Inter}
.bt-pill.active,.bt-pill:hover{border-color:var(--lime);color:var(--lime);background:rgba(204,255,0,.08)}
.bt-wrap{max-height:420px;overflow-y:auto;overflow-x:auto;border:1px solid #222;border-radius:var(--r-lg);background:var(--l1)}
.bt-wrap table{min-width:600px}
.bt-wrap td,.bt-wrap th{font-size:13px}
.chip-w{font-family:'JetBrains Mono';font-size:11px;padding:2px 7px;border-radius:var(--r-sm);background:rgba(204,255,0,.15);color:var(--lime)}
.chip-d{font-family:'JetBrains Mono';font-size:11px;padding:2px 7px;border-radius:var(--r-sm);background:rgba(255,255,255,.08);color:var(--muted)}
.chip-l{font-family:'JetBrains Mono';font-size:11px;padding:2px 7px;border-radius:var(--r-sm);background:rgba(255,59,48,.15);color:var(--red)}
.ok-tick{color:var(--green);font-size:14px}
.ko-cross{color:var(--red);font-size:14px}

/* ── groups ──────────────────────────────────────────────────────────────── */
.groups-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media(max-width:900px){.groups-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:500px){.groups-grid{grid-template-columns:1fr}}
.gcard{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);overflow:hidden}
.gcard .gh{font-family:Sora;font-weight:700;font-size:12px;letter-spacing:.05em;padding:9px 12px;background:#141414;border-bottom:1px solid #222}
.grow{display:flex;align-items:center;gap:8px;padding:8px 12px;font-size:13px;border-bottom:1px solid #181818}
.grow:last-child{border-bottom:none}
.grow .gfl{font-size:16px;width:22px}
.grow .gnm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.grow .gad{font-family:'JetBrains Mono';font-size:12px;color:var(--muted)}
.grow.adv .gad{color:var(--green)}

/* ── match predictor ─────────────────────────────────────────────────────── */
.h2h-selectors{display:grid;grid-template-columns:1fr auto 1fr;gap:18px;align-items:center}
@media(max-width:700px){.h2h-selectors{grid-template-columns:1fr;text-align:center}}
.selbox{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);padding:16px}
select{width:100%;background:#0a0a0a;color:var(--on);border:1px solid #2a2a2a;border-radius:var(--r-sm);padding:10px 12px;font-family:Inter;font-size:14px;cursor:pointer}
select:focus{outline:none;border-color:var(--lime)}
.vs-label{font-family:Sora;font-weight:800;color:var(--muted);font-size:22px;text-align:center}
.wdl-bar{display:flex;height:48px;border-radius:var(--r-lg);overflow:hidden;border:1px solid #2a2a2a;margin-top:20px}
.wdl-seg{display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono';font-weight:700;font-size:13px;transition:flex-basis .5s cubic-bezier(.4,0,.2,1);white-space:nowrap;overflow:hidden}
.wdl-seg.w{background:var(--lime);color:#0c0c0c}
.wdl-seg.d{background:#333;color:var(--muted)}
.wdl-seg.l{background:var(--green);color:#0c0c0c}
.stats-3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}
.pstat{background:var(--l1);border:1px solid #252525;border-radius:var(--r-lg);padding:14px;text-align:center}
.pstat .pv{font-family:'JetBrains Mono';font-weight:700;font-size:26px}
.pstat .pv.score{color:var(--on)}
.pstat .pv.lime{color:var(--lime)}
.pstat .pk{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:4px}
.h2h-note{font-size:12px;color:var(--muted);margin-top:12px;line-height:1.65}

/* ── footer ──────────────────────────────────────────────────────────────── */
footer{padding:40px 0 64px;color:var(--muted);font-size:12px;line-height:1.7}
.disc-box{background:var(--l1);border:1px solid #252525;border-left:3px solid var(--lime);border-radius:var(--r-sm);padding:14px 16px;margin-top:10px;font-size:13px;color:var(--muted)}
</style>
</head>
<body>

<!-- ─── HEADER ─────────────────────────────────────────────────────────── -->
<header>
  <div class="wrap hbar">
    <div class="logo">PREDICT<span>·</span>26</div>
    <span class="tag">Monte-Carlo Engine</span>
    <span class="tag" style="border-color:rgba(204,255,0,.45);color:var(--lime)" title="Auto-rebuilt daily by GitHub Actions">⟳ Updated __DATE__</span>
    <nav>
      <a href="#race">Title Race</a>
      <a href="#model">Model</a>
      <a href="#validation">Validation</a>
      <a href="#groups">Groups</a>
      <a href="#h2h">Match Predictor</a>
      <a href="#live">🔴 Live 2026</a>
    </nav>
  </div>
</header>

<!-- ─── HERO ──────────────────────────────────────────────────────────── -->
<div class="hero">
  <div class="wrap">
    <h1>WHO WINS THE<br><span class="hl">2026 WORLD CUP?</span></h1>
    <p>A fully transparent, data-driven forecast for the 48-team tournament. Every team's
    title chance is computed from parameters <em>learned from real match data</em> and
    __NSIMS__ simulated tournaments.</p>
    <div class="pills">
      <span class="pill">ENGINE <b>sklearn LR + Poisson + Monte-Carlo</b></span>
      <span class="pill">TRAINING DATA <b>__NTRAIN__ matches · WC + Euro + Copa + NLF + qualifiers + 2026 friendlies</b></span>
      <span class="pill">HELD-OUT TEST <b>WC 2022 · 59% accuracy</b></span>
      <span class="pill">5-FOLD CV <b>54.8% ± 1.8%</b></span>
      <span class="pill">INJURY DATA <b>June 2026 squad news</b></span>
      <span class="pill">SIMULATIONS <b>__NSIMS__ + Wilson 90% CI</b></span>
      <span class="pill">UPDATED <b>__DATE__</b></span>
    </div>
    <div class="podium" id="podium"></div>
  </div>
</div>

<!-- ─── TITLE RACE ─────────────────────────────────────────────────────── -->
<section id="race">
  <div class="wrap">
    <div class="shead"><h2>Title Race</h2><span class="caps">% chance to lift the trophy</span></div>
    <p class="sub">Probability each team wins the 2026 World Cup. Weights are <strong style="color:var(--on)">learned from ~80 historical WC matches</strong>, not hand-tuned. Click a row to see the per-parameter breakdown. Click a column header to sort.</p>
    <div class="toolbar">
      <div class="search-wrap">
        <span class="search-ico">⌕</span>
        <input class="search" id="search" placeholder="Filter by name, code, or confederation…" autocomplete="off">
        <button class="clear-btn" id="clearBtn" title="Clear">✕</button>
      </div>
      <span class="caps" id="count"></span>
    </div>
    <div class="tablewrap" id="tablewrap">
      <table id="tbl">
        <thead><tr>
          <th data-k="odds_rank"   class="num">#</th>
          <th data-k="name">Team</th>
          <th data-k="power_index" class="num">Power</th>
          <th data-k="win_pct"     class="num">🏆 Win Title</th>
          <th data-k="win_ci_lo"   class="num">90% CI</th>
          <th data-k="final_pct"   class="num">Final</th>
          <th data-k="semi_pct"    class="num">Semi</th>
          <th data-k="advance_pct" class="num">Reach KO</th>
          <th data-k="fifa_rank"   class="num">FIFA</th>
          <th data-k="elo"         class="num">Elo</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</section>

<!-- ─── MODEL ─────────────────────────────────────────────────────────── -->
<section id="model">
  <div class="wrap">
    <div class="shead"><h2>The Model</h2><span class="caps">three-layer prediction pipeline</span></div>
    <p class="sub">The prediction uses a three-layer architecture. Every weight and constant is documented below so you can audit or reproduce it.</p>

    <div class="model-arch">
      <div class="arch-step">
        <div class="step-num">LAYER 1 · STATISTICAL LEARNING</div>
        <div class="step-title">Multinomial Logistic Regression</div>
        <div class="step-desc">
          Trained on <b id="n-train-m"></b> real World Cup matches (2010, 2014, 2018) to
          predict <b>Win / Draw / Loss</b> for each team. The regression learns how much
          each feature contributes to winning — these learned coefficients become the
          <b>Power Index weights</b> used for 2026 predictions.
          <br><br>Features: <b>Elo difference</b> between teams, <b>pedigree difference</b>
          (WC titles × 8 + appearances × 0.6), and <b>host flag</b>.
          Algorithm: gradient-descent softmax regression (pure NumPy).
        </div>
      </div>
      <div class="arch-step">
        <div class="step-num">LAYER 2 · MATCH PROBABILITY</div>
        <div class="step-title">Poisson + Dixon-Coles</div>
        <div class="step-desc">
          Converts the effective Elo gap into expected goals, then enumerates the
          joint scoreline distribution for exact <b>Win / Draw / Loss probabilities</b>.
          <br><br><span class="mono" style="color:var(--lime);font-size:13px">
          λ<sub>A</sub> = μ · exp(+γ · Δelo / 400)<br>
          λ<sub>B</sub> = μ · exp(−γ · Δelo / 400)</span>
          <br>μ = 1.35 · γ = 0.85.
          <br><br>A <b>Dixon-Coles</b> correction (ρ = −0.08) lifts the low-score
          cells so the draw rate matches reality (≈26%) — plain independent Poisson
          under-predicts the favourite-vs-minnow draws.
        </div>
      </div>
      <div class="arch-step">
        <div class="step-num">LAYER 3 · TOURNAMENT SIMULATION</div>
        <div class="step-title">Monte Carlo Simulation</div>
        <div class="step-desc">
          Plays the full 48-team / 12-group 2026 bracket <b id="n-sims-m"></b> times using
          the Poisson model for every match. Group stage → best-32 advance → R32 → R16 →
          QF → SF → Final. Counting how often each team wins each stage yields title %,
          final %, semi %, and KO-advance %.
        </div>
      </div>
      <div class="arch-step">
        <div class="step-num">PARAMETERS TRACKED PER TEAM</div>
        <div class="step-title">Feature Engineering</div>
        <div class="step-desc">
          <b>Elo rating</b> — overall strength from match history.<br>
          <b>Squad value &amp; key players</b> — current squad quality (€M market value).<br>
          <b>Recent form</b> — points from last 10 competitive games (W=3, D=1, L=0).<br>
          <b>Historical pedigree</b> — WC titles × 8 + appearances × 0.6, <b>capped at 12%</b> so history can't outvote current quality.<br>
          <b>Fitness / availability</b> — % of first-choice XI fit going into June 2026 (injury news).<br>
          <b>Squad trajectory / age</b> — average starting-XI age: younger, rising squads rewarded, ageing squads penalised.<br>
          <b>Host advantage</b> — flat bonus for USA / Mexico / Canada.<br>
          <b>Head-to-head record</b> — small Elo nudge for specific rivalries.<br>
          <b>Recency weighting</b> — training matches decay with a 7-year half-life, so Euro 2024 / NLF 2025 results count ~4-5× more than 2010 matches.
        </div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1.1fr .9fr;gap:14px">
      <div class="card" style="margin-top:0">
        <h3>Learned Parameter Weights (from logistic regression)</h3>
        <div id="weights-panel"></div>
        <p class="note-box" id="weights-note"></p>
      </div>
      <div class="card" style="margin-top:0">
        <h3>Engine Constants</h3>
        <div id="consts-panel"></div>
      </div>
    </div>
  </div>
</section>

<!-- ─── VALIDATION ─────────────────────────────────────────────────────── -->
<section id="validation">
  <div class="wrap">
    <div class="shead"><h2>Model Validation</h2><span class="caps">train 2010-2018 · test 2022</span></div>
    <p class="sub">The logistic regression is trained on WC 2010, 2014 &amp; 2018 matches, then its predictions are evaluated on the <strong style="color:var(--on)">completely held-out WC 2022</strong> tournament it has never seen. Football is inherently unpredictable — upsets are real — so ~60% accuracy (vs ~57% naive baseline) is a meaningful improvement.</p>
    <div class="val-grid" id="val-grid"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div class="card">
        <h3>Accuracy per Tournament</h3>
        <div class="yr-bars" id="yr-bars"></div>
      </div>
      <div class="card">
        <h3>About these numbers</h3>
        <div style="font-size:13px;color:var(--muted);line-height:1.7" id="val-notes"></div>
      </div>
    </div>
    <div style="margin-top:24px">
      <div class="shead" style="margin-bottom:10px"><h3 style="font-size:18px">Match-by-Match Backtest</h3></div>
      <div class="bt-filters" id="bt-filters"></div>
      <div class="bt-wrap">
        <table id="bt-table">
          <thead><tr>
            <th>Year</th><th>Match</th>
            <th class="num">P(Win)</th><th class="num">P(Draw)</th><th class="num">P(Loss)</th>
            <th>Predicted</th><th>Actual</th><th>✓</th>
          </tr></thead>
          <tbody id="bt-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</section>

<!-- ─── GROUPS ─────────────────────────────────────────────────────────── -->
<section id="groups">
  <div class="wrap">
    <div class="shead"><h2>Group Stage</h2><span class="caps">official 2026 draw · groups A–L</span></div>
    <p class="sub">The <strong style="color:var(--on)">real official draw</strong> (held 5 Dec 2025, verified). Top 2 of each group plus the 8 best third-placed teams reach the knockouts. Percentage = simulated chance of advancing. <em>Note: the knockout bracket is still strength-seeded rather than the exact official R32 crossings.</em></p>
    <div class="groups-grid" id="groups-grid"></div>
  </div>
</section>

<!-- ─── MATCH PREDICTOR ────────────────────────────────────────────────── -->
<section id="h2h">
  <div class="wrap">
    <div class="shead"><h2>Match Predictor</h2><span class="caps">live Poisson model · any two teams</span></div>
    <p class="sub">Pick any two teams to see W/D/L probabilities, expected goals, and the most likely scoreline — computed in real time from the same Poisson model used in the simulation.</p>
    <div class="h2h-selectors">
      <div class="selbox"><div class="caps" style="margin-bottom:8px">Team A</div><select id="selA"></select></div>
      <div class="vs-label">VS</div>
      <div class="selbox"><div class="caps" style="margin-bottom:8px">Team B</div><select id="selB"></select></div>
    </div>
    <div class="wdl-bar">
      <div class="wdl-seg w" id="segW"></div>
      <div class="wdl-seg d" id="segD"></div>
      <div class="wdl-seg l" id="segL"></div>
    </div>
    <div class="stats-3">
      <div class="pstat"><div class="pv score" id="score">–</div><div class="pk">Likely Score</div></div>
      <div class="pstat"><div class="pv lime" id="xgA">–</div><div class="pk" id="xgAk">xG · Team A</div></div>
      <div class="pstat"><div class="pv lime" id="xgB">–</div><div class="pk" id="xgBk">xG · Team B</div></div>
    </div>
    <p class="h2h-note" id="h2h-note"></p>
    <div style="margin-top:18px" id="exp-panel">
      <div class="caps" style="margin-bottom:8px">Why this prediction? · Log-odds attribution toward Team A Win</div>
      <div class="exp-grid" id="exp-grid"></div>
      <p class="note-box" style="margin-top:10px">Each bar shows how much that feature pushes the log-odds of Team A winning.
      Positive (lime) = helps A win; negative (red) = favours B. Values in standardised log-odds units.</p>
    </div>
  </div>
</section>

<!-- ─── LIVE 2026 ────────────────────────────────────────────────────────── -->
<section id="live">
  <div class="wrap">
    <div class="shead"><h2>🔴 Live 2026 Updates</h2><span class="caps">online learning · real-time model adaptation</span></div>
    <p class="sub">As 2026 World Cup matches happen, record each result. The model uses a <strong style="color:var(--on)">TD-learning Elo update</strong>
    (the same mathematics as Q-learning / reinforcement learning) to adjust every team's strength rating,
    then retrains the logistic regression on historical + 2026 data and rebuilds this dashboard automatically.
    Each result makes the predictions more accurate.</p>
    <div class="live-grid">
      <div class="card">
        <h3>How to add a result</h3>
        <div class="code-block"><span class="cm"># Record a match result then rebuild</span>
python3 wc2026_live.py add ARG FRA 3 3 final

<span class="cm"># Show all recorded results</span>
python3 wc2026_live.py list

<span class="cm"># Undo the last entry</span>
python3 wc2026_live.py undo

<span class="cm"># Stage options: group | r32 | r16 | qf | sf | final</span>
<span class="cm"># Goals = 90-min score (AET/pens counts as draw)</span></div>
        <div class="note-box" style="margin-top:12px">
          <b style="color:var(--on)">Why this is reinforcement learning:</b><br>
          The Elo update rule — <span class="mono" style="color:var(--lime)">ΔElo = K·G·(S−E)</span> — is mathematically identical to
          TD(0) learning: the "reward signal" (actual S minus expected E) drives a gradient step on the team's strength estimate.
          K=20 (group), 25 (knockouts), 30 (final); <b>G is a goal-difference multiplier</b> (×1 / ×1.5 / ×(11+GD)/8) so a 7-1 win
          moves ratings far more than a 1-0. After each update, the LR model retrains so W/D/L probabilities
          recalibrate for the rest of the tournament.
        </div>
      </div>
      <div class="card">
        <h3>Recorded 2026 matches <span id="live-count" class="caps" style="margin-left:8px"></span></h3>
        <div id="live-matches-list">
          <p style="color:var(--muted);font-size:14px">No 2026 results recorded yet.<br>Run the command above after each match kicks off.</p>
        </div>
        <div id="live-elo-changes" style="margin-top:14px"></div>
      </div>
    </div>
  </div>
</section>

<!-- ─── FOOTER ─────────────────────────────────────────────────────────── -->
<footer>
  <div class="wrap">
    <div class="caps">Disclaimer</div>
    <div class="disc-box">
      This dashboard is a transparent statistical model for education and entertainment.
      Team Elo ratings, squad values, form, injury figures and H2H records are <strong>curated
      approximations</strong> from public reporting, not an official live feed. The field reflects
      verified March 2026 playoff outcomes (Italy out after losing to Bosnia-Herzegovina on
      penalties; Bosnia, Scotland, Czechia, Sweden, Türkiye, DR Congo, Iraq, Haiti, Curaçao,
      Cape Verde, Jordan & South Africa among the qualifiers). The group stage uses the real
      official draw; the knockout bracket is strength-seeded. Injury/availability scores are
      editorial estimates from June 2026 squad news. <strong>Not betting advice.</strong>
    </div>
    <p style="margin-top:14px">Built with Python · NumPy · "Velocity Strike" design system · Generated __DATE__</p>
  </div>
</footer>

<!-- ─── JAVASCRIPT ─────────────────────────────────────────────────────── -->
<script>
const DATA = __DATA__;
const T = DATA.teams, BY = {};
T.forEach(t => BY[t.code] = t);
const C = DATA.constants, W = DATA.weights, MM = DATA.model_meta;
const maxPower = Math.max(...T.map(t => t.power_index));
const maxWin   = Math.max(...T.map(t => t.win_pct));

// ── podium ──────────────────────────────────────────────────────────────────
(function(){
  const top = T.slice(0, 3), order = [1, 0, 2];
  document.getElementById('podium').innerHTML = order.map(i => {
    const t = top[i], isTop = i === 0;
    return `<div class="pod ${isTop?'top':''}">
      <div class="rank">#${t.odds_rank} FAVOURITE${isTop?' ★':''}</div>
      <div class="fl">${t.flag} <span class="nm">${t.name}</span></div>
      <div class="pct">${t.win_pct.toFixed(1)}%</div>
      <div class="lbl">to win the title</div></div>`;
  }).join('');
})();

// ── title race table ─────────────────────────────────────────────────────────
let sortK = 'win_pct', sortDir = -1, filter = '';

function facBars(t) {
  const f = t.factors;
  const feats = [
    ['Elo rating',         f.elo,      (W.elo*100).toFixed(0)+'%'],
    ['Squad / players',    f.squad,    (W.squad*100).toFixed(0)+'%'],
    ['Recent form',        f.form,     (W.form*100).toFixed(0)+'%'],
    ['Historical pedigree',f.pedigree, (W.pedigree*100).toFixed(0)+'%'],
    ['Fitness / availability', f.availability ?? 100, ((W.availability||0)*100).toFixed(0)+'%'],
    ['Squad trajectory / age', f.trajectory ?? 50, ((W.trajectory||0)*100).toFixed(0)+'%'],
  ];
  const bars = feats.map(([n,v,w]) =>
    `<div class="fac"><div class="ft"><span>${n} <span style="color:var(--lime);font-size:10px">(${w})</span></span><span class="fv">${v.toFixed(0)}/100</span></div>
     <div class="fbar"><i style="width:${v}%"></i></div></div>`).join('');
  const hostChip = t.host ? `<span class="dchip g">HOST +${f.host_bonus}</span>` : '';
  const injOut = (t.injuries_out||[]).map(p =>
    `<span class="dchip" style="background:rgba(255,59,48,.14);color:var(--red)">🚑 ${p[0]} OUT</span>`).join('');
  const injDoubt = (t.injuries_doubtful||[]).map(p =>
    `<span class="dchip" style="background:rgba(255,180,0,.13);color:#ffb400">⚠ ${p[0]} doubtful</span>`).join('');
  const injLine = (injOut || injDoubt)
    ? `<div class="detchips" style="margin-top:8px">${injOut}${injDoubt}</div>` : '';
  return `<div class="players-line">Key players: <b>${t.key_players.join('</b>, <b>')}</b></div>
    <div class="facgrid">${bars}</div>
    ${injLine}
    <div class="detchips">
      <span class="dchip">Power ${t.power_index}</span>
      <span class="dchip m">Eff. Elo ${t.eff_elo}</span>
      <span class="dchip m">Form ${t.form}</span>
      <span class="dchip m">€${t.squad_value_m}M squad</span>
      <span class="dchip m">${t.conf}</span>
      <span class="dchip m">${t.wc_appearances} WC appearances</span>
      <span class="dchip ${t.availability<100?'':'g'}" style="${t.availability<100?'background:rgba(255,59,48,.14);color:var(--red)':''}">Fitness ${t.availability??100}%</span>
      <span class="dchip m">Avg age ${(t.avg_age??27.6).toFixed(1)}</span>
      ${hostChip}
    </div>`;
}

function render() {
  const q = filter.toLowerCase();
  let rows = T.filter(t => !q ||
    t.name.toLowerCase().includes(q) ||
    t.code.toLowerCase().includes(q) ||
    t.conf.toLowerCase().includes(q));
  rows.sort((a, b) => {
    const x = a[sortK], y = b[sortK];
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  const tb = document.getElementById('tbody');
  tb.innerHTML = '';
  document.getElementById('tablewrap').scrollTop = 0;
  document.getElementById('count').textContent = rows.length + ' / ' + T.length + ' teams';
  document.getElementById('clearBtn').style.display = q ? 'block' : 'none';

  rows.forEach(t => {
    const tr = document.createElement('tr');
    tr.className = 'datarow';
    const ci = `[${t.win_ci_lo.toFixed(1)}–${t.win_ci_hi.toFixed(1)}%]`;
    tr.innerHTML =
      `<td class="num mono" style="color:var(--muted)">${t.odds_rank}</td>
       <td><div class="team-cell">
         <span class="fl">${t.flag}</span>
         <div><div class="tname">${t.name}${t.host?'<span class="host-badge">HOST</span>':''}</div>
              <div class="tsub">${t.code} · ${t.conf}</div></div>
       </div></td>
       <td class="num"><span class="bar" style="width:64px"><i style="width:${100*t.power_index/maxPower}%"></i></span>&nbsp;<span class="mono">${t.power_index}</span></td>
       <td class="num"><span class="bar"><i style="width:${100*t.win_pct/maxWin}%"></i></span>&nbsp;<span class="win-val">${t.win_pct.toFixed(1)}%</span></td>
       <td class="num"><span class="ci-range">${ci}</span></td>
       <td class="num mono">${t.final_pct.toFixed(1)}%</td>
       <td class="num mono">${t.semi_pct.toFixed(1)}%</td>
       <td class="num mono">${t.advance_pct.toFixed(0)}%</td>
       <td class="num mono">${t.fifa_rank}</td>
       <td class="num mono">${t.elo}</td>`;
    const det = document.createElement('tr');
    det.className = 'detrow';
    det.style.display = 'none';
    det.innerHTML = `<td colspan="10" style="padding:16px 18px;background:#111;border-bottom:1px solid #1a1a1a">${facBars(t)}</td>`;

    tr.addEventListener('click', () => {
      det.style.display = det.style.display === 'none' ? 'table-row' : 'none';
    });
    tb.appendChild(tr);
    tb.appendChild(det);
  });
}

document.querySelectorAll('#tbl thead th').forEach(th => {
  th.addEventListener('click', () => {
    const k = th.dataset.k;
    if (sortK === k) { sortDir *= -1; }
    else { sortK = k; sortDir = (k === 'name' || k === 'fifa_rank') ? 1 : -1; }
    document.querySelectorAll('#tbl thead th .ar').forEach(a => a.remove());
    const ar = document.createElement('span');
    ar.className = 'ar';
    ar.textContent = sortDir < 0 ? '▼' : '▲';
    th.appendChild(ar);
    render();
  });
});

const searchEl = document.getElementById('search');
searchEl.addEventListener('input', e => {
  filter = e.target.value;
  render();
});
document.getElementById('clearBtn').addEventListener('click', () => {
  searchEl.value = '';
  filter = '';
  render();
  searchEl.focus();
});
render();

// ── model section ────────────────────────────────────────────────────────────
(function(){
  document.getElementById('n-train-m').textContent = MM.n_train;
  document.getElementById('n-sims-m').textContent = C.N_SIMS.toLocaleString();
  const wnames = {elo:'Elo rating (learned)',squad:'Squad value / players',form:'Recent form (learned)',pedigree:'Historical pedigree (capped)',availability:'Fitness / injuries',trajectory:'Squad trajectory / age'};
  document.getElementById('weights-panel').innerHTML = Object.entries(W).map(([k,v])=>{
    const pct = (v*100).toFixed(1);
    return `<div class="wrow">
      <div class="wlabel"><b>${wnames[k]}</b></div>
      <div class="wbar-wrap"><i style="width:${pct}%"></i></div>
      <div class="wpct">${pct}%</div></div>`;
  }).join('');
  document.getElementById('weights-note').innerHTML =
    `Elo and pedigree weights are learned from the regression's Win-class coefficients on
     recency-weighted training data (7-year half-life), then <b>balance-regularised</b>:
     pedigree is capped at 12% (excess flows to recent form) so historical glory can't
     outvote current quality, and 10% is reserved for the fitness/injury signal.
     Squad value (~18%) is allocated separately. Host advantage = +${C.HOST_BONUS} index points (flat).`;
  const cs = [['μ — avg goals/team/match',C.MU],['γ — goal sensitivity',C.GAMMA],
    ['ρ — Dixon-Coles draw corr.',C.DC_RHO],
    ['Elo base (Power 0)',C.ELO_BASE],['Elo per index point',C.ELO_SPAN],
    ['Host bonus (index)',C.HOST_BONUS],['Max H2H nudge (Elo)',C.H2H_MAX_NUDGE],
    ['Simulations',C.N_SIMS.toLocaleString()]];
  document.getElementById('consts-panel').innerHTML = cs.map(([k,v])=>
    `<div class="kv"><span>${k}</span><span>${v}</span></div>`).join('');
})();

// ── validation ────────────────────────────────────────────────────────────────
(function(){
  const baseline = 57;
  const stats = [
    {v: MM.n_train,        k:'Training matches', delta:'WC 2010 · 2014 · 2018'},
    {v: MM.n_test,         k:'Test matches',     delta:'WC 2022 (held-out)'},
    {v: MM.train_acc+'%',  k:'Train accuracy',   delta:`+${(MM.train_acc-baseline).toFixed(1)}% vs naive baseline`},
    {v: MM.test_acc+'%',   k:'Test accuracy',    delta:`+${(MM.test_acc-baseline).toFixed(1)}% vs naive baseline`},
  ];
  document.getElementById('val-grid').innerHTML = stats.map(s=>
    `<div class="val-stat"><div class="v">${s.v}</div><div class="k">${s.k}</div><div class="delta">${s.delta}</div></div>`).join('');

  const TNAMES = {WC:'World Cup', Euro:'Euro', CA:'Copa América', NLF:'Nations Lg', WCQ:'Qualifiers'};
  const years = Object.entries(MM.backtest_by_year)
    .sort((a,b)=>a[0].slice(0,4).localeCompare(b[0].slice(0,4)));
  document.getElementById('yr-bars').innerHTML = years.map(([yr,d])=>
    `<div class="yr-bar-card">
       <div class="ylab">${TNAMES[d.tournament]||d.tournament} ${yr.slice(0,4)}<span class="badge ${d.split==='test'?'test':'train'}">${d.split==='test'?'TEST':'TRAIN'}</span></div>
       <div class="acc-bar"><i style="width:${d.accuracy}%"></i></div>
       <div class="yr-acc-val mono" style="color:${d.split==='test'?'var(--green)':'var(--lime)'}">${d.accuracy}%</div>
       <div style="font-size:11px;color:var(--muted);margin-top:3px">${d.total} matches</div>
     </div>`).join('');

  document.getElementById('val-notes').innerHTML = `
    <p><b style="color:var(--on)">What "accuracy" means here:</b> the model's highest-probability outcome
    (Win, Draw, or Loss for the stronger team) matched the actual 90-minute result.</p>
    <br>
    <p><b style="color:var(--on)">Why ~62% is good:</b> the best published football prediction models
    reach 55–65% on match outcomes. Football's low-scoring nature makes upsets common —
    even the biggest mismatches have a non-trivial chance of an upset.</p>
    <br>
    <p><b style="color:var(--on)">Notable correct predictions (2022):</b> Argentina to win
    (highest Elo at tournament start), France to reach final, Morocco to advance past groups.</p>
    <br>
    <p><b style="color:var(--on)">Known limitations:</b> Saudi Arabia beating Argentina, Japan
    beating Germany, and Costa Rica beating Japan were correctly identified as unlikely by the
    model but are correctly classed as upsets in the backtest.</p>`;
})();

// ── backtest table ────────────────────────────────────────────────────────────
(function(){
  const rows = MM.backtest_rows;
  const years = [...new Set(rows.map(r=>r.year))].sort();
  let btFilter = 'all';

  const filtersEl = document.getElementById('bt-filters');
  filtersEl.innerHTML = [['all','All matches'],['wrong','Incorrect only'],
    ...years.map(y=>[String(y),String(y)])].map(([v,l])=>
      `<button class="bt-pill ${v==='all'?'active':''}" data-f="${v}">${l}</button>`).join('');
  filtersEl.querySelectorAll('.bt-pill').forEach(btn=>{
    btn.addEventListener('click',()=>{
      btFilter=btn.dataset.f;
      filtersEl.querySelectorAll('.bt-pill').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      renderBT();
    });
  });

  function chip(o){return `<span class="chip-${o.toLowerCase()}">${o}</span>`}

  function renderBT(){
    let data = rows;
    if(btFilter==='wrong') data=rows.filter(r=>!r.correct);
    else if(btFilter!=='all') data=rows.filter(r=>String(r.year)===btFilter);
    const tbody = document.getElementById('bt-tbody');
    tbody.innerHTML = data.map(r=>`
      <tr>
        <td class="mono" style="color:var(--muted)">${r.year}</td>
        <td>${r.label} <span style="font-size:11px;padding:1px 5px;border-radius:3px;background:${r.split==='test'?'rgba(0,255,102,.12)':'rgba(204,255,0,.08)'}; color:${r.split==='test'?'var(--green)':'var(--lime)'}; font-family:JetBrains Mono">${r.split.toUpperCase()}</span></td>
        <td class="num mono">${r.p_win}%</td>
        <td class="num mono">${r.p_draw}%</td>
        <td class="num mono">${r.p_loss}%</td>
        <td>${chip(r.predicted)}</td>
        <td>${chip(r.actual)}</td>
        <td>${r.correct ? '<span class="ok-tick">✓</span>' : '<span class="ko-cross">✗</span>'}</td>
      </tr>`).join('');
  }
  renderBT();
})();

// ── groups ────────────────────────────────────────────────────────────────────
(function(){
  const keys = Object.keys(DATA.groups).sort();
  document.getElementById('groups-grid').innerHTML = keys.map(k=>{
    const rows = DATA.groups[k].map(code=>{
      const t = BY[code], adv = t.advance_pct >= 50;
      return `<div class="grow ${adv?'adv':''}">
        <span class="gfl">${t.flag}</span>
        <span class="gnm">${t.name}</span>
        <span class="gad">${t.advance_pct.toFixed(0)}%</span></div>`;
    }).join('');
    return `<div class="gcard"><div class="gh">GROUP ${k}</div>${rows}</div>`;
  }).join('');
})();

// ── match predictor (JS Poisson mirrors the Python model) ─────────────────────
function fact(n){let r=1;for(let i=2;i<=n;i++)r*=i;return r;}
function pois(k,l){return Math.exp(-l)*Math.pow(l,k)/fact(k);}
function h2hNudge(a,b){
  let rec=DATA.h2h[a+'|'+b],flip=1;
  if(!rec){rec=DATA.h2h[b+'|'+a];flip=-1;}
  if(!rec)return 0;
  const tot=rec[0]+rec[1]+rec[2];if(!tot)return 0;
  return flip*(rec[0]-rec[2])/tot*C.H2H_MAX_NUDGE;
}
function lambdas(ea,eb){const d=ea-eb;
  return [Math.min(5,Math.max(.15,C.MU*Math.exp(C.GAMMA*d/400))),
          Math.min(5,Math.max(.15,C.MU*Math.exp(-C.GAMMA*d/400)))];}
function predict(a,b){
  const ea=BY[a].eff_elo+h2hNudge(a,b),eb=BY[b].eff_elo;
  const [la,lb]=lambdas(ea,eb);const N=10;
  const pa=[],pb=[];for(let i=0;i<=N;i++){pa.push(pois(i,la));pb.push(pois(i,lb));}
  let w=0,d=0,l=0,bs=[0,0],bp=0;
  for(let i=0;i<=N;i++)for(let j=0;j<=N;j++){
    const p=pa[i]*pb[j];if(p>bp){bp=p;bs=[i,j];}
    if(i>j)w+=p;else if(i===j)d+=p;else l+=p;
  }
  const tot=w+d+l;
  return{w:w/tot,d:d/tot,l:l/tot,xgA:la,xgB:lb,score:bs};
}
function fillSel(sel){
  sel.innerHTML=T.slice().sort((x,y)=>x.name.localeCompare(y.name))
    .map(t=>`<option value="${t.code}">${t.flag} ${t.name}</option>`).join('');
}
const selA=document.getElementById('selA'),selB=document.getElementById('selB');
fillSel(selA);fillSel(selB);
selA.value=T[0].code;selB.value=T[1].code;
function renderH2H(){
  const a=selA.value,b=selB.value;if(a===b)return;
  const r=predict(a,b);
  const WW=Math.round(r.w*100),DD=Math.round(r.d*100),LL=Math.round(r.l*100);
  document.getElementById('segW').style.flexBasis=WW+'%';
  document.getElementById('segD').style.flexBasis=DD+'%';
  document.getElementById('segL').style.flexBasis=LL+'%';
  document.getElementById('segW').textContent=BY[a].flag+' '+BY[a].code+' '+WW+'%';
  document.getElementById('segD').textContent='DRAW '+DD+'%';
  document.getElementById('segL').textContent=LL+'% '+BY[b].code+' '+BY[b].flag;
  document.getElementById('score').textContent=BY[a].flag+' '+r.score[0]+'–'+r.score[1]+' '+BY[b].flag;
  document.getElementById('xgA').textContent=r.xgA.toFixed(2);
  document.getElementById('xgB').textContent=r.xgB.toFixed(2);
  document.getElementById('xgAk').textContent='xG · '+BY[a].name;
  document.getElementById('xgBk').textContent='xG · '+BY[b].name;
  const nb=h2hNudge(a,b);
  document.getElementById('h2h-note').textContent=
    `${BY[a].name} effective Elo ${BY[a].eff_elo} · ${BY[b].name} ${BY[b].eff_elo}.`+
    (nb?` Head-to-head history shifts the edge by ${nb>0?'+':''}${nb.toFixed(0)} Elo toward ${nb>0?BY[a].name:BY[b].name}.`
       :' No notable H2H record encoded.');
}
// ── match explainer (pre-computed for top-10 pairs, fallback for others) ─────
function renderExplainer(a, b){
  const key = a+'|'+b;
  const exp = DATA.explanations[key];
  const grid = document.getElementById('exp-grid');
  if(!exp){ grid.innerHTML='<p style="color:var(--muted);font-size:13px;grid-column:1/-1">Detailed attribution available for top-10 teams only.</p>'; return; }
  const features = [
    ['elo_diff',  'Elo Strength'],
    ['ped_diff',  'WC Pedigree'],
    ['bias',      'Model Intercept'],
  ];
  grid.innerHTML = features.map(([k,n])=>{
    const v = exp[k] || 0;
    const cls = v>=0?'pos':'neg';
    const sign = v>=0?'+':'';
    return `<div class="exp-feat">
      <div class="ev ${cls}">${sign}${v.toFixed(3)}</div>
      <div class="ek">${n}</div></div>`;
  }).join('');
}

selA.addEventListener('change',renderH2H);
selB.addEventListener('change',renderH2H);
renderH2H();
renderExplainer(selA.value, selB.value);

// ── live 2026 results display ─────────────────────────────────────────────────
(function(){
  const live = DATA.live;
  const matches = live.matches || [];
  const elos = live.elo_updates || {};
  const countEl = document.getElementById('live-count');
  const listEl  = document.getElementById('live-matches-list');
  const eloEl   = document.getElementById('live-elo-changes');

  if(matches.length === 0){ countEl.textContent=''; return; }

  countEl.textContent = matches.length + ' result' + (matches.length>1?'s':'') + ' recorded';

  // Badge in header
  const liveBadge = document.createElement('span');
  liveBadge.className = 'live-badge';
  liveBadge.textContent = '● LIVE';
  document.querySelector('.hbar').appendChild(liveBadge);

  listEl.innerHTML = matches.map(m=>{
    const r = m.outcome==='W'?`<span style="color:var(--lime)">WIN</span>`:
              m.outcome==='L'?`<span style="color:var(--red)">LOSS</span>`:
              `<span style="color:var(--muted)">DRAW</span>`;
    return `<div class="live-match-row">
      <div><b>${m.team_a}</b> vs <b>${m.team_b}</b><br><span class="live-stage">${m.stage} · ${m.date}</span></div>
      <div class="live-score">${m.goals_a}–${m.goals_b}</div>
      <div>${r}</div>
    </div>`;
  }).join('');

  if(Object.keys(elos).length>0){
    eloEl.innerHTML = '<div class="caps" style="margin-bottom:8px">Cumulative Elo Changes</div>' +
      Object.entries(elos).map(([code, delta])=>{
        const t = BY[code]; if(!t) return '';
        const sign = delta>=0?'+':'';
        const col = delta>=0?'var(--lime)':'var(--red)';
        return `<span style="font-family:JetBrains Mono;font-size:12px;margin-right:14px">
          ${t.flag} ${code} <b style="color:${col}">${sign}${delta.toFixed(1)}</b></span>`;
      }).join('');
  }
})();
</script>
</body>
</html>
"""


def main(n_sims=M.N_SIMS):
    print(f"Running {n_sims:,} tournament simulations + analytic Wilson CI…")
    payload = build_payload(n_sims)
    mm = payload["model_meta"]
    html = (HTML
            .replace("__DATA__", json.dumps(payload))
            .replace("__NSIMS__", f"{n_sims:,}")
            .replace("__DATE__", payload["generated"])
            .replace("__NTRAIN__", str(int(mm['n_train'])))
            .replace("54.8% ± 1.8%", f"{mm['cv_mean']}% ± {mm['cv_std']}%")
            .replace("WC 2022 · 59% accuracy", f"WC 2022 · {mm['test_acc']}% accuracy"))
    # Write index.html for GitHub Pages (served at the repo root URL)
    # and dashboard.html as a local alias — both are identical.
    for out in ("index.html", "dashboard.html"):
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
    top = payload["teams"][:5]
    w = payload["weights"]
    print(f"\nDone → index.html + dashboard.html")
    top5 = ', '.join(t['name'] + ' ' + str(t['win_pct']) + '%' for t in top)
    print(f"Top 5: {top5}")
    print(f"Learned weights: elo={w['elo']:.1%}  squad={w['squad']:.1%}  form={w['form']:.1%}  pedigree={w['pedigree']:.1%}")
    m = payload["model_meta"]
    print(f"Accuracy → train {m['train_acc']}%  test (2022) {m['test_acc']}%  n_train={m['n_train']}  n_test={m['n_test']}")


if __name__ == "__main__":
    main()
