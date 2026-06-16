"""
wc2026_fetch.py — auto-ingest real WC 2026 results from TheSportsDB.

Pulls the FIFA World Cup 2026 results (free, keyless TheSportsDB endpoint),
maps team names to this project's internal codes, and records any NEW finished
GROUP-stage match via the live TD-learning module (wc2026_live.add_result),
which updates Elo ratings and adds the match to training.

Designed to be run by the daily GitHub Action *before* build_dashboard.py.
Safe to run repeatedly: it dedupes on the unordered team pair, so already-
recorded matches are skipped. Network/parse failures are non-fatal (exit 0)
so the daily rebuild always proceeds.

Knockout matches are intentionally skipped: the model records the 90-minute
score (extra-time/penalties count as a draw), and the feed's final score may
include extra time. Record knockout results manually with wc2026_live.py.
"""

import os
import sys
import json
import unicodedata
import datetime
import urllib.request

DIR        = os.path.dirname(os.path.abspath(__file__))
LEAGUE_ID  = "4429"   # FIFA World Cup on TheSportsDB
BASE       = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d={d}&l=" + LEAGUE_ID
TOURNAMENT_START = datetime.date(2026, 6, 11)   # scan every day from kickoff…
MAX_DAYS   = 45       # …up to this many days (whole tournament window)
GROUP_ROUNDS = {"1", "2", "3"}   # TheSportsDB intRound for the 3 group matchdays

# TheSportsDB team name (normalised) -> internal code
NAME_TO_CODE = {
    "mexico": "MEX", "southafrica": "RSA", "southkorea": "KOR",
    "korearepublic": "KOR", "czechrepublic": "CZE", "czechia": "CZE",
    "canada": "CAN", "bosniaherzegovina": "BIH", "bosniaandherzegovina": "BIH",
    "qatar": "QAT", "switzerland": "SUI", "brazil": "BRA", "morocco": "MAR",
    "haiti": "HAI", "scotland": "SCO", "usa": "USA", "unitedstates": "USA",
    "paraguay": "PAR", "australia": "AUS", "turkey": "TUR", "turkiye": "TUR",
    "germany": "GER", "curacao": "CUW", "ivorycoast": "CIV", "cotedivoire": "CIV",
    "ecuador": "ECU", "netherlands": "NED", "japan": "JPN", "sweden": "SWE",
    "tunisia": "TUN", "belgium": "BEL", "egypt": "EGY", "iran": "IRN",
    "newzealand": "NZL", "spain": "SPA", "capeverde": "CPV", "caboverde": "CPV",
    "saudiarabia": "KSA", "uruguay": "URU", "france": "FRA", "senegal": "SEN",
    "iraq": "IRQ", "norway": "NOR", "argentina": "ARG", "algeria": "ALG",
    "austria": "AUT", "jordan": "JOR", "portugal": "POR", "drcongo": "COD",
    "congodr": "COD", "democraticrepublicofthecongo": "COD",
    "uzbekistan": "UZB", "colombia": "COL", "england": "ENG", "croatia": "CRO",
    "ghana": "GHA", "panama": "PAN",
}


def _norm(name):
    """Lowercase, strip accents and non-alphanumerics for robust matching."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _fetch_day(date_str):
    url = BASE.format(d=date_str)
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026-predictor"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode())
    return data.get("events") or []


def main():
    try:
        from wc2026_live import add_result, _load
    except Exception as e:
        print(f"[fetch] cannot import live module: {e}")
        return 0

    # Existing recorded pairs (unordered) — for dedupe
    existing = _load().get("matches", [])
    seen_pairs = {frozenset((m["team_a"], m["team_b"])) for m in existing}

    # Scan every day from the tournament start through tomorrow (UTC).
    # Iterating from kickoff (not a rolling window) guarantees completeness
    # even if the feed assigns matches to slightly different dates.
    today = datetime.datetime.now(datetime.timezone.utc).date()
    end = today + datetime.timedelta(days=1)
    n_days = min(MAX_DAYS, (end - TOURNAMENT_START).days + 1)
    dates = [(TOURNAMENT_START + datetime.timedelta(days=i)).isoformat()
             for i in range(max(0, n_days))]

    candidates = []
    for d in dates:
        try:
            events = _fetch_day(d)
        except Exception as e:
            print(f"[fetch] {d}: request failed ({e}) — skipping")
            continue
        for e in events:
            status = (e.get("strStatus") or "").lower()
            if status not in ("ft", "match finished", "finished", "aet", "pen"):
                continue
            rnd = str(e.get("intRound") or "")
            if rnd not in GROUP_ROUNDS:        # group stage only (safe 90-min)
                if e.get("strHomeTeam"):
                    print(f"[fetch] skip non-group: {e.get('strHomeTeam')} v "
                          f"{e.get('strAwayTeam')} (round {rnd})")
                continue
            ca = NAME_TO_CODE.get(_norm(e.get("strHomeTeam")))
            cb = NAME_TO_CODE.get(_norm(e.get("strAwayTeam")))
            ha, ab = e.get("intHomeScore"), e.get("intAwayScore")
            if not ca or not cb or ha is None or ab is None:
                if e.get("strHomeTeam"):
                    print(f"[fetch] unmapped/no-score: {e.get('strHomeTeam')} "
                          f"v {e.get('strAwayTeam')}")
                continue
            candidates.append((e.get("dateEvent") or "", ca, cb, int(ha), int(ab)))

    # Record new pairs only, in chronological order
    candidates.sort(key=lambda x: x[0])
    added = 0
    for date_str, ca, cb, ha, ab in candidates:
        if frozenset((ca, cb)) in seen_pairs:
            continue
        add_result(ca, cb, ha, ab, "group", date_str or None)
        seen_pairs.add(frozenset((ca, cb)))
        added += 1

    print(f"[fetch] done — {added} new group result(s) recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
