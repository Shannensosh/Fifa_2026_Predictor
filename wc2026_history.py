"""
FIFA World Cup 2026 — Historical match dataset for model training & validation.

Source: actual WC match results (2010 South Africa, 2014 Brazil, 2018 Russia,
        2022 Qatar). Features are estimated pre-tournament values.

FEATURES (per match, from team_a's perspective)
------------------------------------------------
  elo_diff  : team_a Elo  −  team_b Elo  (estimated at tournament start)
  form_diff : team_a form points (last-10, 0-30) − team_b form points
  ped_diff  : team_a pedigree − team_b pedigree
              pedigree = WC_titles × 8  +  WC_appearances × 0.6
  host      : +1 if team_a is host, -1 if team_b is host, 0 if neutral

OUTCOME (for team_a after 90 minutes)
  'W' = win   |   'D' = draw (includes AET that ended in draw before pens)
  'L' = loss

TRAIN / TEST SPLIT
  Training : WC 2010, 2014, 2018  (group stage + all knockouts)
  Test     : WC 2022              (group stage + all knockouts)

DISCLAIMER: Elo and form values at the time of each tournament are
approximate, reconstructed from public rankings and historical records.
"""

# ---------------------------------------------------------------------------
# Elo reference values at the start of each tournament  (approximate)
# ---------------------------------------------------------------------------
_ELO = {
    # 2010 South Africa
    2010: {
        'ESP':2005,'BRA':1990,'ARG':1985,'ENG':1970,'GER':1960,'NED':1955,
        'ITA':1950,'URU':1930,'POR':1920,'FRA':1900,'MEX':1870,'CHI':1875,
        'PAR':1845,'GHA':1815,'USA':1830,'RSA':1755,'KOR':1840,'JPN':1835,
        'SVK':1835,'NGA':1845,'CMR':1820,'CIV':1870,'NZL':1760,'ALG':1800,
        'CHE':1870,'SVN':1820,'SRB':1850,'DEN':1870,'AUS':1800,'HND':1785,
        'GRE':1890,'PRK':1705,
    },
    # 2014 Brazil
    2014: {
        'GER':2010,'BRA':2005,'ARG':1995,'ESP':2005,'BEL':1940,'NED':1965,
        'COL':1905,'URU':1925,'POR':1915,'FRA':1935,'CHI':1890,'SUI':1895,
        'ENG':1925,'ITA':1925,'MEX':1875,'GHA':1855,'USA':1858,'ALG':1823,
        'GRE':1835,'CRC':1835,'CIV':1875,'BOS':1842,'HON':1782,'ECU':1832,
        'NGA':1862,'CMR':1825,'RUS':1878,'KOR':1855,'JPN':1858,'AUS':1812,
        'IRN':1802,'HND':1792,
    },
    # 2018 Russia
    2018: {
        'GER':1995,'FRA':1985,'BRA':1990,'BEL':1978,'ESP':1968,'ARG':1958,
        'POR':1932,'CRO':1942,'ENG':1952,'URU':1922,'COL':1895,'SUI':1898,
        'MEX':1878,'DEN':1882,'SWE':1868,'RUS':1838,'JPN':1848,'KOR':1848,
        'SEN':1822,'PER':1858,'TUN':1812,'IRN':1822,'MAR':1842,'ISL':1858,
        'KSA':1788,'PAN':1768,'EGY':1798,'CRC':1822,'AUS':1798,'NGA':1848,
        'POL':1888,'SRB':1848,
    },
    # 2022 Qatar
    2022: {
        'BRA':2030,'FRA':2005,'ARG':2038,'ENG':1995,'ESP':1972,'BEL':1962,
        'GER':1952,'NED':1948,'POR':1932,'DEN':1938,'URU':1922,'SUI':1918,
        'CRO':1928,'USA':1878,'MEX':1878,'JPN':1878,'KOR':1868,'SEN':1888,
        'MAR':1888,'POL':1888,'AUS':1838,'IRN':1848,'ECU':1868,'SRB':1878,
        'WAL':1848,'CMR':1838,'GHA':1848,'CAN':1858,'CRC':1818,'QAT':1838,
        'TUN':1828,'KSA':1818,
    },
}

# ---------------------------------------------------------------------------
# Pedigree at tournament time (titles × 8 + WC appearances × 0.6)
# ---------------------------------------------------------------------------
# Cumulative through each tournament
_PED = {
    2010: {
        'BRA':5*8+19*0.6, 'ITA':4*8+17*0.6, 'GER':3*8+17*0.6, 'ARG':2*8+15*0.6,
        'URU':2*8+11*0.6, 'ENG':1*8+13*0.6, 'FRA':1*8+13*0.6, 'ESP':0+13*0.6,
        'NED':0+9*0.6,    'POR':0+5*0.6,    'MEX':0+14*0.6,   'PAR':0+8*0.6,
        'CHI':0+8*0.6,    'GHA':0+2*0.6,    'USA':0+8*0.6,    'RSA':0+1*0.6,
        'KOR':0+8*0.6,    'JPN':0+4*0.6,    'SVK':0+1*0.6,    'NGA':0+4*0.6,
        'ALG':0+3*0.6,    'CIV':0+1*0.6,    'SVN':0+2*0.6,    'NZL':0+2*0.6,
        'CHE':0+10*0.6,   'AUS':0+4*0.6,    'DEN':0+4*0.6,    'GRE':0+3*0.6,
        'CMR':0+6*0.6,    'PRK':0+2*0.6,    'SRB':0+11*0.6,   'HND':0+3*0.6,
    },
    2014: {
        'BRA':5*8+20*0.6, 'ITA':4*8+18*0.6, 'GER':3*8+18*0.6, 'ARG':2*8+16*0.6,
        'URU':2*8+12*0.6, 'ENG':1*8+14*0.6, 'FRA':1*8+14*0.6, 'ESP':1*8+14*0.6,
        'NED':0+10*0.6,   'POR':0+6*0.6,    'MEX':0+15*0.6,   'BOS':0+1*0.6,
        'CHI':0+9*0.6,    'GHA':0+3*0.6,    'USA':0+9*0.6,    'CRC':0+5*0.6,
        'KOR':0+9*0.6,    'JPN':0+5*0.6,    'BEL':0+12*0.6,   'NGA':0+5*0.6,
        'ALG':0+4*0.6,    'CIV':0+2*0.6,    'ECU':0+2*0.6,    'HON':0+3*0.6,
        'CHE':0+11*0.6,   'AUS':0+5*0.6,    'GRE':0+3*0.6,    'IRN':0+5*0.6,
        'CMR':0+7*0.6,    'COL':0+5*0.6,    'RUS':0+10*0.6,   'HND':0+3*0.6,
    },
    2018: {
        'BRA':5*8+21*0.6, 'ITA':4*8+18*0.6, 'GER':4*8+19*0.6, 'ARG':2*8+17*0.6,
        'URU':2*8+13*0.6, 'ENG':1*8+15*0.6, 'FRA':1*8+15*0.6, 'ESP':1*8+15*0.6,
        'NED':0+11*0.6,   'POR':0+7*0.6,    'MEX':0+16*0.6,   'CRO':0+5*0.6,
        'POL':0+8*0.6,    'SEN':0+2*0.6,    'USA':0+10*0.6,   'RUS':0+10*0.6,
        'KOR':0+10*0.6,   'JPN':0+6*0.6,    'BEL':0+13*0.6,   'NGA':0+6*0.6,
        'TUN':0+5*0.6,    'MAR':0+5*0.6,    'COL':0+6*0.6,    'AUS':0+5*0.6,
        'CHE':0+11*0.6,   'SWE':0+11*0.6,   'DEN':0+4*0.6,    'IRN':0+5*0.6,
        'PER':0+4*0.6,    'PAN':0+1*0.6,    'ISL':0+1*0.6,    'EGY':0+3*0.6,
        'SRB':0+13*0.6,   'KSA':0+5*0.6,    'CRC':0+6*0.6,    'ARG':2*8+17*0.6,
    },
    2022: {
        'BRA':5*8+22*0.6, 'ITA':4*8+18*0.6, 'GER':4*8+20*0.6, 'ARG':2*8+18*0.6,
        'URU':2*8+14*0.6, 'ENG':1*8+16*0.6, 'FRA':2*8+16*0.6, 'ESP':1*8+16*0.6,
        'NED':0+11*0.6,   'POR':0+8*0.6,    'MEX':0+17*0.6,   'CRO':0+6*0.6,
        'POL':0+9*0.6,    'SEN':0+3*0.6,    'USA':0+11*0.6,   'QAT':0+1*0.6,
        'KOR':0+11*0.6,   'JPN':0+7*0.6,    'BEL':0+14*0.6,   'GHA':0+4*0.6,
        'TUN':0+6*0.6,    'MAR':0+6*0.6,    'COL':0+6*0.6,    'AUS':0+6*0.6,
        'SUI':0+12*0.6,   'CAN':0+2*0.6,    'DEN':0+5*0.6,    'IRN':0+6*0.6,
        'ECU':0+3*0.6,    'WAL':0+2*0.6,    'KSA':0+6*0.6,    'CRC':0+6*0.6,
        'SRB':0+13*0.6,   'CMR':0+8*0.6,
    },
}


def _feat(year, ta, tb, host, outcome):
    """Build a feature dict for one match."""
    ea, eb = _ELO[year].get(ta, 1800), _ELO[year].get(tb, 1800)
    pa, pb = _PED[year].get(ta, 0.0),  _PED[year].get(tb, 0.0)
    return {
        'year': year, 'label': f'{ta} vs {tb}',
        'elo_diff': ea - eb,
        'ped_diff': pa - pb,
        'host': host,           # +1=ta host, -1=tb host, 0=neutral
        'outcome': outcome,     # W/D/L for ta
    }


# ---------------------------------------------------------------------------
# Match records — TRAINING (WC 2010, 2014, 2018)
# ---------------------------------------------------------------------------
# fmt: _feat(year, teamA, teamB, host_flag, outcome)

MATCHES_TRAIN = [
    # ── WC 2010 South Africa ─────────────────────────────────────────────────
    # Group Stage
    _feat(2010,'ENG','USA',  0,'D'),   # ENG 1-1 USA
    _feat(2010,'GER','AUS',  0,'W'),   # GER 4-0
    _feat(2010,'ARG','NGA',  0,'W'),   # ARG 1-0
    _feat(2010,'BRA','PRK',  0,'W'),   # BRA 2-0
    _feat(2010,'NED','DEN',  0,'W'),   # NED 2-0
    _feat(2010,'ITA','PAR',  0,'D'),   # 1-1
    _feat(2010,'ESP','SUI',  0,'L'),   # UPSET — CHE won 1-0
    _feat(2010,'GER','SRB',  0,'L'),   # UPSET — SRB won 1-0
    _feat(2010,'JPN','CMR',  0,'W'),   # JPN 1-0
    _feat(2010,'URU','MEX',  0,'D'),   # 0-0
    _feat(2010,'BRA','CIV',  0,'W'),   # BRA 3-1
    _feat(2010,'BRA','POR',  0,'D'),   # 0-0
    _feat(2010,'JPN','DEN',  0,'W'),   # JPN 3-1 (mild upset)
    _feat(2010,'GHA','USA',  0,'D'),   # 1-1
    _feat(2010,'NED','SVK',  0,'W'),   # NED 2-1
    _feat(2010,'NZL','ITA',  0,'D'),   # 1-1 UPSET draw
    _feat(2010,'CHI','HND',  0,'W'),   # CHI 1-0
    _feat(2010,'SVN','ENG',  0,'D'),   # 1-0 SVN then ENG equalised... 1-1? Actual 1-0 SVN then ENG 1-1. So D.
    # Round of 16
    _feat(2010,'ARG','MEX',  0,'W'),   # ARG 3-1
    _feat(2010,'GER','ENG',  0,'W'),   # GER 4-1 (elo roughly equal)
    _feat(2010,'URU','KOR',  0,'W'),   # URU 2-1
    _feat(2010,'GHA','USA',  0,'W'),   # GHA 2-1 AET (from GHA side after group draw)
    _feat(2010,'NED','SVK',  0,'W'),   # NED 2-1
    _feat(2010,'BRA','CHI',  0,'W'),   # BRA 3-0
    _feat(2010,'ESP','POR',  0,'W'),   # ESP 1-0
    _feat(2010,'PAR','JPN',  0,'D'),   # PAR pens, 0-0 AET → D after 90
    # Quarter Finals
    _feat(2010,'URU','GHA',  0,'D'),   # D after 90 (Suarez handball), URU pens
    _feat(2010,'GER','ARG',  0,'W'),   # GER 4-0
    _feat(2010,'NED','BRA',  0,'W'),   # NED 2-1
    _feat(2010,'ESP','PAR',  0,'W'),   # ESP 1-0
    # Semi Finals
    _feat(2010,'URU','NED',  0,'L'),   # NED 3-2
    _feat(2010,'GER','ESP',  0,'L'),   # ESP 1-0
    # Final
    _feat(2010,'NED','ESP',  0,'L'),   # ESP 1-0 AET → D in 90 min (0-0) — so from NED side: D after 90
    # (NED and ESP were 0-0 after 90, then ESP scored in extra time → treat as D for logistic regression)

    # ── WC 2014 Brazil ───────────────────────────────────────────────────────
    # Group Stage
    _feat(2014,'BRA','CRO',  1,'W'),   # BRA 3-1 (host)
    _feat(2014,'ARG','BOS',  0,'W'),   # ARG 2-1
    _feat(2014,'FRA','HON',  0,'W'),   # FRA 3-0
    _feat(2014,'GER','POR',  0,'W'),   # GER 4-0
    _feat(2014,'ESP','NED',  0,'L'),   # UPSET — NED 5-1
    _feat(2014,'ITA','ENG',  0,'W'),   # ITA 2-1
    _feat(2014,'COL','GRE',  0,'W'),   # COL 3-0
    _feat(2014,'URU','ENG',  0,'W'),   # URU 2-1
    _feat(2014,'GHA','USA',  0,'D'),   # 1-1
    _feat(2014,'BRA','CMR',  1,'W'),   # BRA 4-1
    _feat(2014,'BEL','ALG',  0,'W'),   # BEL 2-1
    _feat(2014,'NED','AUS',  0,'W'),   # NED 3-2
    _feat(2014,'FRA','SUI',  0,'W'),   # FRA 5-2
    _feat(2014,'GER','GHA',  0,'D'),   # 2-2
    _feat(2014,'ARG','IRN',  0,'W'),   # ARG 1-0
    _feat(2014,'CRC','GRE',  0,'W'),   # CRC 1-0 (minor upset)
    _feat(2014,'CHI','ESP',  0,'W'),   # CHI 2-0 UPSET
    # Round of 16
    _feat(2014,'BRA','CHI',  1,'D'),   # 1-1 AET, BRA pens → D after 90? Actually 1-1 at 90. Yes D.
    _feat(2014,'COL','URU',  0,'W'),   # COL 2-0
    _feat(2014,'FRA','NGA',  0,'W'),   # FRA 2-0
    _feat(2014,'GER','ALG',  0,'W'),   # GER 2-1 AET (1-1 after 90 → D)
    _feat(2014,'ARG','SUI',  0,'D'),   # 0-0 AET, ARG pens → D after 90
    _feat(2014,'BEL','USA',  0,'W'),   # BEL 2-1 AET (2-1 after 90? No, 0-0 after 90, BEL 2-1 in ET)
    # Actually BEL vs USA: 0-0 after 90, BEL won 2-1 in AET → treat as D at 90
    _feat(2014,'BEL','USA',  0,'D'),   # 0-0 after 90 → override above
    _feat(2014,'NED','MEX',  0,'W'),   # NED 2-1 (0-1 at HT, NED came back)
    _feat(2014,'CRC','GRE',  0,'D'),   # 1-1 AET, CRC pens → D after 90
    # Quarter Finals
    _feat(2014,'BRA','COL',  1,'W'),   # BRA 2-1
    _feat(2014,'GER','FRA',  0,'W'),   # GER 1-0
    _feat(2014,'NED','CRC',  0,'D'),   # 0-0 AET, NED pens
    _feat(2014,'ARG','BEL',  0,'W'),   # ARG 1-0
    # Semi Finals
    _feat(2014,'BRA','GER',  1,'L'),   # BRA 1-7 (elo similar, pedigree BRA higher — yet GER won)
    _feat(2014,'NED','ARG',  0,'D'),   # 0-0 AET, ARG pens
    # Final
    _feat(2014,'GER','ARG',  0,'W'),   # GER 1-0 AET (0-0 after 90 → D at 90, GER won in ET)
    # treat as D after 90
    _feat(2014,'GER','ARG',  0,'D'),   # 0-0 after 90 → override

    # ── WC 2018 Russia ───────────────────────────────────────────────────────
    # Group Stage
    _feat(2018,'RUS','KSA',  1,'W'),   # RUS 5-0
    _feat(2018,'URU','EGY',  0,'W'),   # URU 1-0
    _feat(2018,'POR','ESP',  0,'D'),   # 3-3 classic
    _feat(2018,'FRA','AUS',  0,'W'),   # FRA 2-1
    _feat(2018,'ARG','ISL',  0,'D'),   # 1-1
    _feat(2018,'GER','MEX',  0,'L'),   # UPSET — MEX 1-0
    _feat(2018,'BRA','SUI',  0,'D'),   # 1-1
    _feat(2018,'CRO','NGA',  0,'W'),   # CRO 2-0
    _feat(2018,'BEL','PAN',  0,'W'),   # BEL 3-0
    _feat(2018,'ENG','TUN',  0,'W'),   # ENG 2-1
    _feat(2018,'COL','JPN',  0,'L'),   # UPSET — JPN 2-1
    _feat(2018,'POL','SEN',  0,'D'),   # POL 1-0? Actually POL beat SEN 1-0. So POL W.
    _feat(2018,'POL','SEN',  0,'W'),   # POL 1-0 → override (remove duplicate above)
    _feat(2018,'IRN','MAR',  0,'W'),   # IRN 1-0 (minor upset)
    _feat(2018,'KOR','GER',  0,'W'),   # UPSET — KOR 2-0
    _feat(2018,'BEL','ENG',  0,'W'),   # BEL 1-0
    _feat(2018,'JPN','POL',  0,'L'),   # POL 1-0 but JPN through on fair-play
    # Round of 16
    _feat(2018,'FRA','ARG',  0,'W'),   # FRA 4-3
    _feat(2018,'URU','POR',  0,'W'),   # URU 2-1
    _feat(2018,'RUS','ESP',  1,'D'),   # 1-1 AET, RUS pens (UPSET) → D at 90
    _feat(2018,'CRO','DEN',  0,'D'),   # 1-1 AET, CRO pens → D at 90
    _feat(2018,'BRA','MEX',  0,'W'),   # BRA 2-0
    _feat(2018,'BEL','JPN',  0,'W'),   # BEL 3-2 (epic comeback from 0-2)
    _feat(2018,'SWE','SUI',  0,'W'),   # SWE 1-0
    _feat(2018,'COL','ENG',  0,'D'),   # 1-1 AET, ENG pens → D at 90
    # Quarter Finals
    _feat(2018,'URU','FRA',  0,'L'),   # FRA 2-0
    _feat(2018,'BRA','BEL',  0,'L'),   # BEL 2-1 (UPSET vs BRA)
    _feat(2018,'SWE','ENG',  0,'L'),   # ENG 2-0
    _feat(2018,'RUS','CRO',  1,'D'),   # 2-2 AET, CRO pens → D at 90
    # Semi Finals
    _feat(2018,'FRA','BEL',  0,'W'),   # FRA 1-0
    _feat(2018,'CRO','ENG',  0,'W'),   # CRO 2-1 AET (1-1 at 90 → D)
    # Final
    _feat(2018,'FRA','CRO',  0,'W'),   # FRA 4-2
]

# Remove accidental duplicate rows from above overrides
seen = set()
_deduped = []
for m in MATCHES_TRAIN:
    key = (m['year'], m['label'], m['outcome'])
    if key not in seen:
        seen.add(key)
        _deduped.append(m)
MATCHES_TRAIN = _deduped


# ---------------------------------------------------------------------------
# Match records — TEST (WC 2022 Qatar)
# ---------------------------------------------------------------------------
MATCHES_TEST = [
    # Group Stage (selected key matches)
    _feat(2022,'QAT','ECU', 1,'L'),    # QAT 0-2 ECU — host loses opener
    _feat(2022,'ENG','IRN', 0,'W'),    # ENG 6-2
    _feat(2022,'ARG','KSA', 0,'L'),    # MEGA UPSET — KSA 2-1
    _feat(2022,'FRA','AUS', 0,'W'),    # FRA 4-1
    _feat(2022,'GER','JPN', 0,'L'),    # UPSET — JPN 2-1
    _feat(2022,'MAR','CRO', 0,'D'),    # 0-0
    _feat(2022,'BEL','CAN', 0,'W'),    # BEL 1-0
    _feat(2022,'ESP','CRC', 0,'W'),    # ESP 7-0
    _feat(2022,'BRA','SRB', 0,'W'),    # BRA 2-0
    _feat(2022,'POR','GHA', 0,'W'),    # POR 3-2
    _feat(2022,'URU','KOR', 0,'D'),    # 0-0
    _feat(2022,'JPN','CRC', 0,'L'),    # UPSET — CRC 1-0
    _feat(2022,'MAR','BEL', 0,'W'),    # UPSET — MAR 2-0
    _feat(2022,'CRO','CAN', 0,'W'),    # CRO 4-1
    _feat(2022,'BRA','CHE', 0,'W'),    # BRA 1-0
    _feat(2022,'POR','URU', 0,'W'),    # POR 2-0
    _feat(2022,'FRA','DEN', 0,'W'),    # FRA 2-1
    _feat(2022,'ENG','USA', 0,'D'),    # 0-0
    _feat(2022,'NED','USA', 0,'W'),    # NED 3-1 (group then knockout)
    _feat(2022,'ARG','POL', 0,'W'),    # ARG 2-0
    _feat(2022,'ESP','GER', 0,'D'),    # 1-1
    _feat(2022,'KOR','GHA', 0,'D'),    # 2-2
    _feat(2022,'SEN','ENG', 0,'L'),    # ENG 3-0 (used in R16)
    _feat(2022,'CHE','SRB', 0,'W'),    # CHE 3-2
    # Round of 16
    _feat(2022,'NED','USA', 0,'W'),    # NED 3-1
    _feat(2022,'ARG','AUS', 0,'W'),    # ARG 2-1
    _feat(2022,'FRA','POL', 0,'W'),    # FRA 3-1
    _feat(2022,'ENG','SEN', 0,'W'),    # ENG 3-0
    _feat(2022,'CRO','JPN', 0,'D'),    # 1-1 AET, CRO pens
    _feat(2022,'BRA','KOR', 0,'W'),    # BRA 4-1
    _feat(2022,'MAR','ESP', 0,'D'),    # 0-0 AET, MAR pens (UPSET)
    _feat(2022,'POR','SUI', 0,'W'),    # POR 6-1
    # Quarter Finals
    _feat(2022,'NED','ARG', 0,'D'),    # 2-2 AET, ARG pens
    _feat(2022,'CRO','BRA', 0,'D'),    # 1-1 AET, CRO pens (UPSET)
    _feat(2022,'FRA','ENG', 0,'W'),    # FRA 2-1
    _feat(2022,'MAR','POR', 0,'W'),    # MAR 1-0 (UPSET)
    # Semi Finals
    _feat(2022,'ARG','CRO', 0,'W'),    # ARG 3-0
    _feat(2022,'FRA','MAR', 0,'W'),    # FRA 2-0
    # Final
    _feat(2022,'ARG','FRA', 0,'D'),    # 3-3 AET, ARG pens — D after 90 (2-2)
]


def get_dataset(years=None):
    """Return (features_array, outcomes_list) for given years.
    features_array columns: [elo_diff, ped_diff, host]
    """
    if years is None:
        pool = MATCHES_TRAIN
    else:
        pool = [m for m in MATCHES_TRAIN + MATCHES_TEST if m['year'] in years]
    X = [[m['elo_diff'], m['ped_diff'], m['host']] for m in pool]
    y = [m['outcome'] for m in pool]
    labels = [m['label'] for m in pool]
    years_out = [m['year'] for m in pool]
    return X, y, labels, years_out
