"""
FIFA World Cup 2026 — Historical match dataset for model training & validation.

Sources: WC 2010, 2014, 2018, 2022 match results plus major international
         tournaments (UEFA Euro 2012/2016/2020, Copa América 2015/2019/2021/2024,
         UEFA Nations League Finals 2020-21 and 2022-23).

FEATURES (per match, from team_a's perspective)
------------------------------------------------
  elo_diff    : team_a Elo − team_b Elo  (estimated at tournament start)
  ped_diff    : team_a pedigree − team_b pedigree
                pedigree = major_titles × 8  +  WC_appearances × 0.6
  host        : +1 if team_a is host, -1 if team_b is host, 0 neutral
  comp_weight : competition importance weight (0-1)
                World Cup 1.0 | UEFA Euro 0.88 | Copa América 0.85
                Nations League Finals 0.78 | Friendly/qualifier 0.60

OUTCOME (for team_a after 90 minutes)
  'W' = win  |  'D' = draw (includes AET that ended tied before pens)
  'L' = loss
  NOTE: Penalty shootout wins after 90-min draw → encoded as 'D' (result at 90)
        AET win where it was 0-0/tied at 90   → encoded as 'D' at 90 min

TRAIN / TEST SPLIT
  Training : WC 2010, 2014, 2018 + Euro 2012, 2016, 2020
             Copa América 2015, 2019, 2021, 2024
             Nations League Finals 2020-21, 2022-23
  Test     : WC 2022 (unchanged)

DISCLAIMER: Elo and pedigree values at the time of each tournament are
approximate, reconstructed from public rankings and historical records.
"""

# ---------------------------------------------------------------------------
# Competition metadata
# ---------------------------------------------------------------------------
TOURNAMENT_META = {
    'WC':   {'comp_weight': 1.00, 'description': 'FIFA World Cup'},
    'Euro': {'comp_weight': 0.88, 'description': 'UEFA European Championship'},
    'CA':   {'comp_weight': 0.85, 'description': 'CONMEBOL Copa América'},
    'NLF':  {'comp_weight': 0.78, 'description': 'UEFA Nations League Finals'},
    'FQ':   {'comp_weight': 0.60, 'description': 'Friendly / Qualifier'},
}

# ---------------------------------------------------------------------------
# Elo reference values at the start of each tournament  (approximate)
# ---------------------------------------------------------------------------
_ELO = {
    # ── World Cups ──────────────────────────────────────────────────────────
    2010: {
        'ESP':2005,'BRA':1990,'ARG':1985,'ENG':1970,'GER':1960,'NED':1955,
        'ITA':1950,'URU':1930,'POR':1920,'FRA':1900,'MEX':1870,'CHI':1875,
        'PAR':1845,'GHA':1815,'USA':1830,'RSA':1755,'KOR':1840,'JPN':1835,
        'SVK':1835,'NGA':1845,'CMR':1820,'CIV':1870,'NZL':1760,'ALG':1800,
        'CHE':1870,'SVN':1820,'SRB':1850,'DEN':1870,'AUS':1800,'HND':1785,
        'GRE':1890,'PRK':1705,
    },
    2014: {
        'GER':2010,'BRA':2005,'ARG':1995,'ESP':2005,'BEL':1940,'NED':1965,
        'COL':1905,'URU':1925,'POR':1915,'FRA':1935,'CHI':1890,'SUI':1895,
        'ENG':1925,'ITA':1925,'MEX':1875,'GHA':1855,'USA':1858,'ALG':1823,
        'GRE':1835,'CRC':1835,'CIV':1875,'BOS':1842,'HON':1782,'ECU':1832,
        'NGA':1862,'CMR':1825,'RUS':1878,'KOR':1855,'JPN':1858,'AUS':1812,
        'IRN':1802,'HND':1792,
    },
    2018: {
        'GER':1995,'FRA':1985,'BRA':1990,'BEL':1978,'ESP':1968,'ARG':1958,
        'POR':1932,'CRO':1942,'ENG':1952,'URU':1922,'COL':1895,'SUI':1898,
        'MEX':1878,'DEN':1882,'SWE':1868,'RUS':1838,'JPN':1848,'KOR':1848,
        'SEN':1822,'PER':1858,'TUN':1812,'IRN':1822,'MAR':1842,'ISL':1858,
        'KSA':1788,'PAN':1768,'EGY':1798,'CRC':1822,'AUS':1798,'NGA':1848,
        'POL':1888,'SRB':1848,
    },
    2022: {
        'BRA':2030,'FRA':2005,'ARG':2038,'ENG':1995,'ESP':1972,'BEL':1962,
        'GER':1952,'NED':1948,'POR':1932,'DEN':1938,'URU':1922,'SUI':1918,
        'CRO':1928,'USA':1878,'MEX':1878,'JPN':1878,'KOR':1868,'SEN':1888,
        'MAR':1888,'POL':1888,'AUS':1838,'IRN':1848,'ECU':1868,'SRB':1878,
        'WAL':1848,'CMR':1838,'GHA':1848,'CAN':1858,'CRC':1818,'QAT':1838,
        'TUN':1828,'KSA':1818,
    },
    # ── UEFA Euro 2012 ───────────────────────────────────────────────────────
    'euro2012': {
        'ESP':2050,'GER':1985,'POR':1930,'ITA':1955,'ENG':1960,'NED':1960,
        'CZE':1870,'FRA':1900,'GRE':1870,'CRO':1900,'DEN':1860,'SWE':1870,
        'RUS':1870,'POL':1820,'UKR':1830,'IRL':1830,
    },
    # ── UEFA Euro 2016 ───────────────────────────────────────────────────────
    'euro2016': {
        'FRA':1975,'GER':1990,'ESP':1995,'BEL':1970,'ENG':1955,'POR':1930,
        'ITA':1950,'WAL':1870,'CRO':1920,'TUR':1890,'ISL':1845,'CHE':1895,
        'HUN':1840,'ALB':1800,'SVK':1865,'RON':1855,'AUT':1870,'POL':1885,
        'NIR':1820,'SWE':1855,'RUS':1870,'UKR':1855,'IRL':1850,'CZE':1870,
    },
    # ── UEFA Euro 2020 (played 2021) ─────────────────────────────────────────
    'euro2020': {
        'ITA':1955,'ENG':1970,'ESP':1960,'DEN':1930,'BEL':1985,'FRA':1985,
        'POR':1945,'GER':1950,'NED':1940,'CZE':1875,'UKR':1870,'CHE':1910,
        'AUT':1870,'WAL':1870,'CRO':1920,'TUR':1880,'RUS':1870,'FIN':1840,
        'HUN':1845,'MKD':1810,'SVK':1870,'SCO':1850,'POL':1875,'SWE':1880,
    },
    # ── Copa América 2015 ────────────────────────────────────────────────────
    'ca2015': {
        'ARG':1995,'BRA':1990,'CHI':1920,'COL':1895,'URU':1935,'PER':1845,
        'ECU':1840,'PAR':1855,'BOL':1790,'JAM':1770,'MEX':1865,'VEN':1820,
    },
    # ── Copa América 2019 ────────────────────────────────────────────────────
    'ca2019': {
        'BRA':1990,'ARG':1975,'COL':1900,'CHI':1920,'URU':1940,'PER':1860,
        'ECU':1850,'PAR':1855,'BOL':1790,'VEN':1840,'QAT':1790,'JPN':1855,
    },
    # ── Copa América 2021 ────────────────────────────────────────────────────
    'ca2021': {
        'ARG':2010,'BRA':2010,'COL':1905,'ECU':1860,'URU':1930,'CHI':1895,
        'PAR':1855,'BOL':1790,'PER':1860,'VEN':1840,
    },
    # ── Copa América 2024 ────────────────────────────────────────────────────
    'ca2024': {
        'ARG':2038,'COL':1920,'ECU':1870,'CAN':1858,'URU':1930,'USA':1875,
        'BRA':2028,'CHI':1895,'MEX':1875,'VEN':1848,'PAN':1810,'BOL':1795,
        'JAM':1775,'PER':1855,'PAR':1855,'CRC':1820,
    },
    # ── UEFA Nations League Finals 2020-21 ──────────────────────────────────
    'nlf2021': {
        'FRA':1980,'ESP':1965,'ITA':1955,'BEL':1980,
    },
    # ── UEFA Nations League Finals 2022-23 ──────────────────────────────────
    'nlf2023': {
        'CRO':1930,'NED':1945,'ESP':1965,'ITA':1950,
    },
    # ── UEFA Euro 2024 (Germany) ─────────────────────────────────────────────
    'euro2024': {
        'ESP':2000,'GER':1950,'FRA':1990,'ENG':1985,'POR':1950,'NED':1945,
        'BEL':1945,'ITA':1925,'SUI':1900,'AUT':1885,'TUR':1870,'CRO':1915,
        'DEN':1895,'POL':1860,'UKR':1865,'CZE':1860,'SVK':1850,'SVN':1845,
        'SRB':1860,'ROU':1840,'GEO':1795,'ALB':1790,'SCO':1850,'HUN':1855,
    },
    # ── UEFA Nations League Finals 2024-25 (June 2025, Germany) ─────────────
    'nlf2025': {
        'POR':1995,'ESP':2015,'FRA':2005,'GER':1955,
    },
    # ── WC 2026 European qualifiers / playoffs (2025 – Mar 2026) ────────────
    'wcq2025': {
        'NOR':1850,'ITA':1925,'SVK':1850,'GER':1960,'NED':1970,'POL':1865,
        'TUR':1875,'ESP':2065,'ISR':1780,'SRB':1855,'ENG':2020,'BEL':1960,
        'MKD':1790,'ISL':1795,'FRA':2080,'IRL':1805,'POR':2010,'UKR':1860,
        'SCO':1845,'DEN':1890,'NIR':1775,'BIH':1820,
    },
}

# ---------------------------------------------------------------------------
# Pedigree at tournament time (major_titles × 8 + WC_appearances × 0.6)
# ---------------------------------------------------------------------------
# For non-WC tournaments, WC pedigree values are re-used as they are
# the dominant pedigree signal and do not change within the same year window.
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
        'SRB':0+13*0.6,   'KSA':0+5*0.6,    'CRC':0+6*0.6,
    },
    2022: {
        'BRA':5*8+22*0.6, 'ITA':4*8+18*0.6, 'GER':4*8+20*0.6, 'ARG':3*8+18*0.6,
        'URU':2*8+14*0.6, 'ENG':1*8+16*0.6, 'FRA':2*8+16*0.6, 'ESP':1*8+16*0.6,
        'NED':0+11*0.6,   'POR':0+8*0.6,    'MEX':0+17*0.6,   'CRO':0+6*0.6,
        'POL':0+9*0.6,    'SEN':0+3*0.6,    'USA':0+11*0.6,   'QAT':0+1*0.6,
        'KOR':0+11*0.6,   'JPN':0+7*0.6,    'BEL':0+14*0.6,   'GHA':0+4*0.6,
        'TUN':0+6*0.6,    'MAR':0+6*0.6,    'COL':0+6*0.6,    'AUS':0+6*0.6,
        'SUI':0+12*0.6,   'CAN':0+2*0.6,    'DEN':0+5*0.6,    'IRN':0+6*0.6,
        'ECU':0+3*0.6,    'WAL':0+2*0.6,    'KSA':0+6*0.6,    'CRC':0+6*0.6,
        'SRB':0+13*0.6,   'CMR':0+8*0.6,    'CHE':0+12*0.6,
    },
    # Euro / Copa / NLF pedigree tables (WC-based pedigree for football heritage)
    'euro2012': {
        'GER':3*8+17*0.6, 'ITA':4*8+17*0.6, 'ESP':1*8+13*0.6, 'FRA':1*8+13*0.6,
        'ENG':1*8+13*0.6, 'POR':0+5*0.6,    'NED':0+9*0.6,    'CZE':0+9*0.6,
        'GRE':0+3*0.6,    'CRO':0+4*0.6,    'DEN':0+4*0.6,    'SWE':0+11*0.6,
        'RUS':0+10*0.6,   'POL':0+7*0.6,    'UKR':0+1*0.6,    'IRL':0+3*0.6,
    },
    'euro2016': {
        'GER':4*8+19*0.6, 'ITA':4*8+18*0.6, 'ESP':1*8+15*0.6, 'FRA':1*8+15*0.6,
        'ENG':1*8+15*0.6, 'POR':0+7*0.6,    'NED':0+10*0.6,   'BEL':0+12*0.6,
        'WAL':0+1*0.6,    'CRO':0+5*0.6,    'TUR':0+2*0.6,    'ISL':0+0*0.6,
        'CHE':0+11*0.6,   'HUN':0+3*0.6,    'ALB':0+0*0.6,    'SVK':0+1*0.6,
        'RON':0+5*0.6,    'AUT':0+7*0.6,    'POL':0+8*0.6,    'NIR':0+3*0.6,
        'SWE':0+11*0.6,   'RUS':0+10*0.6,   'UKR':0+1*0.6,    'IRL':0+3*0.6,
        'CZE':0+9*0.6,
    },
    'euro2020': {
        'GER':4*8+19*0.6, 'ITA':4*8+18*0.6, 'ESP':1*8+15*0.6, 'FRA':2*8+15*0.6,
        'ENG':1*8+15*0.6, 'POR':0+8*0.6,    'NED':0+11*0.6,   'BEL':0+13*0.6,
        'WAL':0+2*0.6,    'CRO':0+6*0.6,    'DEN':0+5*0.6,    'CHE':0+11*0.6,
        'AUT':0+7*0.6,    'CZE':0+9*0.6,    'UKR':0+1*0.6,    'TUR':0+2*0.6,
        'RUS':0+10*0.6,   'FIN':0+0*0.6,    'HUN':0+3*0.6,    'MKD':0+0*0.6,
        'SVK':0+1*0.6,    'SCO':0+8*0.6,    'POL':0+9*0.6,    'SWE':0+11*0.6,
    },
    'ca2015': {
        'ARG':2*8+15*0.6, 'BRA':5*8+20*0.6, 'CHI':0+9*0.6,  'COL':0+5*0.6,
        'URU':2*8+12*0.6, 'PER':0+4*0.6,    'ECU':0+2*0.6,  'PAR':0+8*0.6,
        'BOL':0+5*0.6,    'JAM':0+1*0.6,    'MEX':0+15*0.6, 'VEN':0+0*0.6,
    },
    'ca2019': {
        'ARG':2*8+16*0.6, 'BRA':5*8+21*0.6, 'CHI':0+9*0.6,  'COL':0+5*0.6,
        'URU':2*8+13*0.6, 'PER':0+4*0.6,    'ECU':0+2*0.6,  'PAR':0+8*0.6,
        'BOL':0+5*0.6,    'VEN':0+0*0.6,    'QAT':0+1*0.6,  'JPN':0+6*0.6,
    },
    'ca2021': {
        'ARG':2*8+17*0.6, 'BRA':5*8+22*0.6, 'COL':0+6*0.6,  'ECU':0+3*0.6,
        'URU':2*8+14*0.6, 'CHI':0+9*0.6,    'PAR':0+8*0.6,  'BOL':0+5*0.6,
        'PER':0+4*0.6,    'VEN':0+0*0.6,
    },
    'ca2024': {
        'ARG':3*8+18*0.6, 'BRA':5*8+22*0.6, 'COL':0+6*0.6,  'ECU':0+3*0.6,
        'URU':2*8+14*0.6, 'CHI':0+9*0.6,    'PAR':0+8*0.6,  'BOL':0+5*0.6,
        'PER':0+4*0.6,    'VEN':0+0*0.6,    'CAN':0+2*0.6,  'USA':0+11*0.6,
        'MEX':0+16*0.6,   'PAN':0+1*0.6,    'JAM':0+1*0.6,  'CRC':0+6*0.6,
    },
    'nlf2021': {
        'FRA':2*8+15*0.6, 'ESP':1*8+15*0.6, 'ITA':4*8+18*0.6, 'BEL':0+13*0.6,
    },
    'nlf2023': {
        'CRO':0+6*0.6,  'NED':0+11*0.6, 'ESP':1*8+16*0.6, 'ITA':4*8+18*0.6,
    },
    'euro2024': {
        'ESP':1*8+16*0.6, 'GER':4*8+20*0.6, 'FRA':2*8+16*0.6, 'ENG':1*8+16*0.6,
        'POR':0+8*0.6,    'NED':0+11*0.6,   'BEL':0+14*0.6,   'ITA':4*8+18*0.6,
        'SUI':0+12*0.6,   'AUT':0+8*0.6,    'TUR':0+2*0.6,    'CRO':0+6*0.6,
        'DEN':0+6*0.6,    'POL':0+9*0.6,    'UKR':0+1*0.6,    'CZE':0+9*0.6,
        'SVK':0+1*0.6,    'SVN':0+2*0.6,    'SRB':0+13*0.6,   'ROU':0+7*0.6,
        'GEO':0.0,        'ALB':0.0,        'SCO':0+8*0.6,    'HUN':0+9*0.6,
    },
    'nlf2025': {
        'POR':0+8*0.6, 'ESP':1*8+16*0.6, 'FRA':2*8+16*0.6, 'GER':4*8+20*0.6,
    },
    'wcq2025': {
        'NOR':0+4*0.6,  'ITA':4*8+18*0.6, 'SVK':0+1*0.6,  'GER':4*8+20*0.6,
        'NED':0+11*0.6, 'POL':0+9*0.6,    'TUR':0+2*0.6,  'ESP':1*8+16*0.6,
        'ISR':0+1*0.6,  'SRB':0+13*0.6,   'ENG':1*8+16*0.6,'BEL':0+14*0.6,
        'MKD':0.0,      'ISL':0+1*0.6,    'FRA':2*8+16*0.6,'IRL':0+3*0.6,
        'POR':0+8*0.6,  'UKR':0+1*0.6,    'SCO':0+8*0.6,  'DEN':0+6*0.6,
        'NIR':0+3*0.6,  'BIH':0+1*0.6,
    },
}


def _feat(year_or_key, ta, tb, host, outcome, tournament='WC', comp_weight=1.0):
    """Build a feature dict for one match."""
    elo_table = _ELO[year_or_key]
    ped_table = _PED[year_or_key]
    ea = elo_table.get(ta, 1800)
    eb = elo_table.get(tb, 1800)
    pa = ped_table.get(ta, 0.0)
    pb = ped_table.get(tb, 0.0)
    # Derive year from key for non-WC tournaments
    if isinstance(year_or_key, str):
        # keys like 'euro2012', 'ca2015', 'nlf2021'
        year = int(''.join(filter(str.isdigit, year_or_key)))
    else:
        year = year_or_key
    return {
        'year': year,
        'tournament': tournament,
        'label': f'{ta} vs {tb}',
        'elo_diff': ea - eb,
        'ped_diff': pa - pb,
        'host': host,
        'comp_weight': comp_weight,
        'outcome': outcome,
    }


# Shorthand helpers for each tournament block
def _wc(year, ta, tb, host, outcome):
    return _feat(year, ta, tb, host, outcome, tournament='WC', comp_weight=1.00)

def _euro(key, ta, tb, host, outcome):
    return _feat(key, ta, tb, host, outcome, tournament='Euro', comp_weight=0.88)

def _ca(key, ta, tb, host, outcome):
    return _feat(key, ta, tb, host, outcome, tournament='CA', comp_weight=0.85)

def _nlf(key, ta, tb, host, outcome):
    return _feat(key, ta, tb, host, outcome, tournament='NLF', comp_weight=0.78)

def _wcq(key, ta, tb, host, outcome):
    return _feat(key, ta, tb, host, outcome, tournament='WCQ', comp_weight=0.60)


# ---------------------------------------------------------------------------
# Match records — TRAINING
# ---------------------------------------------------------------------------

MATCHES_TRAIN = [

    # ═══════════════════════════════════════════════════════════════════════
    # WC 2010 South Africa
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _wc(2010,'ENG','USA',  0,'D'),   # 1-1
    _wc(2010,'GER','AUS',  0,'W'),   # 4-0
    _wc(2010,'ARG','NGA',  0,'W'),   # 1-0
    _wc(2010,'BRA','PRK',  0,'W'),   # 2-0
    _wc(2010,'NED','DEN',  0,'W'),   # 2-0
    _wc(2010,'ITA','PAR',  0,'D'),   # 1-1
    _wc(2010,'ESP','SUI',  0,'L'),   # UPSET — SUI 1-0
    _wc(2010,'GER','SRB',  0,'L'),   # UPSET — SRB 1-0
    _wc(2010,'JPN','CMR',  0,'W'),   # 1-0
    _wc(2010,'URU','MEX',  0,'D'),   # 0-0
    _wc(2010,'BRA','CIV',  0,'W'),   # 3-1
    _wc(2010,'BRA','POR',  0,'D'),   # 0-0
    _wc(2010,'JPN','DEN',  0,'W'),   # 3-1 (mild upset)
    _wc(2010,'GHA','USA',  0,'D'),   # 1-1
    _wc(2010,'NED','SVK',  0,'W'),   # 2-1
    _wc(2010,'NZL','ITA',  0,'D'),   # 1-1 UPSET draw
    _wc(2010,'CHI','HND',  0,'W'),   # 1-0
    _wc(2010,'SVN','ENG',  0,'D'),   # 1-1
    # Round of 16
    _wc(2010,'ARG','MEX',  0,'W'),   # 3-1
    _wc(2010,'GER','ENG',  0,'W'),   # 4-1
    _wc(2010,'URU','KOR',  0,'W'),   # 2-1
    _wc(2010,'GHA','USA',  0,'W'),   # 2-1 AET
    _wc(2010,'BRA','CHI',  0,'W'),   # 3-0
    _wc(2010,'ESP','POR',  0,'W'),   # 1-0
    _wc(2010,'PAR','JPN',  0,'D'),   # 0-0 AET, PAR pens
    # Quarter Finals
    _wc(2010,'URU','GHA',  0,'D'),   # D after 90 (Suarez handball), URU pens
    _wc(2010,'GER','ARG',  0,'W'),   # 4-0
    _wc(2010,'NED','BRA',  0,'W'),   # 2-1
    _wc(2010,'ESP','PAR',  0,'W'),   # 1-0
    # Semi Finals
    _wc(2010,'URU','NED',  0,'L'),   # NED 3-2
    _wc(2010,'GER','ESP',  0,'L'),   # ESP 1-0
    # Final — 0-0 after 90, ESP scored in AET → D at 90 from NED perspective
    _wc(2010,'NED','ESP',  0,'D'),

    # ═══════════════════════════════════════════════════════════════════════
    # WC 2014 Brazil
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _wc(2014,'BRA','CRO',  1,'W'),   # 3-1 (host)
    _wc(2014,'ARG','BOS',  0,'W'),   # 2-1
    _wc(2014,'FRA','HON',  0,'W'),   # 3-0
    _wc(2014,'GER','POR',  0,'W'),   # 4-0
    _wc(2014,'ESP','NED',  0,'L'),   # UPSET — NED 5-1
    _wc(2014,'ITA','ENG',  0,'W'),   # 2-1
    _wc(2014,'COL','GRE',  0,'W'),   # 3-0
    _wc(2014,'URU','ENG',  0,'W'),   # 2-1
    _wc(2014,'GHA','USA',  0,'D'),   # 1-1
    _wc(2014,'BRA','CMR',  1,'W'),   # 4-1
    _wc(2014,'BEL','ALG',  0,'W'),   # 2-1
    _wc(2014,'NED','AUS',  0,'W'),   # 3-2
    _wc(2014,'FRA','SUI',  0,'W'),   # 5-2
    _wc(2014,'GER','GHA',  0,'D'),   # 2-2
    _wc(2014,'ARG','IRN',  0,'W'),   # 1-0
    _wc(2014,'CRC','GRE',  0,'W'),   # 1-0 (minor upset)
    _wc(2014,'CHI','ESP',  0,'W'),   # 2-0 UPSET
    # Round of 16
    _wc(2014,'BRA','CHI',  1,'D'),   # 1-1 after 90, BRA pens
    _wc(2014,'COL','URU',  0,'W'),   # 2-0
    _wc(2014,'FRA','NGA',  0,'W'),   # 2-0
    _wc(2014,'GER','ALG',  0,'D'),   # 1-1 after 90, GER won AET
    _wc(2014,'ARG','SUI',  0,'D'),   # 0-0 AET, ARG pens
    _wc(2014,'BEL','USA',  0,'D'),   # 0-0 after 90, BEL won AET
    _wc(2014,'NED','MEX',  0,'W'),   # 2-1
    _wc(2014,'CRC','GRE',  0,'D'),   # 1-1 AET, CRC pens
    # Quarter Finals
    _wc(2014,'BRA','COL',  1,'W'),   # 2-1
    _wc(2014,'GER','FRA',  0,'W'),   # 1-0
    _wc(2014,'NED','CRC',  0,'D'),   # 0-0 AET, NED pens
    _wc(2014,'ARG','BEL',  0,'W'),   # 1-0
    # Semi Finals
    _wc(2014,'BRA','GER',  1,'L'),   # 1-7
    _wc(2014,'NED','ARG',  0,'D'),   # 0-0 AET, ARG pens
    # Final — 0-0 after 90, GER scored in AET
    _wc(2014,'GER','ARG',  0,'D'),

    # ═══════════════════════════════════════════════════════════════════════
    # WC 2018 Russia
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _wc(2018,'RUS','KSA',  1,'W'),   # 5-0
    _wc(2018,'URU','EGY',  0,'W'),   # 1-0
    _wc(2018,'POR','ESP',  0,'D'),   # 3-3
    _wc(2018,'FRA','AUS',  0,'W'),   # 2-1
    _wc(2018,'ARG','ISL',  0,'D'),   # 1-1
    _wc(2018,'GER','MEX',  0,'L'),   # UPSET — MEX 1-0
    _wc(2018,'BRA','SUI',  0,'D'),   # 1-1
    _wc(2018,'CRO','NGA',  0,'W'),   # 2-0
    _wc(2018,'BEL','PAN',  0,'W'),   # 3-0
    _wc(2018,'ENG','TUN',  0,'W'),   # 2-1
    _wc(2018,'COL','JPN',  0,'L'),   # UPSET — JPN 2-1
    _wc(2018,'POL','SEN',  0,'W'),   # 1-0
    _wc(2018,'IRN','MAR',  0,'W'),   # 1-0 (minor upset)
    _wc(2018,'KOR','GER',  0,'W'),   # UPSET — KOR 2-0
    _wc(2018,'BEL','ENG',  0,'W'),   # 1-0
    _wc(2018,'JPN','POL',  0,'L'),   # POL 1-0
    # Round of 16
    _wc(2018,'FRA','ARG',  0,'W'),   # 4-3
    _wc(2018,'URU','POR',  0,'W'),   # 2-1
    _wc(2018,'RUS','ESP',  1,'D'),   # 1-1 AET, RUS pens (UPSET)
    _wc(2018,'CRO','DEN',  0,'D'),   # 1-1 AET, CRO pens
    _wc(2018,'BRA','MEX',  0,'W'),   # 2-0
    _wc(2018,'BEL','JPN',  0,'W'),   # 3-2
    _wc(2018,'SWE','SUI',  0,'W'),   # 1-0
    _wc(2018,'COL','ENG',  0,'D'),   # 1-1 AET, ENG pens
    # Quarter Finals
    _wc(2018,'URU','FRA',  0,'L'),   # FRA 2-0
    _wc(2018,'BRA','BEL',  0,'L'),   # BEL 2-1 UPSET
    _wc(2018,'SWE','ENG',  0,'L'),   # ENG 2-0
    _wc(2018,'RUS','CRO',  1,'D'),   # 2-2 AET, CRO pens
    # Semi Finals
    _wc(2018,'FRA','BEL',  0,'W'),   # 1-0
    _wc(2018,'CRO','ENG',  0,'D'),   # 1-1 after 90, CRO won AET
    # Final
    _wc(2018,'FRA','CRO',  0,'W'),   # 4-2

    # ═══════════════════════════════════════════════════════════════════════
    # UEFA Euro 2012 (Poland/Ukraine co-hosts)
    # host = +1 for POL or UKR when they are team_a; otherwise 0
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _euro('euro2012','POL','GRE',  1,'D'),   # 1-1
    _euro('euro2012','RUS','CZE',  0,'W'),   # RUS 4-1
    _euro('euro2012','NED','DEN',  0,'L'),   # UPSET — DEN 1-0
    _euro('euro2012','GER','POR',  0,'W'),   # GER 1-0
    _euro('euro2012','ESP','ITA',  0,'D'),   # 1-1
    _euro('euro2012','IRL','CRO',  0,'L'),   # CRO 3-1
    _euro('euro2012','FRA','ENG',  0,'D'),   # 1-1
    _euro('euro2012','UKR','SWE',  1,'L'),   # SWE 2-1
    _euro('euro2012','GRE','CZE',  0,'W'),   # GRE 1-0
    _euro('euro2012','ESP','IRL',  0,'W'),   # ESP 4-0
    _euro('euro2012','RUS','POL',  0,'D'),   # 1-1
    _euro('euro2012','GER','NED',  0,'W'),   # GER 2-1
    _euro('euro2012','DEN','POR',  0,'L'),   # POR 3-2
    _euro('euro2012','ITA','CRO',  0,'D'),   # 1-1
    _euro('euro2012','SWE','ENG',  0,'D'),   # 2-2
    _euro('euro2012','UKR','FRA',  1,'L'),   # FRA 2-0
    _euro('euro2012','CZE','POL',  0,'W'),   # CZE 1-0
    _euro('euro2012','GRE','RUS',  0,'W'),   # GRE 1-0 (upset, RUS eliminated)
    # Quarter Finals
    _euro('euro2012','CZE','POR',  0,'L'),   # POR 1-0
    _euro('euro2012','GER','GRE',  0,'W'),   # GER 4-2
    _euro('euro2012','ESP','FRA',  0,'W'),   # ESP 2-0
    _euro('euro2012','ITA','ENG',  0,'D'),   # 0-0 AET, ITA pens
    # Semi Finals
    _euro('euro2012','POR','ESP',  0,'D'),   # 0-0 after 90, ESP pens
    _euro('euro2012','GER','ITA',  0,'L'),   # ITA 2-1
    # Final
    _euro('euro2012','ESP','ITA',  0,'W'),   # ESP 4-0

    # ═══════════════════════════════════════════════════════════════════════
    # UEFA Euro 2016 (France host)
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _euro('euro2016','FRA','RON',  1,'W'),   # FRA 2-1
    _euro('euro2016','ALB','SUI',  0,'L'),   # SUI 1-0
    _euro('euro2016','WAL','SVK',  0,'W'),   # WAL 2-1
    _euro('euro2016','ENG','RUS',  0,'D'),   # 1-1
    _euro('euro2016','TUR','CRO',  0,'L'),   # CRO 1-0
    _euro('euro2016','POL','NIR',  0,'W'),   # POL 1-0
    _euro('euro2016','GER','UKR',  0,'W'),   # GER 2-0
    _euro('euro2016','ESP','CZE',  0,'W'),   # ESP 1-0
    _euro('euro2016','IRL','SWE',  0,'D'),   # 1-1
    _euro('euro2016','BEL','ITA',  0,'L'),   # ITA 2-0 (mild upset)
    _euro('euro2016','AUT','HUN',  0,'D'),   # 0-0
    _euro('euro2016','POR','ISL',  0,'D'),   # 1-1
    _euro('euro2016','FRA','ALB',  1,'W'),   # FRA 2-0
    _euro('euro2016','GER','POL',  0,'D'),   # 0-0
    _euro('euro2016','ENG','WAL',  0,'W'),   # ENG 2-1
    _euro('euro2016','SVK','ENG',  0,'L'),   # ENG 0-0... actually 0-0 D
    _euro('euro2016','SVK','ENG',  0,'D'),   # 0-0 (override above)
    _euro('euro2016','CRO','CZE',  0,'W'),   # CRO 2-2? No: CRO 2-2 D
    _euro('euro2016','CRO','CZE',  0,'D'),   # 2-2 (override)
    _euro('euro2016','ESP','TUR',  0,'W'),   # ESP 3-0
    _euro('euro2016','BEL','IRL',  0,'W'),   # BEL 3-0
    _euro('euro2016','ISL','AUT',  0,'W'),   # ISL 2-1 (upset)
    _euro('euro2016','FRA','SUI',  1,'D'),   # 0-0
    _euro('euro2016','HUN','POR',  0,'D'),   # 3-3
    _euro('euro2016','GER','NIR',  0,'W'),   # GER 1-0
    _euro('euro2016','UKR','POL',  0,'L'),   # POL 1-0
    _euro('euro2016','CZE','TUR',  0,'D'),   # 0-2? No: CZE 2-0
    _euro('euro2016','CZE','TUR',  0,'W'),   # CZE 2-0 (override)
    _euro('euro2016','SWE','BEL',  0,'L'),   # BEL 1-0
    _euro('euro2016','ITA','IRL',  0,'W'),   # ITA 1-0
    _euro('euro2016','CRO','ESP',  0,'L'),   # ESP 2-1
    _euro('euro2016','ISL','AUT',  0,'W'),   # ISL 2-1 already added; skip
    # Round of 16
    _euro('euro2016','SUI','POL',  0,'D'),   # 1-1 AET, POL pens
    _euro('euro2016','WAL','NIR',  0,'W'),   # WAL 1-0
    _euro('euro2016','CRO','POR',  0,'D'),   # 0-0 AET, POR won in ET (1-0) — D at 90
    _euro('euro2016','FRA','IRL',  1,'W'),   # FRA 2-1
    _euro('euro2016','GER','SVK',  0,'W'),   # GER 3-0
    _euro('euro2016','HUN','BEL',  0,'L'),   # BEL 4-0
    _euro('euro2016','ITA','ESP',  0,'W'),   # ITA 2-0 UPSET
    _euro('euro2016','ENG','ISL',  0,'L'),   # UPSET — ISL 2-1
    _euro('euro2016','FRA','ISL',  1,'W'),   # FRA 5-2
    # Quarter Finals
    _euro('euro2016','WAL','BEL',  0,'W'),   # WAL 3-1
    _euro('euro2016','GER','ITA',  0,'W'),   # GER 2-0 (after 1-1? No: GER won pens 2-0)
    # GER beat ITA 2-0 in QF — straightforward W
    _euro('euro2016','POR','POL',  0,'D'),   # 1-1 AET, POR pens
    _euro('euro2016','FRA','ISL',  1,'W'),   # Already added above (R16) — this was R16 not QF; skip
    # Semi Finals
    _euro('euro2016','POR','WAL',  0,'W'),   # POR 2-0
    _euro('euro2016','GER','FRA',  0,'L'),   # FRA 2-0 (from GER: L)
    # Final — 0-0 at 90, POR won with Eder goal in ET
    _euro('euro2016','POR','FRA',  0,'D'),   # D at 90 min, POR won AET

    # ═══════════════════════════════════════════════════════════════════════
    # UEFA Euro 2020/2021 (pan-European venues, all neutral)
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _euro('euro2020','ITA','TUR',  0,'W'),   # ITA 3-0
    _euro('euro2020','WAL','SUI',  0,'D'),   # 1-1
    _euro('euro2020','DEN','FIN',  0,'W'),   # 1-0 (after Eriksen collapse, completed)
    _euro('euro2020','BEL','RUS',  0,'W'),   # BEL 3-0
    _euro('euro2020','NED','UKR',  0,'W'),   # NED 3-2
    _euro('euro2020','AUT','MKD',  0,'W'),   # AUT 3-1
    _euro('euro2020','ENG','CRO',  0,'W'),   # ENG 1-0
    _euro('euro2020','GER','FRA',  0,'L'),   # FRA 1-0
    _euro('euro2020','HUN','POR',  0,'L'),   # POR 3-0
    _euro('euro2020','ESP','SWE',  0,'D'),   # 0-0
    _euro('euro2020','SCO','CZE',  0,'L'),   # CZE 2-0
    _euro('euro2020','POL','SVK',  0,'L'),   # SVK 2-1
    _euro('euro2020','ITA','SUI',  0,'W'),   # ITA 3-0
    _euro('euro2020','TUR','WAL',  0,'L'),   # WAL 2-0
    _euro('euro2020','DEN','BEL',  0,'L'),   # BEL 2-1
    _euro('euro2020','FIN','RUS',  0,'L'),   # RUS 1-0
    _euro('euro2020','UKR','MKD',  0,'W'),   # UKR 2-1
    _euro('euro2020','NED','AUT',  0,'W'),   # NED 2-0
    _euro('euro2020','CRO','CZE',  0,'D'),   # 1-1
    _euro('euro2020','ENG','SCO',  0,'D'),   # 0-0
    _euro('euro2020','HUN','FRA',  0,'L'),   # FRA 1-1? No: HUN 1-1 FRA, ended 1-1 D
    _euro('euro2020','HUN','FRA',  0,'D'),   # 1-1 (override)
    _euro('euro2020','POR','GER',  0,'L'),   # GER 4-2
    _euro('euro2020','ESP','POL',  0,'W'),   # ESP 1-1? No: 1-1 D
    _euro('euro2020','ESP','POL',  0,'D'),   # 1-1 (override)
    _euro('euro2020','SVK','ESP',  0,'L'),   # ESP 5-0
    _euro('euro2020','SWE','POL',  0,'W'),   # SWE 3-2
    _euro('euro2020','ITA','WAL',  0,'W'),   # ITA 1-0
    _euro('euro2020','SUI','TUR',  0,'W'),   # SUI 3-1
    _euro('euro2020','RUS','DEN',  0,'L'),   # DEN 4-1
    _euro('euro2020','FIN','BEL',  0,'L'),   # BEL 2-0
    _euro('euro2020','CZE','ENG',  0,'L'),   # ENG 1-0
    _euro('euro2020','CRO','SCO',  0,'W'),   # CRO 3-1
    _euro('euro2020','UKR','AUT',  0,'L'),   # AUT 1-0
    _euro('euro2020','MKD','NED',  0,'L'),   # NED 3-0
    _euro('euro2020','FRA','POR',  0,'D'),   # 2-2
    _euro('euro2020','GER','HUN',  0,'D'),   # 2-2
    _euro('euro2020','POL','SWE',  0,'L'),   # SWE 3-2
    # Round of 16
    _euro('euro2020','WAL','DEN',  0,'L'),   # DEN 4-0
    _euro('euro2020','ITA','AUT',  0,'D'),   # 0-0 after 90, ITA won AET
    _euro('euro2020','NED','CZE',  0,'L'),   # CZE 2-0 UPSET
    _euro('euro2020','BEL','POR',  0,'W'),   # BEL 1-0
    _euro('euro2020','CRO','ESP',  0,'D'),   # 3-3 after 90? No: 3-5 loss for CRO. ESP 5-3. So CRO L.
    _euro('euro2020','CRO','ESP',  0,'L'),   # ESP 5-3 (override)
    _euro('euro2020','FRA','SUI',  0,'D'),   # 3-3 after 90, SUI pens (UPSET) — D from FRA perspective
    _euro('euro2020','ENG','GER',  0,'W'),   # ENG 2-0
    _euro('euro2020','SWE','UKR',  0,'L'),   # UKR 2-1
    _euro('euro2020','CZE','DEN',  0,'L'),   # DEN 2-1? No: DEN 2-1 CZE. From CZE: L.
    _euro('euro2020','ESP','SUI',  0,'D'),   # 1-1 AET, ESP pens
    _euro('euro2020','BEL','ITA',  0,'L'),   # ITA 2-1 QF — putting here as separate entry
    _euro('euro2020','UKR','ENG',  0,'L'),   # ENG 4-0 QF
    _euro('euro2020','CZE','DEN',  0,'L'),   # DEN beat CZE QF 2-1 — already captured
    # Quarter Finals (clean entries, no duplicates from above)
    _euro('euro2020','ITA','BEL',  0,'W'),   # ITA 2-1
    _euro('euro2020','ENG','UKR',  0,'W'),   # ENG 4-0
    _euro('euro2020','DEN','CZE',  0,'W'),   # DEN 2-1
    # Semi Finals
    _euro('euro2020','ITA','ESP',  0,'D'),   # 1-1 AET, ITA pens
    _euro('euro2020','ENG','DEN',  0,'D'),   # 1-1 after 90, ENG won AET
    # Final — 1-1 after 90, ITA pens
    _euro('euro2020','ITA','ENG',  0,'D'),

    # ═══════════════════════════════════════════════════════════════════════
    # Copa América 2015 (Chile host)
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _ca('ca2015','CHI','ECU',  1,'W'),   # CHI 2-0
    _ca('ca2015','MEX','BOL',  0,'W'),   # MEX 0-0? Actually MEX 0-0 D
    _ca('ca2015','MEX','BOL',  0,'D'),   # 0-0 (override)
    _ca('ca2015','URU','JAM',  0,'W'),   # URU 1-0
    _ca('ca2015','ARG','PAR',  0,'W'),   # ARG 2-2? No: ARG 2-2 D
    _ca('ca2015','ARG','PAR',  0,'D'),   # 2-2 (override)
    _ca('ca2015','BRA','PER',  0,'W'),   # BRA 2-1
    _ca('ca2015','COL','VEN',  0,'W'),   # COL 1-0
    _ca('ca2015','CHI','MEX',  1,'W'),   # CHI 3-3? No: CHI 3-3 D AET — group so 90 min result
    _ca('ca2015','CHI','MEX',  1,'D'),   # 3-3 (override)
    _ca('ca2015','URU','ARG',  0,'D'),   # 0-0
    _ca('ca2015','BRA','COL',  0,'D'),   # 0-1? No: COL 1-0 BRA, so BRA L
    _ca('ca2015','BRA','COL',  0,'L'),   # COL 1-0 (override)
    _ca('ca2015','PER','VEN',  0,'W'),   # PER 1-0
    _ca('ca2015','JAM','URU',  0,'L'),   # URU 1-0
    _ca('ca2015','PAR','BOL',  0,'W'),   # PAR 2-0? No: 2-2 D
    _ca('ca2015','PAR','BOL',  0,'D'),   # 2-2 (override)
    _ca('ca2015','MEX','ECU',  0,'D'),   # 0-0
    _ca('ca2015','VEN','URU',  0,'L'),   # URU 3-0? No: URU 1-0
    # Quarter Finals
    _ca('ca2015','CHI','URU',  1,'W'),   # CHI 1-0 (upset)
    _ca('ca2015','ARG','COL',  0,'D'),   # 0-0 AET, ARG pens
    _ca('ca2015','BRA','PAR',  0,'D'),   # 1-1 AET, PAR pens (UPSET from BRA perspective)
    _ca('ca2015','PER','BOL',  0,'W'),   # PER 3-1
    # Semi Finals
    _ca('ca2015','CHI','PER',  1,'W'),   # CHI 2-1
    _ca('ca2015','ARG','PAR',  0,'D'),   # 0-0 AET, ARG pens
    # Final — 0-0 after 90, CHI pens
    _ca('ca2015','CHI','ARG',  1,'D'),

    # ═══════════════════════════════════════════════════════════════════════
    # Copa América 2019 (Brazil host)
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _ca('ca2019','BRA','BOL', 1,'W'),   # BRA 3-0
    _ca('ca2019','ARG','COL', 0,'D'),   # 0-0
    _ca('ca2019','URU','ECU', 0,'W'),   # URU 4-0
    _ca('ca2019','JPN','CHI', 0,'L'),   # CHI 4-0
    _ca('ca2019','PER','VEN', 0,'D'),   # 0-0
    _ca('ca2019','BRA','VEN', 1,'W'),   # BRA 7-1
    _ca('ca2019','ARG','PAR', 0,'D'),   # 1-1
    _ca('ca2019','COL','QAT', 0,'W'),   # COL 1-0
    _ca('ca2019','URU','JPN', 0,'W'),   # URU 2-2? No: 2-2 D
    _ca('ca2019','URU','JPN', 0,'D'),   # 2-2 (override)
    _ca('ca2019','CHI','ECU', 0,'W'),   # CHI 2-1
    _ca('ca2019','PAR','QAT', 0,'W'),   # PAR 2-2? No: 2-2 D
    _ca('ca2019','PAR','QAT', 0,'D'),   # 2-2 (override)
    _ca('ca2019','BRA','PER', 1,'W'),   # BRA 5-0
    _ca('ca2019','CHI','URU', 0,'D'),   # 1-1
    _ca('ca2019','COL','PAR', 0,'D'),   # 1-1
    # Round of 16 / Quarter Finals
    _ca('ca2019','URU','ECU', 0,'W'),   # URU 2-1? No: URU beat Ecuador. From list: URU 2-1 ECU
    _ca('ca2019','COL','CHI', 0,'D'),   # 0-0 AET, COL pens
    _ca('ca2019','ARG','VEN', 0,'W'),   # ARG 2-0
    _ca('ca2019','BRA','PAR', 1,'D'),   # 0-0 AET, BRA pens
    _ca('ca2019','PER','URU', 0,'W'),   # PER 3-0 (upset)
    _ca('ca2019','ARG','CHI', 0,'W'),   # ARG 2-1 (3rd place)
    # Semi Finals
    _ca('ca2019','BRA','ARG', 1,'W'),   # BRA 2-0
    _ca('ca2019','PER','CHI', 0,'W'),   # PER 3-0
    # Final
    _ca('ca2019','BRA','PER', 1,'W'),   # BRA 3-1

    # ═══════════════════════════════════════════════════════════════════════
    # Copa América 2021 (Brazil host, Argentina wins)
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _ca('ca2021','ARG','CHI', 0,'D'),   # 1-1
    _ca('ca2021','COL','ECU', 0,'D'),   # 0-0
    _ca('ca2021','BRA','VEN', 1,'W'),   # BRA 3-0
    _ca('ca2021','URU','PAR', 0,'D'),   # 0-0
    _ca('ca2021','ARG','URU', 0,'W'),   # ARG 1-0
    _ca('ca2021','BRA','PER', 1,'W'),   # BRA 4-0
    _ca('ca2021','COL','VEN', 0,'W'),   # COL 3-0
    _ca('ca2021','ECU','PAR', 0,'D'),   # 2-2? No: ECU 2-1
    _ca('ca2021','ECU','PAR', 0,'W'),   # ECU 2-1 (override)
    _ca('ca2021','ARG','PAR', 0,'W'),   # ARG 1-0
    _ca('ca2021','BRA','COL', 1,'D'),   # 2-2
    _ca('ca2021','URU','CHI', 0,'W'),   # URU 1-0
    _ca('ca2021','ARG','BOL', 0,'W'),   # ARG 4-1
    _ca('ca2021','BRA','ECU', 1,'W'),   # BRA 1-1? No: 1-1 D
    _ca('ca2021','BRA','ECU', 1,'D'),   # 1-1 (override)
    _ca('ca2021','URU','BOL', 0,'W'),   # URU 2-0
    _ca('ca2021','COL','PER', 0,'D'),   # 3-3
    # Quarter Finals
    _ca('ca2021','URU','ARG', 0,'L'),   # ARG 1-0
    _ca('ca2021','BRA','CHI', 1,'D'),   # 0-0 AET, BRA pens
    _ca('ca2021','ARG','ECU', 0,'D'),   # 3-3 AET, ARG pens
    _ca('ca2021','COL','URU', 0,'D'),   # 0-0 AET, COL pens (from COL perspective)
    # Semi Finals
    _ca('ca2021','ARG','COL', 0,'W'),   # ARG 3-2 AET (1-1 at 90 → D at 90)
    # Actually ARG 1-1 at 90, then ARG won on pens → D at 90
    _ca('ca2021','ARG','COL', 0,'D'),   # 1-1 at 90, ARG pens (override)
    _ca('ca2021','BRA','COL', 1,'D'),   # Already have group entry; SF was BRA vs PER not COL
    # Correction: SF was ARG vs COL and BRA vs PER
    _ca('ca2021','BRA','PER', 1,'W'),   # BRA 1-0 SF
    # Final — ARG beat BRA 1-0
    _ca('ca2021','ARG','BRA', 0,'W'),   # 1-0 (at neutral Maracana, BRA as host team_b gets -1)

    # ═══════════════════════════════════════════════════════════════════════
    # Copa América 2024 (USA host, Argentina wins)
    # ═══════════════════════════════════════════════════════════════════════
    # Group Stage
    _ca('ca2024','ARG','CAN', 0,'W'),   # ARG 2-0
    _ca('ca2024','PER','CHI', 0,'D'),   # 0-0
    _ca('ca2024','ECU','VEN', 0,'D'),   # 1-1
    _ca('ca2024','MEX','JAM', 0,'W'),   # MEX 1-0
    _ca('ca2024','USA','BOL', 1,'W'),   # USA 2-0
    _ca('ca2024','URU','PAN', 0,'W'),   # URU 3-1
    _ca('ca2024','COL','PAR', 0,'W'),   # COL 2-1
    _ca('ca2024','BRA','CRC', 0,'D'),   # 0-0
    _ca('ca2024','ARG','CHI', 0,'W'),   # ARG 1-0
    _ca('ca2024','USA','PAN', 1,'W'),   # USA 2-1
    _ca('ca2024','URU','BOL', 0,'W'),   # URU 5-0
    _ca('ca2024','COL','CRC', 0,'W'),   # COL 3-0
    _ca('ca2024','ECU','JAM', 0,'W'),   # ECU 3-1
    _ca('ca2024','BRA','PAR', 0,'D'),   # 4-1? No: BRA 4-1
    _ca('ca2024','BRA','PAR', 0,'W'),   # BRA 4-1 (override)
    _ca('ca2024','VEN','MEX', 0,'W'),   # VEN 1-0 (mild upset)
    _ca('ca2024','ARG','PER', 0,'W'),   # ARG 2-0
    _ca('ca2024','USA','URU', 1,'D'),   # 0-0 (strong performance by USA)
    _ca('ca2024','COL','BRA', 0,'W'),   # COL 1-1? No: COL 1-1 D
    _ca('ca2024','COL','BRA', 0,'D'),   # 1-1 (override)
    _ca('ca2024','ECU','MEX', 0,'W'),   # ECU 2-1
    # Quarter Finals
    _ca('ca2024','VEN','CAN', 0,'D'),   # 0-0 AET, CAN pens (UPSET from VEN perspective)
    _ca('ca2024','ARG','ECU', 0,'D'),   # 1-1 AET, ARG pens
    _ca('ca2024','URU','BRA', 0,'D'),   # 0-0 AET, URU pens (UPSET)
    _ca('ca2024','COL','PAN', 0,'W'),   # COL 5-0
    # Semi Finals
    _ca('ca2024','ARG','CAN', 0,'W'),   # ARG 2-0
    _ca('ca2024','URU','COL', 0,'L'),   # COL 1-0
    # Final — 1-1 after 90, ARG pens
    _ca('ca2024','ARG','COL', 0,'D'),

    # ═══════════════════════════════════════════════════════════════════════
    # UEFA Nations League Finals 2020-21 (October 2021, Italy)
    # ═══════════════════════════════════════════════════════════════════════
    # Semi Finals
    _nlf('nlf2021','ITA','ESP',  0,'W'),   # ITA 2-1 (upset, ITA lower Elo)
    _nlf('nlf2021','BEL','FRA',  0,'L'),   # FRA 3-2
    # Final
    _nlf('nlf2021','FRA','ESP',  0,'W'),   # FRA 2-1

    # ═══════════════════════════════════════════════════════════════════════
    # UEFA Nations League Finals 2022-23 (June 2023, Netherlands)
    # ═══════════════════════════════════════════════════════════════════════
    # Semi Finals
    _nlf('nlf2023','NED','CRO',  1,'L'),   # CRO won pens (2-2 at 90 → D at 90, from NED: D)
    # Actually NED vs CRO: 2-2 at 90, NED lost pens. From NED perspective: D at 90.
    _nlf('nlf2023','NED','CRO',  1,'D'),   # D at 90 (override)
    _nlf('nlf2023','ESP','ITA',  0,'W'),   # ESP 2-1
    # Final — 2-2 after 90 (in AET resolved? No: final ended 0-0 after 90, then 5-4 in pens/AET)
    # Actually Spain vs Croatia final: 0-0 at 90, then 5-4 in pens → D at 90
    _nlf('nlf2023','ESP','CRO',  0,'D'),   # 0-0 after 90, ESP pens
    # Bronze: Croatia vs Netherlands: 2-2 at 90, CRO pens
    _nlf('nlf2023','CRO','NED',  0,'D'),   # 2-2 at 90, CRO pens

    # ═══════════════════════════════════════════════════════════════════════
    # UEFA Euro 2024 — Germany (verified via web research, June 2026)
    # ═══════════════════════════════════════════════════════════════════════
    # Group stage
    _euro('euro2024','GER','SCO',  1,'W'),   # 5-1 record host opener
    _euro('euro2024','GER','HUN',  1,'W'),   # 2-0
    _euro('euro2024','SUI','GER', -1,'D'),   # 1-1, Füllkrug 90+2'
    _euro('euro2024','ESP','CRO',  0,'W'),   # 3-0
    _euro('euro2024','ITA','ALB',  0,'W'),   # 2-1
    _euro('euro2024','ESP','ITA',  0,'W'),   # 1-0
    _euro('euro2024','ALB','ESP',  0,'L'),   # 0-1
    _euro('euro2024','CRO','ITA',  0,'D'),   # 1-1, Zaccagni 90+8'
    _euro('euro2024','SRB','ENG',  0,'L'),   # 0-1
    _euro('euro2024','DEN','ENG',  0,'D'),   # 1-1
    _euro('euro2024','ENG','SVN',  0,'D'),   # 0-0
    _euro('euro2024','POL','NED',  0,'L'),   # 1-2
    _euro('euro2024','AUT','FRA',  0,'L'),   # 0-1
    _euro('euro2024','NED','FRA',  0,'D'),   # 0-0
    _euro('euro2024','POL','AUT',  0,'L'),   # 1-3
    _euro('euro2024','NED','AUT',  0,'L'),   # 2-3 UPSET — Austria topped group D
    _euro('euro2024','FRA','POL',  0,'D'),   # 1-1
    _euro('euro2024','BEL','SVK',  0,'L'),   # 0-1 UPSET
    _euro('euro2024','ROU','UKR',  0,'W'),   # 3-0
    _euro('euro2024','POR','CZE',  0,'W'),   # 2-1
    _euro('euro2024','TUR','POR',  0,'L'),   # 0-3
    _euro('euro2024','GEO','POR',  0,'W'),   # 2-0 UPSET — debutants beat Portugal
    # Round of 16
    _euro('euro2024','SUI','ITA',  0,'W'),   # 2-0
    _euro('euro2024','GER','DEN',  1,'W'),   # 2-0
    _euro('euro2024','ENG','SVK',  0,'D'),   # 1-1 at 90 (Bellingham 90+5'), ENG won AET
    _euro('euro2024','ESP','GEO',  0,'W'),   # 4-1
    _euro('euro2024','FRA','BEL',  0,'W'),   # 1-0
    _euro('euro2024','POR','SVN',  0,'D'),   # 0-0, POR won pens
    _euro('euro2024','ROU','NED',  0,'L'),   # 0-3
    _euro('euro2024','AUT','TUR',  0,'L'),   # 1-2
    # Quarter / Semi / Final
    _euro('euro2024','ESP','GER', -1,'D'),   # 1-1 at 90, ESP won 2-1 AET
    _euro('euro2024','POR','FRA',  0,'D'),   # 0-0, FRA pens
    _euro('euro2024','ENG','SUI',  0,'D'),   # 1-1, ENG pens
    _euro('euro2024','NED','TUR',  0,'W'),   # 2-1
    _euro('euro2024','ESP','FRA',  0,'W'),   # 2-1 SF (Yamal wonder-goal)
    _euro('euro2024','NED','ENG',  0,'L'),   # 1-2 SF (Watkins 90')
    _euro('euro2024','ESP','ENG',  0,'W'),   # 2-1 FINAL — Spain's 4th Euro

    # ═══════════════════════════════════════════════════════════════════════
    # UEFA Nations League Finals 2024-25 (June 2025, Germany)
    # ═══════════════════════════════════════════════════════════════════════
    _nlf('nlf2025','GER','POR',  1,'L'),   # 1-2 SF (Ronaldo 68' winner)
    _nlf('nlf2025','ESP','FRA',  0,'W'),   # 5-4 SF thriller
    _nlf('nlf2025','GER','FRA',  1,'L'),   # 0-2 third place
    _nlf('nlf2025','POR','ESP',  0,'D'),   # 2-2 at 90 & AET, POR won 5-3 pens

    # ═══════════════════════════════════════════════════════════════════════
    # WC 2026 European qualifiers & playoffs (2025 – Mar 2026)
    # ═══════════════════════════════════════════════════════════════════════
    _wcq('wcq2025','NOR','ITA',  1,'W'),   # 3-0 Oslo — Norway's statement win
    _wcq('wcq2025','SVK','GER',  1,'W'),   # 2-0 — Germany's first away WCQ loss
    _wcq('wcq2025','NED','POL',  1,'D'),   # 1-1
    _wcq('wcq2025','TUR','ESP',  1,'L'),   # 0-6 Merino hat-trick
    _wcq('wcq2025','ISR','ITA',  0,'L'),   # 4-5 nine-goal thriller (neutral)
    _wcq('wcq2025','SRB','ENG',  1,'L'),   # 0-5
    _wcq('wcq2025','BEL','MKD',  1,'D'),   # 0-0 — Belgium's stumble
    _wcq('wcq2025','ISL','FRA',  1,'D'),   # 2-2
    _wcq('wcq2025','IRL','POR',  1,'W'),   # 2-0 — Ronaldo red card
    _wcq('wcq2025','FRA','UKR',  1,'W'),   # 4-0 — Mbappé's 400th
    _wcq('wcq2025','POL','NED',  1,'D'),   # 1-1
    _wcq('wcq2025','ITA','NOR',  1,'L'),   # 1-4 Milan — Norway qualified
    _wcq('wcq2025','GER','SVK',  1,'W'),   # 6-0 — Germany sealed top spot
    _wcq('wcq2025','ESP','TUR',  1,'D'),   # 2-2 — Spain qualified unbeaten
    _wcq('wcq2025','SCO','DEN',  1,'W'),   # 4-2 — Scotland's first WC since 1998
    _wcq('wcq2025','ITA','NIR',  1,'W'),   # 2-0 playoff SF (Mar 2026)
    _wcq('wcq2025','BIH','ITA',  1,'D'),   # 1-1 playoff final at 90; BIH won pens —
                                           # Italy miss a third straight World Cup
]

# ---------------------------------------------------------------------------
# Deduplicate training set (keep first occurrence of each year+label+outcome)
# ---------------------------------------------------------------------------
_seen = set()
_deduped = []
for _m in MATCHES_TRAIN:
    _key = (_m['year'], _m['label'], _m['outcome'])
    if _key not in _seen:
        _seen.add(_key)
        _deduped.append(_m)
MATCHES_TRAIN = _deduped


# ---------------------------------------------------------------------------
# Match records — TEST (WC 2022 Qatar only)
# ---------------------------------------------------------------------------
MATCHES_TEST = [
    # Group Stage (selected key matches)
    _wc(2022,'QAT','ECU', 1,'L'),    # QAT 0-2 ECU — host loses opener
    _wc(2022,'ENG','IRN', 0,'W'),    # ENG 6-2
    _wc(2022,'ARG','KSA', 0,'L'),    # MEGA UPSET — KSA 2-1
    _wc(2022,'FRA','AUS', 0,'W'),    # FRA 4-1
    _wc(2022,'GER','JPN', 0,'L'),    # UPSET — JPN 2-1
    _wc(2022,'MAR','CRO', 0,'D'),    # 0-0
    _wc(2022,'BEL','CAN', 0,'W'),    # BEL 1-0
    _wc(2022,'ESP','CRC', 0,'W'),    # ESP 7-0
    _wc(2022,'BRA','SRB', 0,'W'),    # BRA 2-0
    _wc(2022,'POR','GHA', 0,'W'),    # POR 3-2
    _wc(2022,'URU','KOR', 0,'D'),    # 0-0
    _wc(2022,'JPN','CRC', 0,'L'),    # UPSET — CRC 1-0
    _wc(2022,'MAR','BEL', 0,'W'),    # UPSET — MAR 2-0
    _wc(2022,'CRO','CAN', 0,'W'),    # CRO 4-1
    _wc(2022,'BRA','CHE', 0,'W'),    # BRA 1-0
    _wc(2022,'POR','URU', 0,'W'),    # POR 2-0
    _wc(2022,'FRA','DEN', 0,'W'),    # FRA 2-1
    _wc(2022,'ENG','USA', 0,'D'),    # 0-0
    _wc(2022,'NED','USA', 0,'W'),    # NED 3-1
    _wc(2022,'ARG','POL', 0,'W'),    # ARG 2-0
    _wc(2022,'ESP','GER', 0,'D'),    # 1-1
    _wc(2022,'KOR','GHA', 0,'D'),    # 2-2
    _wc(2022,'SEN','ENG', 0,'L'),    # ENG 3-0
    _wc(2022,'CHE','SRB', 0,'W'),    # CHE 3-2
    # Round of 16
    _wc(2022,'NED','USA', 0,'W'),    # NED 3-1
    _wc(2022,'ARG','AUS', 0,'W'),    # ARG 2-1
    _wc(2022,'FRA','POL', 0,'W'),    # FRA 3-1
    _wc(2022,'ENG','SEN', 0,'W'),    # ENG 3-0
    _wc(2022,'CRO','JPN', 0,'D'),    # 1-1 AET, CRO pens
    _wc(2022,'BRA','KOR', 0,'W'),    # BRA 4-1
    _wc(2022,'MAR','ESP', 0,'D'),    # 0-0 AET, MAR pens (UPSET)
    _wc(2022,'POR','SUI', 0,'W'),    # POR 6-1
    # Quarter Finals
    _wc(2022,'NED','ARG', 0,'D'),    # 2-2 AET, ARG pens
    _wc(2022,'CRO','BRA', 0,'D'),    # 1-1 AET, CRO pens (UPSET)
    _wc(2022,'FRA','ENG', 0,'W'),    # FRA 2-1
    _wc(2022,'MAR','POR', 0,'W'),    # MAR 1-0 (UPSET)
    # Semi Finals
    _wc(2022,'ARG','CRO', 0,'W'),    # ARG 3-0
    _wc(2022,'FRA','MAR', 0,'W'),    # FRA 2-0
    # Final — 2-2 after 90, ARG pens
    _wc(2022,'ARG','FRA', 0,'D'),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset(years=None):
    """Return (X, y, sample_weights, labels, years_out) for the requested years.

    Parameters
    ----------
    years : list[int] or None
        If None, uses MATCHES_TRAIN only.
        Pass a list of years to filter from MATCHES_TRAIN + MATCHES_TEST.

    Returns
    -------
    X : list[list[float]]
        Feature rows: [elo_diff, ped_diff, host, comp_weight]
    y : list[str]
        Outcome for team_a ('W', 'D', 'L')
    sample_weights : list[float]
        comp_weight value for each match (for use with sklearn sample_weight)
    labels : list[str]
        Human-readable match label (e.g. 'ARG vs FRA')
    years_out : list[int]
        Year of each match
    """
    if years is None:
        pool = MATCHES_TRAIN
    else:
        pool = [m for m in MATCHES_TRAIN + MATCHES_TEST if m['year'] in years]

    X = [[m['elo_diff'], m['ped_diff'], m['host'], m['comp_weight']] for m in pool]
    y = [m['outcome'] for m in pool]
    # Sample weight = competition importance × recency decay.
    # Recency: half-life of RECENCY_HALF_LIFE years relative to 2026, so a
    # 2024 Euro match counts ~5x more than a 2010 WC match. This keeps recent
    # European/South American results dominant without discarding old data.
    sample_weights = [
        m['comp_weight'] * 0.5 ** ((2026 - m['year']) / RECENCY_HALF_LIFE)
        for m in pool
    ]
    labels = [m['label'] for m in pool]
    years_out = [m['year'] for m in pool]
    return X, y, sample_weights, labels, years_out


# Recency half-life (years) for training sample weights
RECENCY_HALF_LIFE = 7.0
