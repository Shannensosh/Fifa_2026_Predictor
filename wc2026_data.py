"""
FIFA World Cup 2026 — Curated team dataset.

IMPORTANT / DISCLAIMER
----------------------
These are *curated estimates* assembled for a transparent, self-contained
prediction model. They are NOT an official feed. Ratings are approximate and
in the spirit of public Elo / FIFA-ranking / market-value figures. The 48-team
field is a plausible illustrative bracket (some teams shown may not qualify in
reality, and the group draw is seeded by rating rather than the official draw).

Each team carries the parameters the model uses:
    fifa_rank      : approximate FIFA world ranking position
    elo            : approximate world-football Elo rating
    titles         : World Cup titles won (historical wins)
    wc_appearances : number of World Cup tournaments appeared in (pedigree)
    squad_value_m  : approximate total squad market value, EUR millions
    key_players    : a few headline players (squad-strength signal)
    form           : last 10 competitive results, newest first ("W"/"D"/"L")
    host           : True for the three 2026 hosts (USA, Mexico, Canada)
"""

HOSTS = {"USA", "MEX", "CAN"}

# ---------------------------------------------------------------------------
# OFFICIAL 2026 World Cup group draw (held 5 Dec 2025, Washington D.C.).
# Verified June 2026 against Wikipedia + NBC Sports (cross-matched team-for-team).
# Hosts seeded as group heads: MEX→A, CAN→B, USA→D.
# Codes use this module's internal convention (SPA = Spain).
# ---------------------------------------------------------------------------
OFFICIAL_GROUPS = {
    "A": ["MEX", "RSA", "KOR", "CZE"],
    "B": ["CAN", "BIH", "QAT", "SUI"],
    "C": ["BRA", "MAR", "HAI", "SCO"],
    "D": ["USA", "PAR", "AUS", "TUR"],
    "E": ["GER", "CUW", "CIV", "ECU"],
    "F": ["NED", "JPN", "SWE", "TUN"],
    "G": ["BEL", "EGY", "IRN", "NZL"],
    "H": ["SPA", "CPV", "KSA", "URU"],
    "I": ["FRA", "SEN", "IRQ", "NOR"],
    "J": ["ARG", "ALG", "AUT", "JOR"],
    "K": ["POR", "COD", "UZB", "COL"],
    "L": ["ENG", "CRO", "GHA", "PAN"],
}

# code: (name, flag, confederation, fifa_rank, elo, titles, wc_appearances,
#        squad_value_m, [key players], form)
TEAMS = {
    "ARG": ("Argentina",   "🇦🇷", "CONMEBOL", 1,  2105, 3, 18, 1180, ["Lionel Messi", "Lautaro Martínez", "Julián Álvarez"], "WWWWWWWLWD"),
    "FRA": ("France",       "🇫🇷", "UEFA",     2,  2085, 2, 16, 1300, ["Kylian Mbappé", "Ousmane Dembélé", "Michael Olise"], "WLWWWWDWWW"),
    "SPA": ("Spain",        "🇪🇸", "UEFA",     3,  2070, 1, 16, 1150, ["Lamine Yamal", "Rodri", "Nico Williams"], "WDDWDWWWWW"),
    "BRA": ("Brazil",       "🇧🇷", "CONMEBOL", 4,  2010, 5, 22, 1100, ["Vinícius Jr", "Raphinha", "Matheus Cunha"], "WWWLLWLWWD"),
    "ENG": ("England",      "🏴",  "UEFA",     5,  2030, 1, 16, 1250, ["Jude Bellingham", "Harry Kane", "Bukayo Saka"], "WWLDWWWWWW"),
    "POR": ("Portugal",     "🇵🇹", "UEFA",     6,  2015, 0, 8,  1080, ["Cristiano Ronaldo", "Bruno Fernandes", "Rafael Leão"], "WWWDWLDWWW"),
    "NED": ("Netherlands",  "🇳🇱", "UEFA",     7,  1975, 0, 11, 980,  ["Virgil van Dijk", "Cody Gakpo", "Frenkie de Jong"], "WLDWWDWWWD"),
    "BEL": ("Belgium",      "🇧🇪", "UEFA",     8,  1965, 0, 14, 820,  ["Kevin De Bruyne", "Jérémy Doku", "Romelu Lukaku"], "WWDWWDWDWW"),
    "GER": ("Germany",      "🇩🇪", "UEFA",     9,  1975, 4, 20, 1000, ["Jamal Musiala", "Florian Wirtz", "Kai Havertz"], "WWWWWWLWWL"),
    "CRO": ("Croatia",      "🇭🇷", "UEFA",     11, 1925, 0, 6,  520,  ["Luka Modrić", "Joško Gvardiol", "Mateo Kovačić"], "WLLWWWWDWW"),
    "URU": ("Uruguay",      "🇺🇾", "CONMEBOL", 12, 1900, 2, 14, 560,  ["Federico Valverde", "Darwin Núñez", "Ronald Araújo"], "DDDLDWWDWW"),
    "COL": ("Colombia",     "🇨🇴", "CONMEBOL", 13, 1915, 0, 6,  490,  ["Luis Díaz", "James Rodríguez", "Jhon Durán"], "WWLLWWDWWW"),
    "MAR": ("Morocco",      "🇲🇦", "CAF",      14, 1920, 0, 6,  420,  ["Achraf Hakimi", "Brahim Díaz", "Azzedine Ounahi"], "DDLDWWWDWW"),
    "USA": ("USA",          "🇺🇸", "CONCACAF", 15, 1845, 0, 11, 380,  ["Christian Pulisic", "Weston McKennie", "Gio Reyna"], "LWLLWWWDWL"),
    "MEX": ("Mexico",       "🇲🇽", "CONCACAF", 16, 1860, 0, 17, 300,  ["Santiago Giménez", "Edson Álvarez", "Hirving Lozano"], "WWWDDWWWLD"),
    "SUI": ("Switzerland",  "🇨🇭", "UEFA",     17, 1875, 0, 12, 420,  ["Granit Xhaka", "Manuel Akanji", "Dan Ndoye"], "WDWLWWDWLW"),
    "CZE": ("Czechia",      "🇨🇿", "UEFA",     18, 1820, 0, 9,  340,  ["Patrik Schick", "Tomáš Souček", "Adam Hložek"], "WWDLWWDWLW"),
    "JPN": ("Japan",        "🇯🇵", "AFC",      19, 1890, 0, 7,  340,  ["Takefusa Kubo", "Ritsu Doan", "Wataru Endo"], "WWWWWWDLDW"),
    "SEN": ("Senegal",      "🇸🇳", "CAF",      20, 1865, 0, 4,  380,  ["Sadio Mané", "Nicolas Jackson", "Pape Sarr"], "WWDLWWWDLW"),
    "IRN": ("Iran",         "🇮🇷", "AFC",      21, 1810, 0, 6,  120,  ["Mehdi Taremi", "Alireza Jahanbakhsh", "Sardar Azmoun"], "WWDWLWDWWL"),
    "KOR": ("South Korea",  "🇰🇷", "AFC",      22, 1840, 0, 11, 280,  ["Son Heung-min", "Lee Kang-in", "Kim Min-jae"], "WDWWLWDWWL"),
    "AUS": ("Australia",    "🇦🇺", "AFC",      23, 1755, 0, 6,  110,  ["Mathew Ryan", "Jackson Irvine", "Riley McGree"], "WDLWWDWLWD"),
    "RSA": ("South Africa", "🇿🇦", "CAF",      24, 1745, 0, 3,  90,   ["Percy Tau", "Lyle Foster", "Ronwen Williams"], "WWDWLWDWWD"),
    "ECU": ("Ecuador",      "🇪🇨", "CONMEBOL", 25, 1825, 0, 4,  330,  ["Moisés Caicedo", "Pervis Estupiñán", "Kendry Páez"], "WDWWLDWWDW"),
    "AUT": ("Austria",      "🇦🇹", "UEFA",     26, 1830, 0, 8,  430,  ["David Alaba", "Marcel Sabitzer", "Konrad Laimer"], "WWDWLWWDWL"),
    "IRQ": ("Iraq",         "🇮🇶", "AFC",      27, 1750, 0, 1,  45,   ["Aymen Hussein", "Ali Jasim", "Zidane Iqbal"], "WWDWWLDWLW"),
    "TUR": ("Turkey",       "🇹🇷", "UEFA",     28, 1810, 0, 2,  490,  ["Arda Güler", "Hakan Çalhanoğlu", "Kenan Yıldız"], "WWLWDWWLWD"),
    "JOR": ("Jordan",       "🇯🇴", "AFC",      29, 1715, 0, 0,  28,   ["Mousa Al-Tamari", "Yazan Al-Naimat", "Nizar Al-Rashdan"], "WWDWLWWDWL"),
    "CAN": ("Canada",       "🇨🇦", "CONCACAF", 30, 1780, 0, 2,  260,  ["Alphonso Davies", "Jonathan David", "Stephen Eustáquio"], "WDLWWDWWLD"),
    "EGY": ("Egypt",        "🇪🇬", "CAF",      31, 1770, 0, 3,  220,  ["Mohamed Salah", "Omar Marmoush", "Mohamed Elneny"], "WWDWLWWDLW"),
    "NOR": ("Norway",       "🇳🇴", "UEFA",     32, 1855, 0, 4,  520,  ["Erling Haaland", "Martin Ødegaard", "Alexander Sørloth"], "DWLWWWWWWW"),
    "HAI": ("Haiti",        "🇭🇹", "CONCACAF", 33, 1655, 0, 1,  35,   ["Frantzdy Pierrot", "Duckens Nazon", "Danley Jean Jacques"], "WLDWWLDWLW"),
    "SWE": ("Sweden",       "🇸🇪", "UEFA",     34, 1755, 0, 12, 360,  ["Alexander Isak", "Viktor Gyökeres", "Dejan Kulusevski"], "WWDLWWDLWW"),
    "SCO": ("Scotland",     "🏴", "UEFA",    35, 1845, 0, 9,  350,  ["Scott McTominay", "Andrew Robertson", "John McGinn"], "WWDWWLWDWW"),
    "BIH": ("Bosnia-Herz.", "🇧🇦", "UEFA",     36, 1820, 0, 2,  180,  ["Edin Džeko", "Ermedin Demirović", "Anel Ahmedhodžić"], "DWWDWLWDWD"),
    "CIV": ("Ivory Coast",  "🇨🇮", "CAF",      36, 1745, 0, 3,  280,  ["Sébastien Haller", "Franck Kessié", "Simon Adingra"], "WWDLWWDWLD"),
    "ALG": ("Algeria",      "🇩🇿", "CAF",      37, 1740, 0, 4,  260,  ["Riyad Mahrez", "Ismaël Bennacer", "Saïd Benrahma"], "WDWWLDWWLW"),
    "PAR": ("Paraguay",     "🇵🇾", "CONMEBOL", 38, 1735, 0, 8,  190,  ["Miguel Almirón", "Julio Enciso", "Antonio Sanabria"], "DWWLDWLWWD"),
    "KSA": ("Saudi Arabia", "🇸🇦", "AFC",      39, 1700, 0, 6,  90,   ["Salem Al-Dawsari", "Firas Al-Buraikan", "Mohammed Kanno"], "DWLWDLWDWL"),
    "QAT": ("Qatar",        "🇶🇦", "AFC",      40, 1690, 0, 1,  80,   ["Akram Afif", "Almoez Ali", "Hassan Al-Haydos"], "WDWLWDLWWD"),
    "PAN": ("Panama",       "🇵🇦", "CONCACAF", 41, 1710, 0, 1,  70,   ["Adalberto Carrasquilla", "Ismael Díaz", "Michael Murillo"], "WWDLWDWLWD"),
    "CUW": ("Curaçao",      "🇨🇼", "CONCACAF", 42, 1640, 0, 0,  22,   ["Tahith Chong", "Juninho Bacuna", "Leandro Bacuna"], "WDWLWDWLWD"),
    "GHA": ("Ghana",        "🇬🇭", "CAF",      43, 1700, 0, 4,  250,  ["Mohammed Kudus", "Thomas Partey", "Iñaki Williams"], "WLDWWLDWLW"),
    "COD": ("DR Congo",     "🇨🇩", "CAF",      44, 1745, 0, 1,  180,  ["Yoane Wissa", "Cédric Bakambu", "Chancel Mbemba"], "WWDWLWDWWL"),
    "TUN": ("Tunisia",      "🇹🇳", "CAF",      45, 1695, 0, 6,  120,  ["Hannibal Mejbri", "Aïssa Laïdouni", "Ellyes Skhiri"], "DWLWDWLDWL"),
    "UZB": ("Uzbekistan",   "🇺🇿", "AFC",      46, 1690, 0, 0,  90,   ["Eldor Shomurodov", "Abbosbek Fayzullaev", "Jaloliddin Masharipov"], "WWDLWDWLWD"),
    "NZL": ("New Zealand",  "🇳🇿", "OFC",      47, 1610, 0, 2,  40,   ["Chris Wood", "Marko Stamenić", "Tim Payne"], "WDWLWDLWWD"),
    "CPV": ("Cape Verde",   "🇨🇻", "CAF",      48, 1690, 0, 0,  45,   ["Ryan Mendes", "Garry Rodrigues", "Jovane Cabral"], "WWDWLWWDLW"),
}

# Notable head-to-head records (all-time, competitive + friendly, approximate),
# keyed (A, B) -> (A_wins, draws, B_wins). The model applies a small nudge from
# these where available. Only a selection of headline rivalries are encoded.
HEAD_TO_HEAD = {
    ("ARG", "BRA"): (43, 26, 41),
    ("ARG", "FRA"): (3, 4, 3),
    ("BRA", "FRA"): (5, 4, 6),
    ("GER", "ITA"): (8, 10, 5),
    ("GER", "ENG"): (16, 7, 14),
    ("ENG", "FRA"): (17, 5, 9),
    ("ESP", "ITA"): (12, 14, 12),  # placeholder, ESP keyed below as SPA
    ("SPA", "ITA"): (12, 14, 12),
    ("SPA", "POR"): (17, 13, 6),
    ("NED", "GER"): (11, 16, 17),
    ("POR", "FRA"): (6, 2, 19),
    ("ARG", "ENG"): (7, 6, 6),
    ("BRA", "GER"): (13, 5, 6),
    ("URU", "ARG"): (58, 47, 95),
    ("MEX", "USA"): (37, 18, 22),
    ("KOR", "JPN"): (42, 23, 16),
}


# ---------------------------------------------------------------------------
# Injury / availability status going into the June 2026 tournament.
# code -> {"out": [(player, importance)], "doubtful": [(player, importance)],
#          "availability": 0-100}
# availability = % of the first-choice XI fully fit (100 = everyone fit).
# Teams not listed default to 100 (no notable injury news found).
# Populated from injury-news research; importance: "star" | "key" | "squad".
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Average age of the likely starting XI going into June 2026.
# Used to derive a "trajectory" score: younger/rising squads are rewarded,
# aging squads penalised. Teams not listed default to DEFAULT_AGE.
# Estimates from public squad reporting (approximate, starting XI).
# ---------------------------------------------------------------------------
SQUAD_AGE = {
    "SPA": 25.3,  # youngest elite core (Yamal, Cubarsí, Pedri)
    "ENG": 26.1,  # young, ascending (Bellingham, Saka, Mainoo)
    "GER": 26.4,  # rebuilt around Wirtz / Musiala
    "NOR": 26.5,  # Haaland / Ødegaard generation
    "USA": 26.6,
    "FRA": 27.0,  # prime window, very deep
    "JPN": 27.1,
    "MAR": 27.2,
    "ECU": 26.0,
    "URU": 27.3,  # refreshed post-Suárez/Cavani
    "NED": 27.6,
    "BRA": 27.6,
    "MEX": 28.1,
    "POR": 28.4,  # young talent but Ronaldo (41) inflates the core
    "BEL": 28.6,  # De Bruyne / Lukaku ageing
    "COL": 28.7,  # James Rodríguez (34)
    "ARG": 29.6,  # ageing champions — Messi (38), Otamendi, De Paul
    "CRO": 30.1,  # oldest core — Modrić (40), Perišić
}

DEFAULT_AGE = 27.6

INJURIES = {
    # Researched June 2026 (web sources; availability = % of first-XI fully fit)
    "ARG": {"availability": 88,
            "out": [("Leonardo Balerdi", "squad"), ("Juan Foyth", "squad")],
            "doubtful": [("Nahuel Molina", "key")]},
    "FRA": {"availability": 88,
            "out": [("Hugo Ekitike", "key"), ("Eduardo Camavinga", "squad"),
                    ("Boubacar Kamara", "squad")],
            "doubtful": [("William Saliba", "key")]},
    "SPA": {"availability": 78,
            "out": [("Fermín López", "key"), ("Dani Carvajal", "key")],
            "doubtful": [("Lamine Yamal", "star"), ("Mikel Merino", "key")]},
    "BRA": {"availability": 72,
            "out": [("Rodrygo", "star"), ("Éder Militão", "key"), ("Estêvão", "star")],
            "doubtful": [("Neymar", "key")]},
    "ENG": {"availability": 90,
            "out": [("Jack Grealish", "squad"), ("Tino Livramento", "squad")],
            "doubtful": [("Reece James", "key")]},
    "POR": {"availability": 100, "out": [], "doubtful": []},
    "GER": {"availability": 85,
            "out": [("Serge Gnabry", "key"), ("Lennart Karl", "key")],
            "doubtful": [("Manuel Neuer", "key")]},
    "NED": {"availability": 68,
            "out": [("Jurriën Timber", "key"), ("Xavi Simons", "key"),
                    ("Jerdy Schouten", "key"), ("Matthijs de Ligt", "key")],
            "doubtful": []},
    "BEL": {"availability": 85, "out": [],
            "doubtful": [("Romelu Lukaku", "key")]},
    "CRO": {"availability": 92, "out": [], "doubtful": []},
    "URU": {"availability": 78, "out": [],
            "doubtful": [("José María Giménez", "key"),
                         ("Giorgian de Arrascaeta", "key"),
                         ("Darwin Núñez", "key")]},
    "MAR": {"availability": 95, "out": [], "doubtful": []},
    "USA": {"availability": 80,
            "out": [("Johnny Cardoso", "key"), ("Patrick Agyemang", "squad")],
            "doubtful": [("Chris Richards", "key")]},
    "MEX": {"availability": 80,
            "out": [("Luis Ángel Malagón", "key"), ("Marcel Ruiz", "squad")],
            "doubtful": [("Edson Álvarez", "key")]},
    "JPN": {"availability": 80,
            "out": [("Kaoru Mitoma", "star"), ("Takumi Minamino", "key")],
            "doubtful": []},
    "COL": {"availability": 97, "out": [], "doubtful": []},
    "NOR": {"availability": 93, "out": [], "doubtful": []},
}

DEFAULT_AVAILABILITY = 100


def normalize_form(form_str):
    """Return points from last-10 form string (W=3, D=1, L=0), max 30."""
    pts = {"W": 3, "D": 1, "L": 0}
    return sum(pts[c] for c in form_str)


def get_teams():
    """Return list of team dicts with all parameters resolved."""
    out = []
    for code, t in TEAMS.items():
        name, flag, conf, rank, elo, titles, apps, value, players, form = t
        inj = INJURIES.get(code, {})
        out.append({
            "code": code,
            "name": name,
            "flag": flag,
            "confederation": conf,
            "fifa_rank": rank,
            "elo": elo,
            "titles": titles,
            "wc_appearances": apps,
            "squad_value_m": value,
            "key_players": players,
            "form": form,
            "form_points": normalize_form(form),
            "host": code in HOSTS,
            "availability": inj.get("availability", DEFAULT_AVAILABILITY),
            "injuries_out": inj.get("out", []),
            "injuries_doubtful": inj.get("doubtful", []),
            "avg_age": SQUAD_AGE.get(code, DEFAULT_AGE),
        })
    return out
