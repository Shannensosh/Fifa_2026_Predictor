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

# code: (name, flag, confederation, fifa_rank, elo, titles, wc_appearances,
#        squad_value_m, [key players], form)
TEAMS = {
    "ARG": ("Argentina",   "🇦🇷", "CONMEBOL", 1,  2105, 3, 18, 1180, ["Lautaro Martínez", "Julián Álvarez", "Enzo Fernández"], "WWWDWWLWWW"),
    "FRA": ("France",       "🇫🇷", "UEFA",     2,  2085, 2, 16, 1300, ["Kylian Mbappé", "Ousmane Dembélé", "Aurélien Tchouaméni"], "WWLWWWDWWW"),
    "SPA": ("Spain",        "🇪🇸", "UEFA",     3,  2060, 1, 16, 1150, ["Lamine Yamal", "Rodri", "Nico Williams"], "WWWWDWWWWL"),
    "BRA": ("Brazil",       "🇧🇷", "CONMEBOL", 4,  2040, 5, 22, 1100, ["Vinícius Jr", "Rodrygo", "Raphinha"], "WLWWDWWLWW"),
    "ENG": ("England",      "🏴",  "UEFA",     5,  2015, 1, 16, 1250, ["Jude Bellingham", "Harry Kane", "Bukayo Saka"], "WWDWWLWDWW"),
    "POR": ("Portugal",     "🇵🇹", "UEFA",     6,  2005, 0, 8,  1080, ["Cristiano Ronaldo", "Bruno Fernandes", "Rafael Leão"], "WWWLWWDWWW"),
    "NED": ("Netherlands",  "🇳🇱", "UEFA",     7,  1985, 0, 11, 980,  ["Virgil van Dijk", "Cody Gakpo", "Frenkie de Jong"], "WDWWLWWDWW"),
    "BEL": ("Belgium",      "🇧🇪", "UEFA",     8,  1955, 0, 14, 820,  ["Kevin De Bruyne", "Jérémy Doku", "Romelu Lukaku"], "WLWWDWLWWD"),
    "GER": ("Germany",      "🇩🇪", "UEFA",     9,  1965, 4, 20, 1000, ["Jamal Musiala", "Florian Wirtz", "Kai Havertz"], "WWLWWDWWLW"),
    "ITA": ("Italy",        "🇮🇹", "UEFA",     10, 1945, 4, 18, 760,  ["Gianluigi Donnarumma", "Federico Chiesa", "Nicolò Barella"], "WDWLWWWDLW"),
    "CRO": ("Croatia",      "🇭🇷", "UEFA",     11, 1925, 0, 6,  520,  ["Luka Modrić", "Joško Gvardiol", "Mateo Kovačić"], "WLWWDWLWWD"),
    "URU": ("Uruguay",      "🇺🇾", "CONMEBOL", 12, 1930, 2, 14, 560,  ["Federico Valverde", "Darwin Núñez", "Ronald Araújo"], "WWDWLWWWDL"),
    "COL": ("Colombia",     "🇨🇴", "CONMEBOL", 13, 1915, 0, 6,  490,  ["Luis Díaz", "James Rodríguez", "Jhon Durán"], "WWDLWWDWWL"),
    "MAR": ("Morocco",      "🇲🇦", "CAF",      14, 1900, 0, 6,  420,  ["Achraf Hakimi", "Brahim Díaz", "Hakim Ziyech"], "WWWDLWWWDW"),
    "USA": ("USA",          "🇺🇸", "CONCACAF", 15, 1860, 0, 11, 380,  ["Christian Pulisic", "Weston McKennie", "Gio Reyna"], "WWDLWWDLWW"),
    "MEX": ("Mexico",       "🇲🇽", "CONCACAF", 16, 1840, 0, 17, 300,  ["Santiago Giménez", "Edson Álvarez", "Hirving Lozano"], "WDWWLDWWLW"),
    "SUI": ("Switzerland",  "🇨🇭", "UEFA",     17, 1875, 0, 12, 420,  ["Granit Xhaka", "Manuel Akanji", "Dan Ndoye"], "WDWLWWDWLW"),
    "DEN": ("Denmark",      "🇩🇰", "UEFA",     18, 1880, 0, 6,  450,  ["Rasmus Højlund", "Pierre-Emile Højbjerg", "Christian Eriksen"], "WWLWDWWLWD"),
    "JPN": ("Japan",        "🇯🇵", "AFC",      19, 1870, 0, 7,  340,  ["Takefusa Kubo", "Kaoru Mitoma", "Wataru Endo"], "WWWDWWLWWD"),
    "SEN": ("Senegal",      "🇸🇳", "CAF",      20, 1865, 0, 4,  380,  ["Sadio Mané", "Nicolas Jackson", "Pape Sarr"], "WWDLWWWDLW"),
    "IRN": ("Iran",         "🇮🇷", "AFC",      21, 1810, 0, 6,  120,  ["Mehdi Taremi", "Alireza Jahanbakhsh", "Sardar Azmoun"], "WWDWLWDWWL"),
    "KOR": ("South Korea",  "🇰🇷", "AFC",      22, 1840, 0, 11, 280,  ["Son Heung-min", "Lee Kang-in", "Kim Min-jae"], "WDWWLWDWWL"),
    "AUS": ("Australia",    "🇦🇺", "AFC",      23, 1755, 0, 6,  110,  ["Mathew Ryan", "Jackson Irvine", "Riley McGree"], "WDLWWDWLWD"),
    "SRB": ("Serbia",       "🇷🇸", "UEFA",     24, 1820, 0, 13, 480,  ["Dušan Vlahović", "Aleksandar Mitrović", "Sergej Milinković-Savić"], "DWLWWDLWWD"),
    "ECU": ("Ecuador",      "🇪🇨", "CONMEBOL", 25, 1825, 0, 4,  330,  ["Moisés Caicedo", "Pervis Estupiñán", "Kendry Páez"], "WDWWLDWWDW"),
    "AUT": ("Austria",      "🇦🇹", "UEFA",     26, 1830, 0, 8,  430,  ["David Alaba", "Marcel Sabitzer", "Konrad Laimer"], "WWDWLWWDWL"),
    "UKR": ("Ukraine",      "🇺🇦", "UEFA",     27, 1800, 0, 1,  360,  ["Mykhailo Mudryk", "Artem Dovbyk", "Oleksandr Zinchenko"], "WDLWWDLWDW"),
    "TUR": ("Turkey",       "🇹🇷", "UEFA",     28, 1810, 0, 2,  490,  ["Arda Güler", "Hakan Çalhanoğlu", "Kenan Yıldız"], "WWLWDWWLWD"),
    "NGA": ("Nigeria",      "🇳🇬", "CAF",      29, 1790, 0, 6,  340,  ["Victor Osimhen", "Ademola Lookman", "Alex Iwobi"], "WDWWLWDWLW"),
    "CAN": ("Canada",       "🇨🇦", "CONCACAF", 30, 1780, 0, 2,  260,  ["Alphonso Davies", "Jonathan David", "Stephen Eustáquio"], "WDLWWDWWLD"),
    "EGY": ("Egypt",        "🇪🇬", "CAF",      31, 1770, 0, 3,  220,  ["Mohamed Salah", "Omar Marmoush", "Mohamed Elneny"], "WWDWLWWDLW"),
    "NOR": ("Norway",       "🇳🇴", "UEFA",     32, 1815, 0, 3,  520,  ["Erling Haaland", "Martin Ødegaard", "Alexander Sørloth"], "WWWDLWWWDL"),
    "POL": ("Poland",       "🇵🇱", "UEFA",     33, 1760, 0, 9,  300,  ["Robert Lewandowski", "Piotr Zieliński", "Nicola Zalewski"], "WDLWDWLWDW"),
    "SWE": ("Sweden",       "🇸🇪", "UEFA",     34, 1755, 0, 12, 360,  ["Alexander Isak", "Viktor Gyökeres", "Dejan Kulusevski"], "WWDLWWDLWW"),
    "WAL": ("Wales",        "🏴",  "UEFA",     35, 1730, 0, 2,  190,  ["Harry Wilson", "Brennan Johnson", "Daniel James"], "DWLDWWLWDL"),
    "CIV": ("Ivory Coast",  "🇨🇮", "CAF",      36, 1745, 0, 3,  280,  ["Sébastien Haller", "Franck Kessié", "Simon Adingra"], "WWDLWWDWLD"),
    "ALG": ("Algeria",      "🇩🇿", "CAF",      37, 1740, 0, 4,  260,  ["Riyad Mahrez", "Ismaël Bennacer", "Saïd Benrahma"], "WDWWLDWWLW"),
    "PAR": ("Paraguay",     "🇵🇾", "CONMEBOL", 38, 1735, 0, 8,  190,  ["Miguel Almirón", "Julio Enciso", "Antonio Sanabria"], "DWWLDWLWWD"),
    "KSA": ("Saudi Arabia", "🇸🇦", "AFC",      39, 1700, 0, 6,  90,   ["Salem Al-Dawsari", "Firas Al-Buraikan", "Mohammed Kanno"], "DWLWDLWDWL"),
    "QAT": ("Qatar",        "🇶🇦", "AFC",      40, 1690, 0, 1,  80,   ["Akram Afif", "Almoez Ali", "Hassan Al-Haydos"], "WDWLWDLWWD"),
    "PAN": ("Panama",       "🇵🇦", "CONCACAF", 41, 1710, 0, 1,  70,   ["Adalberto Carrasquilla", "Ismael Díaz", "Michael Murillo"], "WWDLWDWLWD"),
    "CRC": ("Costa Rica",   "🇨🇷", "CONCACAF", 42, 1680, 0, 6,  60,   ["Keylor Navas", "Manfred Ugalde", "Joel Campbell"], "DWLWDLWDWL"),
    "GHA": ("Ghana",        "🇬🇭", "CAF",      43, 1700, 0, 4,  250,  ["Mohammed Kudus", "Thomas Partey", "Iñaki Williams"], "WLDWWLDWLW"),
    "CMR": ("Cameroon",     "🇨🇲", "CAF",      44, 1715, 0, 8,  230,  ["André Onana", "Bryan Mbeumo", "Vincent Aboubakar"], "WDLWWDLWDW"),
    "TUN": ("Tunisia",      "🇹🇳", "CAF",      45, 1695, 0, 6,  120,  ["Hannibal Mejbri", "Aïssa Laïdouni", "Ellyes Skhiri"], "DWLWDWLDWL"),
    "UZB": ("Uzbekistan",   "🇺🇿", "AFC",      46, 1690, 0, 0,  90,   ["Eldor Shomurodov", "Abbosbek Fayzullaev", "Jaloliddin Masharipov"], "WWDLWDWLWD"),
    "NZL": ("New Zealand",  "🇳🇿", "OFC",      47, 1610, 0, 2,  40,   ["Chris Wood", "Marko Stamenić", "Tim Payne"], "WDWLWDLWWD"),
    "JAM": ("Jamaica",      "🇯🇲", "CONCACAF", 48, 1660, 0, 1,  130,  ["Leon Bailey", "Michail Antonio", "Demarai Gray"], "DWLDWLWDWL"),
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


def normalize_form(form_str):
    """Return points from last-10 form string (W=3, D=1, L=0), max 30."""
    pts = {"W": 3, "D": 1, "L": 0}
    return sum(pts[c] for c in form_str)


def get_teams():
    """Return list of team dicts with all parameters resolved."""
    out = []
    for code, t in TEAMS.items():
        name, flag, conf, rank, elo, titles, apps, value, players, form = t
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
        })
    return out
