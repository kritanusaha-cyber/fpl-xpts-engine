"""Player name resolution across sources.

Name matching carries most of the risk in a project like this. A miss loses a
player's data silently; a false match attaches one player's record to another,
which is worse. Both happened here before this module existed: 37% of projected
starters had no territory data, and a Manchester United defender inherited an
Aston Villa goalkeeper's save record.

Three things fix the bulk of it.

CHARACTERS THAT DO NOT DECOMPOSE. Unicode normalisation strips combining accents,
so "Muñoz" reduces to "munoz". It does nothing for letters that are their own
codepoint: Turkish dotless i, the g-breve, the Scandinavian slashed o, the Polish
crossed l. "Kadioglu" and "Kadioglu" therefore never matched, because one of them
still held a dotless i. These are mapped explicitly.

CLUB AS A DISAMBIGUATOR. Names collide; a name at a club rarely does. The source
carries a team id and FPL carries a club, so the mapping between them is derived
from the players who match unambiguously and then used to resolve the rest. This
is what makes a surname match safe.

POSITION AS A CONSTRAINT. A goalkeeper's record can only belong to a goalkeeper.
Passing the expected position removes a whole class of collision.
"""

from __future__ import annotations

import unicodedata

import pandas as pd

# Letters that survive NFKD because they are distinct codepoints, not
# base-plus-accent. Without these the Turkish, Polish and Nordic names never match.
CHARMAP = str.maketrans({
    "ı": "i", "İ": "i",          # dotless i, dotted capital I
    "ğ": "g", "Ğ": "g",          # g-breve
    "ş": "s", "Ş": "s",          # s-cedilla
    "ø": "o", "Ø": "o",          # o-slash
    "ł": "l", "Ł": "l",          # l-stroke
    "đ": "d", "Đ": "d",          # d-stroke
    "ß": "ss",                          # sharp s
    "æ": "ae", "œ": "oe",
    "'": "", "’": "", "-": " ", ".": " ",
})


def normalise(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.translate(CHARMAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def fpl_directory(season: str = "2025-26") -> pd.DataFrame:
    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    d = pl[["code", "first_name", "second_name", "web_name", "element_type", "team_code"]].copy()
    d["full"] = (d.first_name.fillna("") + " " + d.second_name.fillna("")).map(normalise)
    d["surname"] = d.second_name.fillna("").map(normalise)
    d["firstname"] = d.first_name.fillna("").map(normalise)
    d["web"] = d.web_name.fillna("").map(normalise)
    d["tokens"] = d.full.str.split().apply(set)
    return d


def _club_bridge(matched: pd.DataFrame, directory: pd.DataFrame) -> dict:
    """source team id -> FPL team_code, learned from unambiguous matches."""
    if matched.empty:
        return {}
    m = matched.merge(directory[["code", "team_code"]], on="code", how="left")
    m = m.dropna(subset=["team_id", "team_code"])
    if m.empty:
        return {}
    return (m.groupby("team_id")["team_code"]
              .agg(lambda x: x.value_counts().idxmax()).to_dict())


def resolve(df: pd.DataFrame, name_col: str = "player_name",
            team_col: str | None = "team_id", element_type: int | None = None,
            season: str = "2025-26", overrides: dict | None = None) -> pd.DataFrame:
    """Attach FPL `code`. Unmatched rows keep a null code rather than a guess."""
    d = fpl_directory(season)
    if element_type is not None:
        d = d[d.element_type == element_type]

    out = df.copy()
    out["norm"] = out[name_col].map(normalise)
    out["code"] = pd.NA

    # 1. exact full name, then web name -- unambiguous, no club needed
    for col in ("full", "web"):
        lut = d.drop_duplicates(col).set_index(col)["code"]
        miss = out.code.isna()
        out.loc[miss, "code"] = out.loc[miss, "norm"].map(lut)

    # 2. learn the club bridge from what matched, then use it
    bridge = _club_bridge(out.dropna(subset=["code"]), d) if team_col in out.columns else {}
    if bridge:
        out["_club"] = out[team_col].map(bridge)
        for _, cand in d.groupby("team_code"):
            pass
        miss = out.code.isna() & out["_club"].notna()
        for idx in out.index[miss]:
            club = out.at[idx, "_club"]
            toks = set(out.at[idx, "norm"].split())
            pool = d[d.team_code == club]
            hits = pool[pool.tokens.apply(lambda t: bool(t & toks))]
            if len(hits) == 1:
                out.at[idx, "code"] = hits.iloc[0]["code"]

    # 3. unique surname, then unique first name -- league-wide, so require uniqueness
    for col in ("surname", "firstname"):
        counts = d[col].value_counts()
        lut = d[d[col].isin(counts[counts == 1].index)].set_index(col)["code"]
        miss = out.code.isna()
        out.loc[miss, "code"] = out.loc[miss, "norm"].str.split().str[-1].map(lut)
        miss = out.code.isna()
        out.loc[miss, "code"] = out.loc[miss, "norm"].str.split().str[0].map(lut)

    # 4. the irreducible residue
    if overrides:
        miss = out.code.isna()
        out.loc[miss, "code"] = out.loc[miss, name_col].map(overrides)
    return out.drop(columns=[c for c in ("_club",) if c in out.columns])
