"""FBref ingest, via soccerdata.

FBref sits behind Cloudflare and plain HTTP gets a 403 challenge; soccerdata
drives an undetected browser to get through, which is why it is a dependency
rather than a convenience. Understat, the doc's other suggested source, has
restructured -- the embedded `playersData` JSON the classic scrape (and the
`understat` package) relies on is gone from the page entirely.

What we need from here is small and robust: penalty attempts per player. FPL's
`expected_goals` includes penalty xG, so a designated taker's xG share is
inflated. With PKatt we can subtract it:

    npxG = FPL_xG - PENALTY_XG * PKatt

Pulling only this keeps us well inside FBref's rate limits and avoids depending
on their wider schema, which changes.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

PENALTY_XG = 0.79   # standard xG value of a penalty


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = ["_".join(str(x) for x in c if x).strip("_") if isinstance(c, tuple)
                  else str(c) for c in df.columns]
    return df


def fetch_penalties(season: str = "2025-2026",
                    league: str = "ENG-Premier League") -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore")
    import soccerdata as sd

    fb = sd.FBref(leagues=league, seasons=season)
    sh = _flatten(fb.read_player_season_stats(stat_type="shooting"))
    name = next(c for c in sh.columns if "player" in c.lower())
    team = next(c for c in sh.columns if "team" in c.lower())
    pk = next(c for c in sh.columns if c.endswith("PK") and not c.endswith("PKatt"))
    pkatt = next(c for c in sh.columns if c.endswith("PKatt"))
    out = sh[[name, team, pk, pkatt]].copy()
    out.columns = ["fbref_name", "fbref_team", "pk_scored", "pk_att"]
    out["season"] = season.replace("2025-2026", "2025-26")
    return out


def normalise(s: str) -> str:
    """Strip accents, lowercase, collapse whitespace -- the join key."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace("-", " ").replace("'", "").split())


def resolve_to_fpl(pen: pd.DataFrame, season: str = "2025-26") -> pd.DataFrame:
    """Match FBref players to FPL `code` via normalised full name, then surname.

    Deliberately conservative: an unmatched player is left unmatched rather than
    fuzzily attached to the wrong person. A wrong penalty attribution would move
    xG share for two players at once, so silence beats a bad guess.
    """
    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    pl["full"] = (pl["first_name"].fillna("") + " " + pl["second_name"].fillna("")).map(normalise)
    pl["surname"] = pl["second_name"].fillna("").map(normalise)
    pen = pen.copy()
    pen["norm"] = pen["fbref_name"].map(normalise)

    # pass 1: exact normalised full name
    m = pen.merge(pl[["code", "full"]].rename(columns={"full": "norm"}),
                  on="norm", how="left")

    # pass 2: unique surname match for the leftovers
    miss = m["code"].isna()
    if miss.any():
        counts = pl["surname"].value_counts()
        uniq = pl[pl["surname"].isin(counts[counts == 1].index)]
        lut = dict(zip(uniq["surname"], uniq["code"]))
        # try the last token of the FBref name
        m.loc[miss, "code"] = (m.loc[miss, "norm"].str.split().str[-1].map(lut))

    # pass 3: unique first-token match (covers "Igor Thiago" -> "Thiago")
    miss = m["code"].isna()
    if miss.any():
        m.loc[miss, "code"] = (m.loc[miss, "norm"].str.split().str[0].map(lut))

    # pass 4: the manual override table -- the irreducible residue.
    ov = manual_overrides()
    if ov:
        miss = m["code"].isna()
        m.loc[miss, "code"] = m.loc[miss, "fbref_name"].map(ov)
    return m


def manual_overrides(path: Path = Path("config/manual_overrides.csv")) -> dict:
    if not path.exists():
        return {}
    ov = pd.read_csv(path, comment="#")
    ov = ov[ov["source"] == "fbref"]
    return dict(zip(ov["fbref_name"], ov["fpl_code"]))


def build(out: Path = Path("data/raw/fbref")) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    pen = fetch_penalties()
    res = resolve_to_fpl(pen)
    res.to_parquet(out / "penalties_2025_26.parquet", index=False)
    return res


if __name__ == "__main__":
    r = build()
    takers = r[r.pk_att > 0]
    matched = takers.code.notna()
    print(f"FBref players: {len(r)}   penalty takers: {len(takers)}")
    print(f"  resolved to FPL: {matched.sum()}/{len(takers)} ({matched.mean():.1%})")
    print(f"  penalty attempts covered: {int(takers[matched].pk_att.sum())}"
          f"/{int(takers.pk_att.sum())}")
    un = takers[~matched]
    if len(un):
        print(f"\n  UNRESOLVED (manual override needed): "
              f"{', '.join(un.fbref_name + ' (' + un.pk_att.astype(int).astype(str) + ')')}")
