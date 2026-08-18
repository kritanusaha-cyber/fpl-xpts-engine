"""Foreign-league output for players arriving in the Premier League.

A new signing is not an unknown quantity -- he has a record, just not in this
league. This pulls per-90 output from the Big 5 and discounts it for league
strength, giving a real prior for players the FPL warehouse has never seen.

Coverage limit worth stating plainly: soccerdata's FBref backend serves the Big
5 only (England, Spain, Italy, Germany, France). Arrivals from the Eredivisie,
Primeira Liga, the Championship or outside Europe get no foreign prior and fall
back to role + tier. Roughly half of a typical summer's signings.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

# Output retention when moving to the Premier League, relative to the PL itself.
# THESE ARE ASSUMPTIONS, not fitted values -- there is no matched-move dataset in
# the warehouse to estimate them from yet. They are deliberately conservative and
# are the first thing to validate once a few seasons of arrivals accumulate.
LEAGUE_STRENGTH = {
    "ENG-Premier League": 1.00,
    "ESP-La Liga": 0.85,
    "ITA-Serie A": 0.85,
    "GER-Bundesliga": 0.85,
    "FRA-Ligue 1": 0.80,
}
DEFAULT_STRENGTH = 0.70          # unknown / second-tier / outside Big 5


def normalise(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace("-", " ").replace("'", "").split())


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = ["_".join(str(x) for x in c if x).strip("_") if isinstance(c, tuple)
                  else str(c) for c in df.columns]
    return df


def fetch(season: str = "2025-2026") -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore")
    import soccerdata as sd

    fb = sd.FBref(leagues="Big 5 European Leagues Combined", seasons=season)
    std = _flatten(fb.read_player_season_stats(stat_type="standard"))
    return std


def build(out: Path = Path("data/raw/fbref"), season: str = "2025-2026") -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    d = fetch(season)

    col = lambda frag: next((c for c in d.columns if c.endswith(frag)), None)
    mins = col("Playing Time_Min")
    n90s = col("Playing Time_90s")
    starts = col("Playing Time_Starts")
    mp = col("Playing Time_MP")
    # FBref's Big-5 standard table carries no xG columns, so use realised
    # non-penalty output instead. G-PK is already penalty-stripped, which keeps
    # this consistent with the npxG treatment applied to Premier League players.
    npg90 = col("Per 90 Minutes_G-PK")
    ast90 = col("Per 90 Minutes_Ast")
    gls = col("Performance_Gls")
    ast = col("Performance_Ast")

    keep = {"player": "fbref_name", "team": "fbref_team", "league": "league",
            "pos": "fbref_pos"}
    o = d[[c for c in keep if c in d.columns]].rename(columns=keep).copy()
    o["minutes"] = pd.to_numeric(d[mins], errors="coerce") if mins else pd.NA
    o["n90"] = pd.to_numeric(d[n90s], errors="coerce") if n90s else o["minutes"] / 90
    o["starts"] = pd.to_numeric(d[starts], errors="coerce") if starts else pd.NA
    o["apps"] = pd.to_numeric(d[mp], errors="coerce") if mp else pd.NA
    for name, c in [("goals", gls), ("assists", ast)]:
        o[name] = pd.to_numeric(d[c], errors="coerce") if c else pd.NA
    o["npg_per90"] = pd.to_numeric(d[npg90], errors="coerce") if npg90 else pd.NA
    o["ast_per90"] = pd.to_numeric(d[ast90], errors="coerce") if ast90 else pd.NA
    o["start_rate"] = o["starts"] / o["apps"].replace(0, pd.NA)

    # FBref's combined Big-5 table leaves `league` blank for every Bundesliga
    # club (18 teams, ~500 players). Left unfixed they silently take the
    # unknown-league discount instead of the German one.
    o["league"] = o["league"].fillna("GER-Bundesliga")
    o["strength"] = o["league"].map(LEAGUE_STRENGTH).fillna(DEFAULT_STRENGTH)
    o["npg_per90_adj"] = o["npg_per90"] * o["strength"]
    o["ast_per90_adj"] = o["ast_per90"] * o["strength"]
    o["norm"] = o["fbref_name"].map(normalise)
    o["season"] = season

    o.to_parquet(out / f"big5_{season.replace('-', '_')}.parquet", index=False)
    return o


if __name__ == "__main__":
    o = build()
    played = o[o.minutes.fillna(0) >= 450]
    print(f"Big 5 players: {len(o):,}  with 450+ minutes: {len(played):,}")
    print(f"  leagues: {o.league.value_counts(dropna=False).to_dict()}")
    print()
    for q in ["luka vuskovic", "tarik muharemovic", "kosta nedeljkovic"]:
        m = o[o.norm == q]
        if len(m):
            r = m.iloc[0]
            print(f"  {r.fbref_name:<22}{str(r.fbref_team):<16}{r.league:<18}"
                  f"{r.minutes:>5.0f}min  npG/90 {r.npg_per90:.3f} "
                  f"-> adj {r.npg_per90_adj:.3f} (x{r.strength})")
