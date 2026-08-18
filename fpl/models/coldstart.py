"""Cold start for a season with zero played gameweeks.

2026/27 has not kicked off, so every current-season feature is undefined. The
blend the doc describes (prior -> current data, weight shifting by ~GW8-10) is
therefore at weight 1.0 on priors right now. This module builds those priors.

Two distinct cold-start problems:

  * PLAYERS. 78% of the 2026/27 squad appears in 2025/26 under a stable FPL
    `code` (identical coverage to `opta_code`). The other 22% -- promoted-club
    players, overseas signings, youth -- have no history at all and fall back to
    the position x price-tier prior. That is precisely the case empirical-Bayes
    shrinkage already handles: n90 = 0 puts full weight on the prior.

  * CLUBS. Promoted clubs have no Premier League matches. Rather than let the
    Dixon-Coles fit invent parameters for them, they are assigned an explicit
    promoted-club prior estimated from how promoted sides have actually
    performed in this dataset.
"""

from __future__ import annotations

import gzip
import glob
import json

import duckdb
import numpy as np
import pandas as pd

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def current_squad() -> pd.DataFrame:
    """Latest bootstrap snapshot -> the 2026/27 player universe."""
    path = sorted(glob.glob("data/raw/snapshots/bootstrap/date=*/*.json.gz"))[-1]
    b = json.load(gzip.open(path, "rt"))
    e = pd.DataFrame(b["elements"])
    teams = {t["id"]: t["code"] for t in b["teams"]}
    names = {t["id"]: t["name"] for t in b["teams"]}
    e["club_code"] = e["team"].map(teams)
    e["club_name"] = e["team"].map(names)
    e["position"] = e["element_type"].map(POSITION_MAP)
    e["price"] = e["now_cost"] / 10.0
    keep = ["id", "code", "opta_code", "web_name", "position", "club_code",
            "club_name", "price", "status", "chance_of_playing_next_round",
            "penalties_order", "corners_and_indirect_freekicks_order",
            "direct_freekicks_order", "selected_by_percent"]
    return e[keep].rename(columns={"id": "element"})


def promoted_club_prior(db: str = "data/fpl.duckdb") -> dict:
    """How promoted clubs actually perform, measured rather than guessed.

    A club is 'promoted' in season S if it has team_match rows in S but none in
    S-1. Their first-season attack/defence relative to the league average is the
    prior applied to 2026/27's promoted sides.
    """
    con = duckdb.connect(db)
    t = con.execute("""SELECT season, club_code, avg(goals_for) gf, avg(goals_against) ga
                       FROM team_match GROUP BY season, club_code""").df()
    con.close()
    seasons = sorted(t.season.unique())
    rows = []
    for prev, cur in zip(seasons, seasons[1:]):
        was = set(t[t.season == prev].club_code)
        now = t[t.season == cur]
        newly = now[~now.club_code.isin(was)]
        league = now[["gf", "ga"]].mean()
        for _, r in newly.iterrows():
            rows.append({"gf_ratio": r.gf / league.gf, "ga_ratio": r.ga / league.ga})
    d = pd.DataFrame(rows)
    return {"n": len(d),
            "attack_ratio": float(d.gf_ratio.mean()),
            "defence_ratio": float(d.ga_ratio.mean())}


def player_priors(db: str = "data/fpl.duckdb", source_season: str = "2025-26") -> pd.DataFrame:
    """Per-player priors carried from the most recent completed season."""
    con = duckdb.connect(db)
    hist = con.execute(f"""
        SELECT p.element, p.position, p.minutes, p.expected_goals xg,
               p.expected_assists xa, p.tackles, p.recoveries,
               p.clearances_blocks_interceptions cbi, p.saves, t.xg_for team_xg
        FROM player_gw p
        JOIN team_match t ON p.season=t.season AND p.fixture=t.fixture AND p.team_id=t.team_id
        WHERE p.season='{source_season}' AND p.minutes>0 AND t.xg_for IS NOT NULL
    """).df()
    con.close()

    hist["dc_n"] = np.where(hist.position == "DEF",
                            hist.tackles.fillna(0) + hist.cbi.fillna(0),
                            hist.tackles.fillna(0) + hist.cbi.fillna(0) + hist.recoveries.fillna(0))
    g = hist.groupby("element")
    agg = pd.DataFrame({
        "mins_total": g["minutes"].sum(),
        "apps": g["minutes"].count(),
        "starts60": g["minutes"].apply(lambda s: (s >= 60).mean()),
        "xg_share": g.apply(lambda d: (d.xg.sum() / d.team_xg.sum()) if d.team_xg.sum() > 0 else 0,
                            include_groups=False),
        "xa_share": g.apply(lambda d: (d.xa.sum() / d.team_xg.sum()) if d.team_xg.sum() > 0 else 0,
                            include_groups=False),
        "dc_per90": g.apply(lambda d: d.dc_n.sum() / max(d.minutes.sum(), 1) * 90,
                            include_groups=False),
        "save_per90": g.apply(lambda d: d.saves.fillna(0).sum() / max(d.minutes.sum(), 1) * 90,
                              include_groups=False),
    }).reset_index()
    agg["n90"] = agg["mins_total"] / 90.0

    # element -> stable code, so priors survive the id reshuffle between seasons
    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={source_season}.parquet")
    agg = agg.merge(pl[["id", "code"]].rename(columns={"id": "element"}), on="element", how="left")
    return agg.drop(columns=["element"])


def build(db: str = "data/fpl.duckdb") -> pd.DataFrame:
    squad = current_squad()
    priors = player_priors(db)
    d = squad.merge(priors, on="code", how="left")
    d["has_history"] = d["n90"].notna()
    d["n90"] = d["n90"].fillna(0.0)

    # Position x price-tier prior for everyone, used directly for the 22% with
    # no history and as the shrinkage target for everyone else.
    d["tier"] = (d.groupby("position")["price"]
                   .transform(lambda s: pd.qcut(s.rank(method="first"), 4,
                                                labels=False, duplicates="drop")))
    d["group"] = d["position"] + "_" + d["tier"].astype(str)
    for col in ["xg_share", "xa_share", "dc_per90", "starts60", "save_per90"]:
        grp_mean = d.groupby("group")[col].transform("mean")
        d[f"{col}_prior"] = grp_mean
        # Empirical-Bayes weight: few 90s -> sit near the prior.
        w = d["n90"] / (d["n90"] + 8.0)
        d[col] = w * d[col].fillna(grp_mean) + (1 - w) * grp_mean
    return d


if __name__ == "__main__":
    pp = promoted_club_prior()
    print(f"promoted-club prior from {pp['n']} club-seasons:")
    print(f"  attack  {pp['attack_ratio']:.3f} x league average")
    print(f"  defence {pp['defence_ratio']:.3f} x league average goals conceded")
    d = build()
    print(f"\n2026/27 squad: {len(d)} players")
    print(f"  with 2025/26 history: {d.has_history.sum()} ({d.has_history.mean():.1%})")
    print(f"  cold (prior only)   : {(~d.has_history).sum()}")
    d.to_parquet("data/features/coldstart_2026_27.parquet", index=False)
    print("\ntop 8 by prior xG share:")
    print(d.nlargest(8, "xg_share")[["web_name","club_name","position","price","xg_share","n90"]]
            .to_string(index=False))
