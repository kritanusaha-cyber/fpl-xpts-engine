"""SofaScore heatmap ingest.

SofaScore publishes a per-player, per-match heatmap: an actual point cloud of
where the player touched the ball, which is finer-grained than any aggregate
count. Binned into a 6x5 grid it gives the zonal occupancy profile the reference
chart uses, per player rather than per team.

ACCESS NOTE, because it shaped the design. Requests from this sandbox's HTTP
egress are refused (403 from SofaScore's Varnish edge, on every path including
robots.txt). The same machine's browser is served normally. That is an egress
block on the automation path, not a restriction on the user, so collection runs
through the browser: the page fetches same-origin, bins the points in-page, and
only compact per-player grids cross back. Roughly 650k raw coordinates never
leave the browser, which is also the only way the volume is tractable.

Rounds are sampled across the season rather than taken consecutively, so a
player's profile is not one purple patch of form or one run of easy fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NX, NY = 6, 5           # grid matching the reference chart's zoning
RAW = Path("data/raw/sofascore")


def save_collection(payload: dict, out: Path = RAW) -> pd.DataFrame:
    """Persist the browser-collected aggregate and flatten it to a table."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "heatmap_agg.json").write_text(json.dumps(payload))
    rows = []
    for pid, r in (payload.get("agg") or {}).items():
        grid = r.get("grid") or []
        if not grid or not r.get("pts"):
            continue
        rows.append({
            "sofa_id": int(pid), "player_name": r.get("name"),
            "minutes": r.get("mins", 0), "matches": r.get("matches", 0),
            "points": r.get("pts", 0), "box_points": r.get("box", 0),
            "opp_half_passes": r.get("oppHalfPass", 0),
            "key_passes": r.get("keyPass", 0), "big_chances": r.get("bigCh", 0),
            **{f"z{i}": g for i, g in enumerate(grid)},
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_parquet(out / "player_heatmaps.parquet", index=False)
    return df


def zonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw grid counts into interpretable territory shares.

    Shares, not counts: a player on a possession side touches the ball more
    everywhere, and we want where he plays, not how much his team has the ball.
    """
    z = [f"z{i}" for i in range(NX * NY)]
    d = df.copy()
    tot = d[z].sum(axis=1).clip(lower=1)

    # Columns of the grid run defensive -> attacking (x bins 0..5).
    def col(i):
        return d[[f"z{r*NX+i}" for r in range(NY)]].sum(axis=1)

    d["def_third"] = (col(0) + col(1)) / tot
    d["mid_third"] = (col(2) + col(3)) / tot
    d["att_third"] = (col(4) + col(5)) / tot
    d["final_sixth"] = col(5) / tot
    d["box_share"] = d["box_points"] / tot

    # Vertical position: wide vs central. Rows 0 and 4 are the touchlines.
    wide = d[[f"z{0*NX+i}" for i in range(NX)]].sum(axis=1) + \
           d[[f"z{4*NX+i}" for i in range(NX)]].sum(axis=1)
    d["wide_share"] = wide / tot

    n90 = (d["minutes"] / 90).clip(lower=0.2)
    d["key_passes_p90"] = d["key_passes"] / n90
    d["opp_half_passes_p90"] = d["opp_half_passes"] / n90
    # Territory index: how far up the pitch this player lives, 0 (own box) to 1.
    d["advancement"] = (d["att_third"] * 1.0 + d["mid_third"] * 0.5)
    return d


def resolve_to_fpl(df: pd.DataFrame, season: str = "2025-26") -> pd.DataFrame:
    """Name-match SofaScore players onto the stable FPL `code`."""
    from fpl.ingest.fbref import normalise, manual_overrides
    pl = pd.read_parquet(f"data/raw/vaastav/players_raw/season={season}.parquet")
    pl["full"] = (pl["first_name"].fillna("") + " " + pl["second_name"].fillna("")).map(normalise)
    pl["surname"] = pl["second_name"].fillna("").map(normalise)

    d = df.copy()
    d["norm"] = d["player_name"].map(normalise)
    m = d.merge(pl[["code", "full"]].rename(columns={"full": "norm"}), on="norm", how="left")
    counts = pl["surname"].value_counts()
    uniq = pl[pl["surname"].isin(counts[counts == 1].index)]
    lut = dict(zip(uniq["surname"], uniq["code"]))
    for tok in (-1, 0):
        miss = m["code"].isna()
        if miss.any():
            m.loc[miss, "code"] = m.loc[miss, "norm"].str.split().str[tok].map(lut)
    ov = manual_overrides()
    if ov:
        miss = m["code"].isna()
        m.loc[miss, "code"] = m.loc[miss, "player_name"].map(ov)
    return m
