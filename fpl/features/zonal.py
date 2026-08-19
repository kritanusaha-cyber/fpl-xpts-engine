"""Zonal player features, and the football claim each one encodes.

Every feature here is a territorial statement, not an output statement. That
matters because output is what we are trying to predict; territory is what
generates it, and it stabilises faster.

ATTACKERS -- where the shots come from
    six_yard_share      shots taken inside the six-yard box, as a share
    box_share           shots inside the penalty area
    xg_per_shot         average chance quality
  A striker living in the six-yard box is getting on the end of service. Two
  strikers with equal xG can differ completely here, and the six-yard-box player
  is the one whose returns depend less on his own shot creation.

MIDFIELDERS -- arriving, and supplying
    box_touches_p90     touches in the opposition box
    passes_ft_p90       passes into the final third
    big_chances_p90     big chances created
  Arriving in the box and passing into it are different skills and are separated
  here. A midfielder with high box touches is an arriving runner (goal threat);
  one with high final-third passes is a supplier (assist threat).

FULL-BACKS -- how high they play
    box_touches_p90     the cleanest single proxy for an advanced full-back
    crosses_p90         delivery volume
  A full-back's assist return is mostly a function of how far up the pitch his
  side lets him live. Price does not capture that; territory does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = ["six_yard", "penalty_area", "outside_box"]


def shot_zones(shots: pd.DataFrame) -> pd.DataFrame:
    """Per player: shot distribution across zones, excluding penalties."""
    s = shots[shots["situation"].ne("Penalty")].copy()
    s["xg"] = pd.to_numeric(s["xg"], errors="coerce").fillna(0.0)
    s["is_goal"] = s["event_type"].eq("Goal") & ~s["own_goal"]

    g = s.groupby("player_id")
    out = g.apply(lambda d: pd.Series({
        "shots": len(d),
        "npxg": d.xg.sum(),
        "goals": int(d.is_goal.sum()),
        **{f"shots_{z}": int((d.zone == z).sum()) for z in ZONES},
        **{f"xg_{z}": d.loc[d.zone == z, "xg"].sum() for z in ZONES},
    }), include_groups=False).reset_index()

    n = out["shots"].clip(lower=1)
    out["six_yard_share"] = out["shots_six_yard"] / n
    out["box_share"] = (out["shots_six_yard"] + out["shots_penalty_area"]) / n
    out["xg_per_shot"] = out["npxg"] / n
    return out


def territory(stats: pd.DataFrame) -> pd.DataFrame:
    """Per player, per 90: box touches, final-third passes, crosses."""
    s = stats.copy()
    for c in ("minutes", "box_touches", "passes_final_third", "crosses",
              "big_chances_created", "chances_created", "touches", "xa"):
        if c not in s.columns:
            s[c] = np.nan
        s[c] = pd.to_numeric(s[c], errors="coerce")

    g = s.groupby(["player_id", "player_name"], dropna=False)
    out = g.agg(
        minutes=("minutes", "sum"),
        matches=("match_id", "nunique"),
        line=("line", lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None),
        box_touches=("box_touches", "sum"),
        passes_ft=("passes_final_third", "sum"),
        crosses=("crosses", "sum"),
        big_chances=("big_chances_created", "sum"),
        chances=("chances_created", "sum"),
        touches=("touches", "sum"),
        xa=("xa", "sum"),
    ).reset_index()

    n90 = (out["minutes"] / 90).clip(lower=0.1)
    for src, dst in [("box_touches", "box_touches_p90"), ("passes_ft", "passes_ft_p90"),
                     ("crosses", "crosses_p90"), ("big_chances", "big_chances_p90"),
                     ("chances", "chances_p90"), ("touches", "touches_p90")]:
        out[dst] = out[src] / n90
    # Share of a player's own touches that happen in the opposition box -- a
    # territory measure that does not simply reward high-possession sides.
    out["box_touch_share"] = out["box_touches"] / out["touches"].clip(lower=1)
    return out


def build(stats_path: str = "data/raw/fotmob/player_match_stats.parquet",
          shots_path: str = "data/raw/fotmob/shots_zoned.parquet") -> pd.DataFrame:
    st = pd.read_parquet(stats_path)
    sh = pd.read_parquet(shots_path)
    terr = territory(st)
    zones = shot_zones(sh)
    return terr.merge(zones, on="player_id", how="left")
