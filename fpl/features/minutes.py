"""Features for the minutes model.

Everything here is causal by construction: each feature is built from a
`groupby(player).shift(1)` series, so row t can only ever see rows < t. That is
cheaper than re-filtering the frame per gameweek and it is verified independently
against the @point_in_time path in test_minutes_features.py.

`chance_of_playing_next_round` is deliberately absent. It is the strongest
available injury signal but the API exposes no history for it, so it cannot be
used in a historical backtest without leaking. It enters at prediction time
only, from the snapshot table -- which is why the snapshotter shipped first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIVES = [3, 5, 10]
ORDER = ["season", "element", "kickoff_time", "gw"]


def _lagged(g: pd.core.groupby.SeriesGroupBy) -> pd.Series:
    """Shift within player-season so row t never sees its own outcome."""
    return g.shift(1)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Attach minutes features. Input must be the full player_gw fact table."""
    d = df.copy()
    d["kickoff_time"] = pd.to_datetime(d["kickoff_time"], errors="coerce", utc=True)
    d = d.sort_values(ORDER).reset_index(drop=True)

    # --- targets -----------------------------------------------------------
    d["appeared"] = (d["minutes"] > 0).astype(int)
    d["played_60"] = (d["minutes"] >= 60).astype(int)
    # Ordered 3-class target: 0 = unused, 1 = cameo (1-59), 2 = full (60+)
    d["minutes_class"] = np.where(d["minutes"] >= 60, 2,
                          np.where(d["minutes"] > 0, 1, 0))

    grp = d.groupby(["season", "element"], observed=True)

    # --- recent form -------------------------------------------------------
    d["prev_minutes"] = _lagged(grp["minutes"])
    d["prev_appeared"] = _lagged(grp["appeared"])
    d["prev_played_60"] = _lagged(grp["played_60"])

    # `starts` only exists from 2022/23; fall back to the 60-minute proxy so the
    # feature is defined across all ten seasons rather than only the recent four.
    started = d["starts"].fillna(d["played_60"]).clip(0, 1)
    d["_started"] = started
    g_started = d.groupby(["season", "element"], observed=True)["_started"]

    for hl in HALF_LIVES:
        d[f"ewm_start_{hl}"] = (
            g_started.shift(1)
            .groupby([d["season"], d["element"]], observed=True)
            .transform(lambda s: s.ewm(halflife=hl, adjust=False).mean())
        )
        d[f"ewm_minutes_{hl}"] = (
            grp["minutes"].shift(1)
            .groupby([d["season"], d["element"]], observed=True)
            .transform(lambda s: s.ewm(halflife=hl, adjust=False).mean())
        )

    # Cumulative appearance rate to date -- the slow-moving baseline the EWMAs
    # deviate from.
    d["cum_apps"] = grp["appeared"].transform(lambda s: s.shift(1).expanding().mean())
    d["games_seen"] = grp["appeared"].transform(lambda s: s.shift(1).expanding().count())

    # --- congestion --------------------------------------------------------
    d["days_since_last"] = (
        d["kickoff_time"] - grp["kickoff_time"].shift(1)
    ).dt.total_seconds() / 86400.0

    # Matches this player's club played in the trailing 14 days.
    d["_ko"] = d["kickoff_time"]
    team_fix = (d.dropna(subset=["team_id", "_ko"])
                  .groupby(["season", "team_id", "_ko"], observed=True)
                  .size().reset_index(name="_n"))
    congestion = []
    for (season, team_id), g in team_fix.groupby(["season", "team_id"], observed=True):
        g = g.sort_values("_ko")
        idx = g.set_index("_ko")
        cnt = idx["_n"].rolling("14D", closed="left").count()
        congestion.append(pd.DataFrame({"season": season, "team_id": team_id,
                                        "_ko": cnt.index, "fixtures_14d": cnt.values}))
    if congestion:
        d = d.merge(pd.concat(congestion, ignore_index=True),
                    on=["season", "team_id", "_ko"], how="left")

    # --- squad context -----------------------------------------------------
    # Price tier within position-season: a proxy for squad status that is
    # available from gw1, unlike anything form-based.
    d["price"] = d["value"] / 10.0
    d["price_rank_pos"] = (d.groupby(["season", "gw", "position"], observed=True)["price"]
                             .rank(pct=True))

    # Squad depth: how many same-position teammates are priced above this player.
    d["depth_ahead"] = (d.groupby(["season", "gw", "team_id", "position"], observed=True)["price"]
                          .rank(ascending=False, method="min") - 1)

    d = d.drop(columns=["_started", "_ko"])
    return d


FEATURES = (
    ["prev_minutes", "prev_appeared", "prev_played_60", "cum_apps", "games_seen",
     "days_since_last", "fixtures_14d", "price", "price_rank_pos", "depth_ahead"]
    + [f"ewm_start_{h}" for h in HALF_LIVES]
    + [f"ewm_minutes_{h}" for h in HALF_LIVES]
)
