"""Data-driven player roles within FPL positions.

FPL's four positions are a scoring construct, not a football one. "MID" holds
Saka and Rice; "DEF" holds an overlapping full-back and a stopper centre-back;
"FWD" holds a target man and a false nine. Ranking a player against everyone
sharing his FPL label compares people who are not doing the same job.

Roles are clustered from what players actually DO rather than from a positional
string, using three axes that separate football roles cleanly and are available
for every player:

    xg_share   -- how much of the team's goal threat runs through him
    xa_share   -- how much of its creation
    dc_per90   -- defensive workload

k-means on the standardised profile, k chosen per position for interpretability.
Clusters are then LABELLED by where their centroid sits, so the names describe
the data rather than being assumed in advance.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

AXES = ["xg_share", "xa_share", "dc_per90"]
K_BY_POS = {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 2}


def profiles(db: str = "data/fpl.duckdb", season: str = "2025-26",
             min_mins: int = 450) -> pd.DataFrame:
    con = duckdb.connect(db)
    d = con.execute(f"""
        SELECT p.element, p.position, p.minutes,
               p.expected_goals AS xg, p.expected_assists AS xa,
               p.tackles, p.recoveries, p.clearances_blocks_interceptions AS cbi,
               t.xg_for AS team_xg
        FROM player_gw p
        JOIN team_match t ON p.season=t.season AND p.fixture=t.fixture
                         AND p.team_id=t.team_id
        WHERE p.season='{season}' AND p.minutes>0 AND t.xg_for IS NOT NULL
    """).df()
    con.close()
    d["dc_n"] = np.where(d.position == "DEF",
                         d.tackles.fillna(0) + d.cbi.fillna(0),
                         d.tackles.fillna(0) + d.cbi.fillna(0) + d.recoveries.fillna(0))
    g = (d.groupby(["element", "position"])
           .agg(mins=("minutes", "sum"), xg=("xg", "sum"), xa=("xa", "sum"),
                dc=("dc_n", "sum"), team_xg=("team_xg", "sum")).reset_index())
    g = g[g.mins >= min_mins].copy()
    g["xg_share"] = g.xg / g.team_xg.clip(lower=0.1)
    g["xa_share"] = g.xa / g.team_xg.clip(lower=0.1)
    g["dc_per90"] = g.dc / g.mins * 90
    return g


def _label(centroid: pd.Series, position: str) -> str:
    """Name a cluster from its dominant axis, mechanically.

    An earlier version guessed football labels ("attacking full-back",
    "central striker") from the centroid and got them badly wrong -- Virgil and
    Tarkowski, both centre-backs, landed in "attacking full-back", and Haaland in
    "support forward". The clusters were real; the names were invented.

    These labels describe what the data says, which is all we can honestly claim:
    the cluster is the like-for-like comparison group whatever we call it.
    """
    if position == "GKP":
        return "GKP"
    axis = {"xg_share": "goal threat", "xa_share": "creator",
            "dc_per90": "defensive"}
    dom = centroid[AXES].astype(float).abs().idxmax()
    hi = centroid[dom] > 0
    if not hi and centroid[AXES].astype(float).max() < 0.25:
        return f"{position} \u00b7 low involvement"
    return f"{position} \u00b7 {'high ' if hi else 'low '}{axis[dom]}"


def fit(prof: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a role to every profiled player; return assignments and centroids."""
    out, cents = [], []
    for pos, g in prof.groupby("position"):
        k = min(K_BY_POS.get(pos, 2), max(1, len(g) // 8))
        X = StandardScaler().fit_transform(g[AXES].fillna(0))
        if k <= 1:
            g = g.assign(role_id=0)
            cs = pd.DataFrame([g[AXES].mean()], columns=AXES)
            csz = StandardScaler().fit(g[AXES].fillna(0)).transform(cs)
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
            g = g.assign(role_id=km.labels_)
            csz = km.cluster_centers_
        cz = pd.DataFrame(csz, columns=AXES)
        for rid in range(len(cz)):
            cents.append({"position": pos, "role_id": rid,
                          "role": _label(cz.iloc[rid], pos),
                          **{a: float(cz.iloc[rid][a]) for a in AXES},
                          "n": int((g.role_id == rid).sum())})
        out.append(g)
    assign = pd.concat(out, ignore_index=True)
    cent = pd.DataFrame(cents)
    assign = assign.merge(cent[["position", "role_id", "role"]],
                          on=["position", "role_id"], how="left")
    return assign, cent


def assign_all(players: pd.DataFrame, db: str = "data/fpl.duckdb") -> pd.DataFrame:
    """Give every current player a role, including those with no PL history.

    Players with a record are clustered directly. Newcomers are assigned to the
    nearest centroid using their cold-start priors, so a new signing is compared
    against players doing the same job rather than against his FPL label.
    """
    prof = profiles(db)
    assign, cent = fit(prof)

    # Join through the stable `code`, never `element`: FPL reassigns element ids
    # every season, so an element merge silently attaches 2025/26 roles to
    # whichever 2026/27 player inherited the number. That put Haaland in a
    # centre-back cluster.
    src = pd.read_parquet("data/raw/vaastav/players_raw/season=2025-26.parquet")
    assign = assign.merge(src[["id", "code"]].rename(columns={"id": "element"}),
                          on="element", how="left").dropna(subset=["code"])
    out = players.merge(assign[["code", "role", "role_id"]].drop_duplicates("code"),
                        on="code", how="left")

    # Position is a per-SEASON attribute -- FPL reclassifies players between
    # seasons. A carried-over role whose prefix no longer matches the player's
    # current position is stale and must be re-derived, or a reclassified
    # midfielder keeps being compared against midfielders while scoring as a
    # defender.
    matches = pd.Series(
        [str(r).startswith(str(pos)) for r, pos in zip(out["role"], out["position"])],
        index=out.index)
    stale = out["role"].notna() & ~matches
    out.loc[stale, ["role", "role_id"]] = np.nan

    missing = out["role"].isna()
    if missing.any():
        for pos, g in out[missing].groupby("position"):
            c = cent[cent.position == pos]
            if c.empty:
                continue
            # centroids are in standardised space; rebuild the scaler on the
            # profiled players of this position so newcomers map consistently
            base = prof[prof.position == pos]
            if base.empty:
                continue
            sc = StandardScaler().fit(base[AXES].fillna(0))
            X = sc.transform(g[AXES].fillna(0))
            C = c[AXES].to_numpy()
            nearest = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
            out.loc[g.index, "role"] = c.iloc[nearest]["role"].to_numpy()
            out.loc[g.index, "role_id"] = c.iloc[nearest]["role_id"].to_numpy()
    out["role"] = out["role"].fillna(out["position"])
    return out
