"""Pitch zoning using standard football-analysis terminology.

The grid is 6 columns x 5 rows, the convention used in public football analytics
(and in The Athletic's territory charts). Both axes carry real names:

VERTICAL CHANNELS (5 rows, touchline to touchline)
    Left wing, Left half-space, Centre, Right half-space, Right wing

    The half-spaces are the two channels between the centre and the wings. They
    matter because a player receiving there can face goal, shoot, and play a
    cutback without the touchline compressing his options -- which is why modern
    sides deliberately overload them.

HORIZONTAL BANDS (6 columns, own goal to opposition goal)
    Defensive third (2), Middle third (2), Attacking third (2)

ZONE 14
    The central zone of the attacking third immediately outside the penalty area
    -- row "Centre", column 5 in this grid. It is the single most productive
    creative zone in open play: possession there generates more shots and more
    assists per touch than anywhere else outside the box. A midfielder who lives
    in Zone 14 is a chance creator almost by definition.

Coordinates arrive on a 105 x 68 pitch attacking toward x = 105.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_L, PITCH_W = 105.0, 68.0
NX, NY = 6, 5

CHANNELS = ["Right wing", "Right half-space", "Centre", "Left half-space", "Left wing"]
BANDS = ["Halfway", "Middle", "Approach", "Zone 14 band", "Box edge", "Six-yard"]

# (row, col) of the named zones, row 0 = y nearest 0.
ZONE_14 = (2, 3)          # centre channel, just outside the penalty area
BOX_COLS = (4, 5)         # the last two columns cover the penalty area


# Shots essentially only happen in the attacking half, so the grid spans that
# half rather than the whole pitch. A full-pitch grid built from shots is
# two-thirds empty and wastes the resolution where it matters -- the six columns
# instead run from the halfway line to the goal line, which is the convention
# for shot-zone maps.
X0 = PITCH_L / 2


def zone_index(x: float, y: float) -> tuple[int, int]:
    if not np.isfinite(x) or not np.isfinite(y):
        return (-1, -1)
    col = min(NX - 1, max(0, int((x - X0) / (PITCH_L - X0) * NX)))
    row = min(NY - 1, max(0, int(y / PITCH_W * NY)))
    return row, col


def grid_for(points: pd.DataFrame, xcol: str = "x", ycol: str = "y",
             weight: str | None = None) -> np.ndarray:
    """Counts (or summed weight) per zone, returned as a NY x NX array."""
    g = np.zeros((NY, NX))
    if points.empty:
        return g
    xs = pd.to_numeric(points[xcol], errors="coerce").to_numpy()
    ys = pd.to_numeric(points[ycol], errors="coerce").to_numpy()
    ws = (pd.to_numeric(points[weight], errors="coerce").fillna(0).to_numpy()
          if weight else np.ones(len(points)))
    for x, y, w in zip(xs, ys, ws):
        r, c = zone_index(x, y)
        if r >= 0:
            g[r, c] += w
    return g


def player_grids(shots: pd.DataFrame, min_shots: int = 8) -> pd.DataFrame:
    """Per player: shot-count grid, xG grid, and the named-zone summaries."""
    s = shots[shots["situation"].ne("Penalty")].copy()
    rows = []
    for pid, d in s.groupby("player_id"):
        if len(d) < min_shots:
            continue
        gc = grid_for(d)
        gx = grid_for(d, weight="xg")
        tot = gc.sum() or 1
        rows.append({
            "player_id": pid,
            "player_name": d["player_name"].iloc[0],
            "shots": int(len(d)),
            "grid_shots": gc.flatten().astype(int).tolist(),
            "grid_xg": np.round(gx.flatten(), 3).tolist(),
            # Named-zone reads, as shares of the player's own shots
            "zone14_share": float(gc[ZONE_14] / tot),
            "centre_share": float(gc[2, :].sum() / tot),
            "halfspace_share": float((gc[1, :].sum() + gc[3, :].sum()) / tot),
            "wing_share": float((gc[0, :].sum() + gc[4, :].sum()) / tot),
            "box_share": float(gc[:, 4:].sum() / tot),
        })
    return pd.DataFrame(rows)


def describe(grid_flat: list, shots: int) -> str:
    """One-line football read of where a player operates."""
    g = np.array(grid_flat, dtype=float).reshape(NY, NX)
    tot = g.sum() or 1
    centre = g[2, :].sum() / tot
    wings = (g[0, :].sum() + g[4, :].sum()) / tot
    box = g[:, 4:].sum() / tot
    if box > 0.6:
        return "operates almost entirely inside the box"
    if centre > 0.5:
        return "central operator"
    if wings > 0.45:
        return "works the wide channels"
    return "spread across the half-spaces"


def tiered_grids(grids: pd.DataFrame, positions: pd.Series,
                 hi: float = 1.30, lo: float = 0.70) -> pd.DataFrame:
    """Classify each zone against the baseline for that player's position.

    The Athletic's territory charts use a three-way encoding -- dominant,
    contested, dominated -- rather than a continuous ramp, because the eye reads
    categories far faster than shading across a 30-cell grid. The same logic
    applies here, with the comparison being the player against his positional
    peers rather than a team against its opponent:

        ratio > 1.30   he shoots from here far more than his position does
        0.70 - 1.30    contested / typical for the position
        ratio < 0.70   he rarely gets here

    Baselines are per position because "often in the box" means something
    different for a striker than for a centre-back.
    """
    d = grids.copy()
    d["position"] = d["player_id"].map(positions)
    n = NX * NY

    shares = np.vstack([
        (np.array(g, dtype=float) / max(sum(g), 1)) for g in d["grid_shots"]
    ])
    out = np.zeros_like(shares, dtype=int)
    for pos in d["position"].dropna().unique():
        m = (d["position"] == pos).to_numpy()
        if m.sum() < 5:
            continue
        base = np.median(shares[m], axis=0)
        base = np.where(base <= 0, np.nan, base)
        ratio = shares[m] / base
        tier = np.zeros_like(ratio, dtype=int)
        tier[ratio >= hi] = 1            # dominant
        tier[(ratio > lo) & (ratio < hi)] = 0   # contested
        tier[ratio <= lo] = -1           # below
        tier[~np.isfinite(ratio)] = -1
        # A zone the player never enters is "below", never "contested".
        tier[shares[m] == 0] = -1
        out[m] = tier

    d["grid_tier"] = [row.tolist() for row in out]
    d["grid_share"] = [np.round(row, 4).tolist() for row in shares]
    return d
