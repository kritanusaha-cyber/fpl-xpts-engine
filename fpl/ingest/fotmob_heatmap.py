"""Season touch heatmaps from FotMob, and the zone analysis built on them.

A heatmap is the answer to "where does this player actually play?", which the
FPL position never tells you. Two players both listed MID can be a holding
midfielder who never leaves his own half and a second striker who lives on the
last shoulder. The zones below are picked so that difference shows up.

The season a request returns is not the season you want. FotMob's default
payload keys off whatever competition is current, so a player who has moved
abroad returns his new league: Salah's default response is 34 Super Lig touches,
not the 1,488 he recorded in the Premier League last season. The season has to
be named explicitly through `entryId`, which is a positional index into that
player's own season list and therefore differs from player to player.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fpl.ingest.fotmob import _get, HEADERS, PAUSE

API = "https://www.fotmob.com/api/data"
PL_TOURNAMENT = 47
CACHE = Path("data/raw/fotmob/heatmaps")

# FotMob's pitch, attacking toward x = 105.
PITCH_L, PITCH_W = 105.0, 68.0

# Five vertical channels. The half-spaces are the second and fourth: the lanes
# between the width of the penalty area and the touchline, where a player can
# receive facing goal without being in the crowd of the centre or isolated on
# the touchline. This is where creative midfielders do their damage.
CHANNEL = PITCH_W / 5


def zones(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks for each zone of interest. Zones deliberately overlap:
    a touch can be both in the box and in a half-space, and both facts matter."""
    cy = np.abs(y - PITCH_W / 2)
    return {
        # Inside the six-yard box. Poacher territory -- where a striker who
        # scores tap-ins spends his time and a target man does not.
        "six_yard": (x >= PITCH_L - 5.5) & (cy <= 9.16),
        # Penalty area.
        "box": (x >= PITCH_L - 16.5) & (cy <= 20.16),
        # Between the penalty spot and the six-yard line, centrally. The highest
        # value real estate on the pitch for a striker.
        "danger": (x >= PITCH_L - 11.0) & (cy <= 9.16),
        # Zone 14: central, immediately outside the box. Where a ten receives
        # and turns.
        "zone14": (x >= PITCH_L - 27.5) & (x < PITCH_L - 16.5) & (cy <= 9.16),
        # Half-spaces, final third only -- the channel matters where it creates.
        "half_space": (x >= 70) & (((y >= CHANNEL) & (y < 2 * CHANNEL))
                                   | ((y >= 3 * CHANNEL) & (y < 4 * CHANNEL))),
        # Wide channels in the final third: winger and attacking full-back land.
        "wide_att": (x >= 70) & ((y < CHANNEL) | (y >= 4 * CHANNEL)),
        "final_third": x >= 70,
        "mid_third": (x >= 35) & (x < 70),
        "def_third": x < 35,
        # Own box. A centre-back or a keeper lives here; a full-back who pushes
        # on does not.
        "own_box": (x <= 16.5) & (cy <= 20.16),
    }


def pl_entry(player_id: int, season: str = "2025/2026") -> str | None:
    """The `entryId` naming this player's Premier League season.

    Positional, so it cannot be hardcoded: a player still in the league carries
    a 2026/2027 entry ahead of it and reads "1-0", while one who has left after
    last season reads "0-0".
    """
    d = _get(f"{API}/playerData?id={player_id}")
    for s in d.get("statSeasons") or []:
        if s.get("seasonName") == season:
            for t in s.get("tournaments") or []:
                if t.get("tournamentId") == PL_TOURNAMENT:
                    return t.get("entryId")
    return None


def fetch(player_ids: list[int], cache: Path = CACHE,
          season: str = "2025/2026") -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pid in enumerate(player_ids, 1):
        f = cache / f"{pid}.json"
        if f.exists():
            d = json.loads(f.read_text())
        else:
            e = pl_entry(pid, season)
            if e is None:
                f.write_text(json.dumps({"coordinates": [], "entry": None}))
                time.sleep(PAUSE)
                continue
            time.sleep(PAUSE)
            r = _get(f"{API}/playerStats?playerId={pid}&seasonId={e}")
            # Keep the whole payload, not just the coordinates. The same
            # response carries the season stat panel with FotMob's own
            # percentile ranks and the keeper shotmap; storing only the
            # heatmap means paying for the request again to get them.
            d = {"coordinates": (r.get("heatmap") or {}).get("coordinates") or [],
                 "entry": e,
                 "statsSection": r.get("statsSection"),
                 "shotmap": r.get("shotmap"),
                 "keeperShotmap": r.get("keeperShotmap")}
            f.write_text(json.dumps(d))
            time.sleep(PAUSE)
        for c in d.get("coordinates") or []:
            rows.append({"player_id": pid, "x": c.get("x"), "y": c.get("y")})
        if i % 25 == 0:
            print(f"  {i}/{len(player_ids)} players, {len(rows)} touches")
    return pd.DataFrame(rows)


def zone_shares(touches: pd.DataFrame, min_touches: int = 40) -> pd.DataFrame:
    """Share of a player's touches falling in each zone.

    Shares, not counts. A player with more minutes has more touches everywhere,
    so counts would rank by playing time; the share isolates where he goes when
    he is on the pitch. Volume is kept alongside as `touches`, because a 40%
    box share off 50 touches is a different claim from the same share off 900.
    """
    out = []
    for pid, g in touches.dropna(subset=["x", "y"]).groupby("player_id"):
        x, y = g.x.to_numpy(), g.y.to_numpy()
        n = len(g)
        rec = {"player_id": int(pid), "touches": n,
               "x_mean": float(x.mean()), "y_spread": float(np.std(y))}
        for name, mask in zones(x, y).items():
            rec[f"z_{name}"] = float(mask.sum()) / n
        out.append(rec)
    d = pd.DataFrame(out)
    d["low_sample"] = d.touches < 300
    return d[d.touches >= min_touches].reset_index(drop=True)


# What each zone means for the player being asked about. A striker is judged on
# whether he gets into the six-yard box; a centre-back is not, and showing him
# the same number invites a comparison nobody wants to make. The role labels
# come from the k-means clusters, so a "creator" midfielder and a "workhorse"
# midfielder are read against different zones.
ZONE_READS = {
    "six_yard": ("Six-yard box", "gets on the end of crosses"),
    "danger": ("Central, inside the penalty spot", "occupies the highest-value space on the pitch"),
    "box": ("Opposition box", "plays on the last line"),
    "zone14": ("Zone 14", "receives and turns outside the box"),
    "half_space": ("Half-spaces, final third", "finds the channel between full-back and centre-back"),
    "wide_att": ("Wide, final third", "holds the touchline"),
    "final_third": ("Final third", "plays high up the pitch"),
    "def_third": ("Defensive third", "sits deep"),
    "own_box": ("Own box", "defends the six-yard line"),
}

# The zones worth showing, by FPL position. Ordered: the first entry is the one
# that most separates good from bad in that position.
POSITION_ZONES = {
    "FWD": ["danger", "six_yard", "box", "half_space", "final_third"],
    "MID": ["half_space", "zone14", "box", "wide_att", "final_third"],
    "DEF": ["own_box", "def_third", "final_third", "wide_att", "box"],
    "GKP": ["own_box", "def_third"],
}


def profile(z: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """Percentile each zone share within the player's own position.

    Within position, not across it. A centre-back takes 8% of his touches in
    the final third and a winger 60%; ranked together the defenders occupy the
    bottom eighty places and the number says nothing about whether a given
    defender pushes on more than his peers.
    """
    d = z.copy()
    d["pos"] = pos.reindex(d.index).values if isinstance(pos, pd.Series) else pos
    cols = [c for c in d.columns if c.startswith("z_")]
    for c in cols:
        d[f"p_{c[2:]}"] = d.groupby("pos")[c].rank(pct=True)
    return d
