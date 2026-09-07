#!/usr/bin/env python3
"""Record what the engine projected against what actually happened.

The positional calibration can only refit itself if the pairs exist, and they
have to be captured at the time: once a gameweek is played the projection that
preceded it is gone unless it was written down. One row per player per
completed gameweek, appended, deduplicated on (season, gw, element).

Run after each gameweek, before the next refresh.
"""
from pathlib import Path

import duckdb
import pandas as pd

OUT = Path("data/features/projection_log.parquet")
SEASON = "2026-27"


def main() -> None:
    proj = pd.read_parquet("data/features/horizon_by_gw.parquet")
    con = duckdb.connect("data/fpl.duckdb", read_only=True)
    act = con.execute(
        f"SELECT gw, element, position, total_points, minutes "
        f"FROM player_gw WHERE season='{SEASON}'").df()
    con.close()
    if act.empty:
        print("no completed gameweeks yet")
        return

    m = act.merge(proj[["gw", "element", "xpts"]], on=["gw", "element"], how="inner")
    m["season"] = SEASON
    if OUT.exists():
        m = pd.concat([pd.read_parquet(OUT), m], ignore_index=True)
    # keep="first", not "last". The point of this log is the projection that
    # PRECEDED a gameweek, and the pipeline re-projects played gameweeks every
    # time it refreshes. Keeping the newest silently replaced the forecast with
    # a later model's hindsight view of the same week -- gameweek 1 was
    # recorded at 813 points, then overwritten with 770 after the valuation and
    # calibration changes, so a published out-of-sample figure stopped being
    # reproducible from its own log. First write wins; later ones are dropped.
    m = m.drop_duplicates(["season", "gw", "element"], keep="first")
    m.to_parquet(OUT, index=False)
    print(f"{len(m)} projection/outcome pairs logged "
          f"across gameweeks {sorted(m.gw.unique())}")


if __name__ == "__main__":
    main()
