#!/usr/bin/env python3
"""Project every player for every remaining gameweek, then read the schedule.

The dashboard answers "who is worth owning over the next six gameweeks". That
is the wrong question for planning transfers, because the answer changes as the
fixtures turn. A defender whose side plays four of the top six in October is a
poor hold in October and a good one in November, and the six-gameweek view
cannot see past its own window.

So the whole season is simulated, and every six-gameweek window in it is scored
separately. What comes out is not a ranking but a calendar: which players are
worth owning when, and where the crossings are -- because a crossing is where a
transfer belongs.

Fixture difficulty is the only thing that varies across windows here. Form,
injury and price are not forecast forward; they are what the weekly refresh is
for. This says when a player's fixtures turn, which is the part of a transfer
plan that can be known in August.
"""
import sys

import numpy as np
import pandas as pd

from fpl.predict_horizon import run

WINDOW = 6


def windows(gw_df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Rolling per-player totals over every `window`-gameweek stretch."""
    gws = sorted(gw_df.gw.unique())
    out = []
    for start in gws:
        end = start + window - 1
        if end > max(gws):
            break
        s = gw_df[(gw_df.gw >= start) & (gw_df.gw <= end)]
        agg = s.groupby("element").agg(xpts=("xpts", "sum"),
                                       games=("gw", "nunique")).reset_index()
        agg["start"] = start
        agg["end"] = end
        out.append(agg)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    print("simulating the full season -- this is 38 gameweeks, not 6", flush=True)
    tot, gw = run(horizon=38)
    gw.to_parquet("data/features/season_by_gw.parquet", index=False)

    w = windows(gw)
    # A player's own average window, so "good stretch" means good FOR HIM
    # rather than "is a premium". Without this the table is just a list of
    # expensive players in every window.
    base = w.groupby("element").xpts.mean().rename("own_mean")
    w = w.merge(base, on="element")
    w["vs_own"] = w.xpts - w.own_mean
    # and percentile within the window, so windows stay comparable
    w["pct"] = w.groupby("start").xpts.rank(pct=True)
    w.to_parquet("data/features/season_windows.parquet", index=False)

    print(f"\n{gw.element.nunique()} players over {gw.gw.nunique()} gameweeks, "
          f"{w.start.nunique()} six-gameweek windows")

    cs = pd.read_parquet("data/features/coldstart_2026_27.parquet")
    nm = dict(zip(cs.element, cs.web_name))
    cl = dict(zip(cs.element, cs.club_name))
    po = dict(zip(cs.element, cs.position))

    print("\nbiggest fixture swings -- players whose best window beats their worst by most:")
    sw = (w.groupby("element").xpts.agg(["max", "min", "mean"])
            .assign(swing=lambda x: x["max"] - x["min"]))
    sw = sw[sw["mean"] > sw["mean"].quantile(0.80)].nlargest(12, "swing")
    for el, r in sw.iterrows():
        best = w[(w.element == el)].nlargest(1, "xpts").iloc[0]
        worst = w[(w.element == el)].nsmallest(1, "xpts").iloc[0]
        print(f"  {str(nm.get(el))[:15]:16}{str(po.get(el)):4}{str(cl.get(el))[:12]:13}"
              f"best GW{int(best.start):02d}-{int(best.end):02d} {best.xpts:5.1f}   "
              f"worst GW{int(worst.start):02d}-{int(worst.end):02d} {worst.xpts:5.1f}   "
              f"swing {r.swing:4.1f}")


if __name__ == "__main__":
    sys.exit(main())
