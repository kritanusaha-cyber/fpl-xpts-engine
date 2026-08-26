"""Term structure of a player's expected points.

A yield curve plots return against how long you hold. The same object exists
here and nobody draws it: a player's expected points per gameweek depends on
how long you intend to keep him, because the fixtures he plays in the next
three weeks are not the ones he plays in the next twelve.

    yield(g, h) = mean expected points per gameweek
                  over gameweeks g .. g+h-1

Read the same way a rates curve is read.

  UPWARD SLOPING -- long yield above short. His fixtures improve. If you own
  him, hold; if you do not, he is cheaper to buy now than he will be to buy
  once the run arrives and the market has noticed.

  INVERTED -- short yield above long. He is at his best right now. Own him for
  the run and plan the exit, because the curve says the return decays.

  FLAT -- no timing information. Whatever you decide about him, the calendar is
  not the reason.

The term spread, long minus short, is the single number: the FPL equivalent of
2s10s. It is a statement about *fixtures*, not about form -- nothing here
forecasts a player rediscovering his shooting boots in March.

One caution the rates analogy actually carries over. A curve built only from
fixtures is flat for players whose scoring does not depend much on the
opponent, and that is most attackers: measured over the season, a forward's
fixture swing is a seventh of the spread between forwards. An inverted curve on
a striker is noise dressed as a signal. It is goalkeepers and defenders, whose
points come from clean sheets, where the curve carries real information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Horizons to quote, in gameweeks. Chosen like a rates curve: dense at the
# short end where decisions are actually made, sparse at the long end.
TENORS = [1, 2, 3, 4, 6, 8, 12, 20]

SHORT, LONG = 3, 12


def curve(by_gw: pd.DataFrame, start: int, tenors: list[int] | None = None) -> pd.DataFrame:
    """Yield per gameweek for each player, at each tenor, from `start`."""
    tenors = tenors or TENORS
    d = by_gw[by_gw.gw >= start]
    if d.empty:
        return pd.DataFrame()
    last = int(by_gw.gw.max())
    out = {}
    for h in tenors:
        end = min(start + h - 1, last)
        n = end - start + 1
        if n <= 0:
            continue
        s = d[d.gw <= end].groupby("element").xpts.sum() / n
        out[h] = s
    c = pd.DataFrame(out)
    c.columns = [f"y{h}" for c_ in [0] for h in c.columns]
    return c.reset_index()


def spread(c: pd.DataFrame, short: int = SHORT, long: int = LONG) -> pd.Series:
    """Long yield minus short yield. Positive means fixtures improve."""
    a, b = f"y{short}", f"y{long}"
    if a not in c.columns or b not in c.columns:
        return pd.Series(dtype=float)
    return c[b] - c[a]


def evolution(by_gw: pd.DataFrame, starts: list[int] | None = None,
              short: int = SHORT, long: int = LONG) -> pd.DataFrame:
    """Term spread from every starting gameweek -- the curve moving through time.

    This is what makes it a surface rather than a chart: the same player's curve
    inverts and re-steepens as the season turns, and the points where it crosses
    zero are the points where holding him stops paying and starts costing.
    """
    last = int(by_gw.gw.max())
    starts = starts or [g for g in sorted(by_gw.gw.unique()) if g + short - 1 <= last]
    rows = []
    for g in starts:
        c = curve(by_gw, g, [short, long])
        if c.empty:
            continue
        c = c.assign(start=g, spread=spread(c, short, long))
        rows.append(c[["element", "start", f"y{short}", f"y{long}", "spread"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
