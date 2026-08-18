"""Point-in-time safety.

The single most common reason an FPL backtest looks great and live performance
is mediocre is leakage: a feature computed at gameweek t that saw data from
t or later. This module makes that structurally hard rather than a convention
people remember to follow.

Any feature function decorated with @point_in_time:

  * must take `as_of_gw` (and `season`) as keyword arguments
  * receives a frame already filtered to gw < as_of_gw
  * is audited after the fact -- if the returned frame references any gw >=
    as_of_gw, it raises rather than returning silently wrong numbers

The audit is cheap and runs always. Turning it off is not offered on purpose.
"""

from __future__ import annotations

import functools
from typing import Callable

import pandas as pd


class LeakageError(AssertionError):
    """Raised when a feature function sees data at or beyond its as-of point."""


def filter_history(df: pd.DataFrame, season: str, as_of_gw: int) -> pd.DataFrame:
    """All rows strictly before (season, as_of_gw), including prior seasons.

    Prior seasons are included deliberately -- they are the cold-start prior.
    """
    prior_seasons = df["season"] < season
    this_season = (df["season"] == season) & (df["gw"] < as_of_gw)
    return df[prior_seasons | this_season]


def point_in_time(fn: Callable) -> Callable:
    """Enforce that a feature function only reads the past."""

    @functools.wraps(fn)
    def wrapper(df: pd.DataFrame, *args, season: str, as_of_gw: int, **kwargs):
        history = filter_history(df, season=season, as_of_gw=as_of_gw)

        if "gw" in history.columns and len(history):
            same = history[history["season"] == season]
            if len(same) and same["gw"].max() >= as_of_gw:
                raise LeakageError(
                    f"{fn.__name__}: input contains gw >= {as_of_gw} for {season}"
                )

        out = fn(history, *args, season=season, as_of_gw=as_of_gw, **kwargs)

        # Audit the output too -- a function can join its way back to the future.
        if isinstance(out, pd.DataFrame) and "gw" in out.columns and "season" in out.columns:
            bad = out[(out["season"] == season) & (out["gw"] >= as_of_gw)]
            if len(bad):
                raise LeakageError(
                    f"{fn.__name__}: output leaks {len(bad)} rows at gw >= {as_of_gw}"
                )
        return out

    wrapper._point_in_time = True  # type: ignore[attr-defined]
    return wrapper


def ewma_by(df: pd.DataFrame, group: list[str], value: str,
            half_life: float, order: list[str]) -> pd.Series:
    """Exponentially-decayed mean of `value` within `group`, ordered by `order`.

    Uses only rows already present in `df`, so wrap the caller in
    @point_in_time and the decay is automatically leak-free.
    """
    d = df.sort_values(order)
    return (d.groupby(group, observed=True)[value]
             .transform(lambda s: s.ewm(halflife=half_life, adjust=False).mean()))
