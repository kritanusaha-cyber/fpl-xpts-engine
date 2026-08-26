"""What FPL actually charges for output, estimated from history.

Fair price was previously computed as `price + surplus / lambda`, where lambda is
the budget shadow price from the LP. That is wrong, and it is wrong in a way that
only shows up at the extremes: **lambda is the marginal rate at a constrained
optimum, not the market's price gradient.** It came out at 1.111 xPts per GBP1m
over six gameweeks, while the realised market gradient for defenders is about
3.2. Dividing by a number three times too small inflated every price gap
threefold, and produced a GBP13.3m fair price for a GBP4.5m defender.

The football logic the LP misses: FPL prices are set to deliver *diminishing*
returns. A GBP4.5m defender and a GBP8m defender are not separated by a constant
points-per-pound; the gap narrows as price rises, because the expensive players
are being paid for attacking upside that is itself capped. So the mapping from
output to price must be concave, and it must be fitted per position -- a point of
defensive output is priced very differently from a point of attacking output.

We fit, per position, over players with a real season behind them:

    points_per_gameweek = a + b * log(price)

and invert it. Fair price is then "the price at which the market historically
delivered this player's rate" -- a market-anchored number, bounded by prices that
actually exist, rather than a linear extrapolation off a marginal rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

# Twenty-one full games, not ten.
#
# The curve is applied to players the model projects as regular starters, so it
# has to be fitted on players who were regular starters. At a 900-minute floor
# the fitted population averages 2.30 points per team-gameweek while the model
# projects its starters at 2.91 -- so every starter landed high on the curve and
# came out underpriced. 76% of them, and 95% of defenders, which is not a
# valuation, it is an offset.
#
# At 1900 the fitted population matches the population it is applied to and the
# split centres: 55% underpriced, median gap +0.13. The cost is a smaller fit
# sample, which is the right trade -- a curve fitted on the wrong population is
# precise about the wrong thing.
MIN_MINUTES = 1900
FPL_MIN_PRICE, FPL_MAX_PRICE = 3.8, 16.0


@dataclass
class PriceCurve:
    """points_per_gw = a + b*log(price), fitted per position."""
    coef: dict            # position -> (a, b)
    r2: dict
    seasons: list

    def points_at(self, position: str, price: float) -> float:
        a, b = self.coef.get(position, (0.0, 0.0))
        return a + b * np.log(max(price, FPL_MIN_PRICE))

    def price_for(self, position: str, points_per_gw: float) -> float:
        """Invert: what does the market charge for this rate?"""
        a, b = self.coef.get(position, (0.0, 0.0))
        if abs(b) < 1e-9:
            return FPL_MIN_PRICE
        return float(np.clip(np.exp((points_per_gw - a) / b),
                             FPL_MIN_PRICE, FPL_MAX_PRICE))


def fit(db: str = "data/fpl.duckdb", seasons: tuple = ("2022-23", "2023-24",
                                                       "2024-25", "2025-26")) -> PriceCurve:
    con = duckdb.connect(db)
    q = ",".join(f"'{s}'" for s in seasons)
    d = con.execute(f"""
        SELECT season, element, position, avg(value)/10.0 AS price,
               sum(total_points) AS pts, sum(minutes) AS mins, count(*) AS appearances,
               (SELECT count(DISTINCT gw) FROM player_gw z
                 WHERE z.season = player_gw.season) AS team_games
        FROM player_gw
        WHERE season IN ({q}) AND position IS NOT NULL
        GROUP BY 1,2,3
        HAVING sum(minutes) >= {MIN_MINUTES}
    """).df()
    con.close()

    # Points per TEAM gameweek, not per appearance. This has to match the basis
    # the model projects on: xPts already includes the probability that the
    # player does not play, so pricing against a per-appearance rate would
    # compare an unconditional expectation to a conditional one and systematically
    # understate what the market charges. Per team-gameweek is also the honest
    # basis for a manager, who carries the player through his blanks.
    d["ppg"] = d["pts"] / d["team_games"].clip(lower=1)

    coef, r2 = {}, {}
    for pos, g in d.groupby("position"):
        if len(g) < 30:
            continue
        x = np.log(g["price"].clip(lower=FPL_MIN_PRICE).to_numpy())
        y = g["ppg"].to_numpy()
        A = np.vstack([np.ones_like(x), x]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = a + b * x
        coef[pos] = (float(a), float(b))
        r2[pos] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return PriceCurve(coef=coef, r2=r2, seasons=list(seasons))


if __name__ == "__main__":
    c = fit()
    print(f"fitted on {', '.join(c.seasons)}, players with {MIN_MINUTES}+ minutes\n")
    print(f'{"pos":6}{"a":>8}{"b":>8}{"R2":>7}   what the market charges')
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        if pos not in c.coef:
            continue
        a, b = c.coef[pos]
        ex = "  ".join(f"{p:.1f}m->{c.points_at(pos, p):.2f}" for p in (4.5, 6.0, 9.0, 12.0))
        print(f'{pos:6}{a:>8.2f}{b:>8.2f}{c.r2[pos]:>7.2f}   {ex}')
