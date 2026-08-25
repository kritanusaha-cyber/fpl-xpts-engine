"""Transfer planning with the rules that actually govern it.

The simulation up to now re-optimised greedily each week: pick the best squad for
the next gameweek, subject to a crude cap on changes. That misses the three things
that decide real transfer value.

HORIZON, NOT NEXT WEEK. A transfer is held for weeks, so its value is the weekly
gain multiplied by how long you keep it. Judging a move on the next fixture alone
makes the manager chase one-week fixtures and churn the squad. A +0.4/week upgrade
looks worth having until you notice it never repays the four points it cost.

HITS ARE AN INVESTMENT, NOT A PENALTY. −4 is worth paying when the horizon gain
clears it and not otherwise. That is an explicit inequality, and it is the whole
of hit discipline:

    take the hit  <=>  weekly_gain * hold_weeks  >  4

BANKED FREE TRANSFERS. Up to five accumulate. A manager who understands this
sometimes does nothing for two weeks in order to make two moves at once, which
the greedy version can never represent because it has no concept of saving.

SELL PRICE. `element_sell_at_purchase_price` is false and the sell-on fee is 50%,
so profit is taxed and budget is path-dependent on what you paid. A squad that has
risen in value cannot be fully recycled, and ignoring that overstates what later
transfers can afford.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HIT_COST = 4.0
MAX_BANKED = 5
SELL_ON_FEE = 0.5


def sell_price(purchase: float, current: float) -> float:
    """FPL taxes profit at 50%, rounded down to the nearest 0.1."""
    if current <= purchase:
        return current
    profit = current - purchase
    taxed = np.floor(profit * (1 - SELL_ON_FEE) * 10) / 10
    return purchase + taxed


class Squad:
    """A squad that remembers what it paid, because sell price depends on it."""

    def __init__(self, elements: set, prices: dict, bank: float = 0.0):
        self.players = dict(elements and {e: prices.get(e, 0.0) for e in elements} or {})
        self.bank = bank
        self.free_transfers = 1

    def value(self, current_prices: dict) -> float:
        return sum(sell_price(p, current_prices.get(e, p))
                   for e, p in self.players.items()) + self.bank

    def apply(self, out_: set, in_: set, current_prices: dict) -> bool:
        """Execute a swap if affordable; return False and change nothing if not."""
        proceeds = sum(sell_price(self.players[e], current_prices.get(e, self.players[e]))
                       for e in out_ if e in self.players)
        cost = sum(current_prices.get(e, 0.0) for e in in_)
        if cost > proceeds + self.bank + 1e-9:
            return False
        for e in out_:
            self.players.pop(e, None)
        for e in in_:
            self.players[e] = current_prices.get(e, 0.0)
        self.bank = round(self.bank + proceeds - cost, 1)
        return True


def worth_a_hit(weekly_gain: float, hold_weeks: float, n_hits: int,
                margin: float = 1.0) -> bool:
    """The inequality that decides whether to pay for an extra transfer.

    `margin` exists because the naive test loses points badly. The transfer you
    choose is the one with the largest ESTIMATED gain, and the largest estimate
    in a noisy set is biased upward -- the optimiser's curse. Acting on it means
    systematically paying four points for an edge that is partly estimation
    error. A margin above 1 demands the apparent gain clear the cost by enough
    to survive that bias.
    """
    return weekly_gain * hold_weeks > HIT_COST * n_hits * margin


def plan_transfers(squad: Squad, pool: pd.DataFrame, pred: str,
                   hold_weeks: float = 4.0, max_hits: int = 2,
                   hit_margin: float = 1.0,
                   fx: dict | None = None) -> tuple[set, set, int]:
    """Choose swaps by horizon value, spending banked transfers before hits.

    Candidates are ranked by the gain from replacing the weakest holder in a
    position with the best affordable alternative, valued over `hold_weeks`
    rather than over the next fixture.
    """
    if not squad.players:
        return set(), set(), 0
    prices = dict(zip(pool.element, pool.price))
    val = dict(zip(pool.element, pd.to_numeric(pool[pred], errors="coerce").fillna(0)))
    # Scale each player's value by how many of the coming gameweeks his club
    # actually plays. A player about to blank is worth less over the hold than
    # his single-gameweek projection says, and a double is worth more.
    if fx:
        _club = dict(zip(pool.element, pool.club_code))
        val = {e: v * fx.get(_club.get(e), 1.0) for e, v in val.items()}
    pos = dict(zip(pool.element, pool.position))
    club = dict(zip(pool.element, pool.club_code))

    held = set(squad.players)
    counts: dict = {}
    for e in held:
        counts[club.get(e)] = counts.get(club.get(e), 0) + 1

    moves = []
    for out_e in held:
        p_out = pos.get(out_e)
        if p_out is None:
            continue
        proceeds = sell_price(squad.players[out_e], prices.get(out_e, squad.players[out_e]))
        budget = proceeds + squad.bank
        cands = pool[(pool.position == p_out) & (~pool.element.isin(held))
                     & (pool.price <= budget + 1e-9)]
        if cands.empty:
            continue
        best = cands.nlargest(1, pred).iloc[0]
        # respect the three-per-club limit after the swap
        c_in = best.club_code
        if counts.get(c_in, 0) - (1 if club.get(out_e) == c_in else 0) >= 3:
            continue
        gain = float(val.get(best.element, 0) - val.get(out_e, 0))
        if gain > 0:
            moves.append((gain, out_e, int(best.element)))

    moves.sort(reverse=True)
    out_set, in_set, hits = set(), set(), 0
    allowed_free = min(squad.free_transfers, MAX_BANKED)
    for gain, o, i in moves:
        n = len(out_set)
        if n < allowed_free:
            pass                              # covered by a free transfer
        elif hits < max_hits and worth_a_hit(gain, hold_weeks, 1, hit_margin):
            hits += 1
        else:
            break
        out_set.add(o)
        in_set.add(i)
    return out_set, in_set, hits

def fixture_weights(pool_by_gw: dict, gw: int, horizon: int = 4) -> dict:
    """Per-club multiplier for how good the next few fixtures look.

    The planner's `hold_weeks` treats every one of the coming weeks as
    identical to this one. It is not: a striker facing the best defence in the
    league this week and three promoted sides after it is worth more than his
    current projection says, and the reverse is worth less.

    The fixture list is published months ahead, so using it is not hindsight.
    What would be hindsight is using the *projections* for those gameweeks,
    since those are built from data that has not happened yet at decision time.
    So only the schedule is read forward -- each club's opponents -- and it is
    scored with opponent strength estimated from matches already played.
    """
    hist = [g for g in pool_by_gw if g < gw]
    if not hist:
        return {}
    past = pd.concat([pool_by_gw[g] for g in hist], ignore_index=True)
    # how many points a club's opponents have conceded to date, per club
    conceded = (past.groupby("club_code")["total_points"].mean()
                    .rename("scored").to_dict())
    if not conceded:
        return {}
    lg = float(np.mean(list(conceded.values()))) or 1.0

    ahead = [g for g in pool_by_gw if gw <= g < gw + horizon]
    if not ahead:
        return {}
    w: dict = {}
    for g in ahead:
        p = pool_by_gw[g]
        for club in p["club_code"].dropna().unique():
            # a club playing at all in that gameweek is the first thing that
            # matters -- a blank is a zero, not an average week
            w[club] = w.get(club, 0.0) + 1.0
    n = len(ahead)
    return {c: v / n for c, v in w.items()}
