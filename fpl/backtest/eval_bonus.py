"""Does BPS error matter for bonus? Bonus depends only on rank within a fixture."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fpl.models.bonus import fit_bps, predict_bps, award_bonus

TEST = ["2024-25", "2025-26"]


def main() -> None:
    con = duckdb.connect("data/fpl.duckdb")
    d = con.execute("""SELECT * FROM player_gw
                       WHERE minutes > 0 AND position IS NOT NULL AND bps IS NOT NULL""").df()
    con.close()

    train = d[~d.season.isin(TEST)]
    test = d[d.season.isin(TEST)].copy()
    coef = fit_bps(train)
    test["bps_hat"] = predict_bps(test, coef)

    # Oracle: rank by the true BPS. This validates the award/tie logic itself.
    test["bonus_oracle"] = award_bonus(test, "bps")
    test["bonus_model"] = award_bonus(test, "bps_hat")

    y = test.bonus.astype(int)
    print(f"n = {len(test):,} player-matches over {TEST}\n")
    print(f'{"":26}{"exact":>9}{"MAE":>8}{"corr":>8}')
    for name, p in [("oracle (true BPS rank)", test.bonus_oracle),
                    ("model (predicted BPS)", test.bonus_model)]:
        print(f'{name:26}{(p==y).mean():>9.2%}{np.abs(p-y).mean():>8.4f}'
              f'{np.corrcoef(p,y)[0,1]:>8.4f}')

    print("\nrecovering the bonus WINNER (3 pts) per fixture:")
    for name, col in [("oracle", "bonus_oracle"), ("model", "bonus_model")]:
        hits = tot = 0
        for _, g in test.groupby(["season", "fixture"]):
            actual = set(g.index[g.bonus == 3])
            pred = set(g.index[g[col] == 3])
            if actual:
                hits += len(actual & pred) > 0
                tot += 1
        print(f"  {name:8} {hits/tot:.2%}  ({hits}/{tot} fixtures)")

    print("\nany-bonus (>=1) detection, model:")
    got, act = test.bonus_model > 0, y > 0
    tp = (got & act).sum()
    print(f"  precision {tp/max(got.sum(),1):.2%}   recall {tp/max(act.sum(),1):.2%}")


if __name__ == "__main__":
    main()
