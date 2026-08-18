"""Validate the scoring config by reconstructing historical points.

Any season where the reconstruction is not exact indicates either a rule change
we have not encoded or a data problem. Both are things you want to find now,
not after a model is built on top.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from fpl.models.scoring import load_config, score


def validate(db: str = "data/fpl.duckdb") -> pd.DataFrame:
    cfg = load_config()
    con = duckdb.connect(db)
    df = con.execute("SELECT * FROM player_gw WHERE position IS NOT NULL").df()
    con.close()

    rows = []
    for season, grp in df.groupby("season"):
        g = grp[grp["minutes"] > 0].copy()
        if not len(g):
            continue
        # Only score DefCon in seasons where the rule existed.
        has_dc = bool(g["defcon_scoring_active"].iloc[0]) and g["defcon"].notna().any()
        pred = score(g, cfg, with_defcon=has_dc)
        resid = g["total_points"] - pred
        rows.append({
            "season": season,
            "rows": len(g),
            "exact": float((resid == 0).mean()),
            "mae": float(resid.abs().mean()),
            "defcon": g["defcon_source"].iloc[0] if has_dc else "not-scored",
            "top_resid": resid.value_counts().head(3).to_dict(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = validate()
    print(f"{'season':9}{'rows':>8}{'exact':>9}{'MAE':>8}  {'defcon':<14} residuals")
    for _, x in r.iterrows():
        print(f"{x.season:9}{x.rows:>8,}{x.exact:>9.2%}{x.mae:>8.3f}  {x.defcon:<14} {x.top_resid}")
