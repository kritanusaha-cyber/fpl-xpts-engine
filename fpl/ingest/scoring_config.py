"""Generate config/scoring_2026_27.yaml from the live FPL API.

The full point-value table ships in bootstrap-static's `game_config.scoring`,
so we generate rather than transcribe it -- that removes a class of silent
transcription bugs and makes a future rule change a no-op.

Two things are NOT in the API and must be maintained here by hand:

  * DefCon thresholds. The API gives the point value (2) but not the count
    needed to earn it. Values below were derived empirically by reconstructing
    every 2025/26 scoring event and locating the residual jump -- see
    `validate_defcon` in fpl/backtest/validate_scoring.py.
  * The BPS weight table, which is not exposed at all.

DefCon composition is position-dependent and was likewise derived empirically
(100% exact match on 26,330 rows of 2025/26 data):

    DEF        = tackles + clearances_blocks_interceptions
    MID / FWD  = tackles + clearances_blocks_interceptions + recoveries

Note that defenders' DefCon EXCLUDES recoveries. This is easy to get wrong and
it matters: including them inflates defender DefCon by ~50%.
"""

from __future__ import annotations

from pathlib import Path

import requests
import yaml

BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

# Empirically derived, not taken from documentation. Clean separation observed:
# highest non-scoring count is threshold-1, lowest scoring count is threshold.
DEFCON_THRESHOLDS = {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12}

DEFCON_COMPONENTS = {
    "GKP": [],
    "DEF": ["tackles", "clearances_blocks_interceptions"],
    "MID": ["tackles", "clearances_blocks_interceptions", "recoveries"],
    "FWD": ["tackles", "clearances_blocks_interceptions", "recoveries"],
}


def build_config(bootstrap: dict | None = None) -> dict:
    if bootstrap is None:
        bootstrap = requests.get(BOOTSTRAP, timeout=60).json()

    gc = bootstrap["game_config"]
    scoring = gc["scoring"]
    rules = gc["rules"]

    return {
        "_generated_from": BOOTSTRAP,
        "_note": "Generated file. Edit the generator, not this. "
                 "defcon.thresholds and bps are hand-maintained (not in API).",
        "season": "2026-27",
        "scoring": scoring,
        "defcon": {
            "thresholds": DEFCON_THRESHOLDS,
            "components": DEFCON_COMPONENTS,
            "_source": "empirical, 2025-26 vaastav data",
        },
        "squad": {
            "budget": rules["squad_total_spend"],          # 1000 = £100.0m
            "size": rules["squad_squadsize"],              # 15
            "starting": rules["squad_squadplay"],          # 11
            "team_limit": rules["squad_team_limit"],       # 3 per club
            "sell_on_fee": rules["transfers_sell_on_fee"], # 0.5 -> profit taxed 50%
            "sell_at_purchase_price": rules["element_sell_at_purchase_price"],
            "max_extra_free_transfers": rules["max_extra_free_transfers"],
            "transfers_cap": rules["transfers_cap"],
        },
        "positions": {
            et["singular_name_short"]: {
                "squad_select": et["squad_select"],
                "squad_min_play": et["squad_min_play"],
                "squad_max_play": et["squad_max_play"],
            }
            for et in bootstrap["element_types"]
        },
        "chips": [
            {"name": c["name"], "type": c["chip_type"],
             "start_event": c["start_event"], "stop_event": c["stop_event"]}
            for c in bootstrap["chips"]
        ],
    }


def write(path: Path = Path("config/scoring_2026_27.yaml")) -> dict:
    cfg = build_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return cfg


if __name__ == "__main__":
    cfg = write()
    print("wrote config/scoring_2026_27.yaml")
    print(f"  chips: {len(cfg['chips'])}  budget: {cfg['squad']['budget']/10:.1f}m")
    print(f"  defcon thresholds: {cfg['defcon']['thresholds']}")
