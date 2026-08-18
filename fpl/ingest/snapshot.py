"""Daily versioned snapshot of the mutable FPL API endpoints.

Prices, ownership, injury flags and set-piece order all move continuously and
the API exposes no history for any of them. Anything not captured here is lost
permanently -- no upstream source backfills it. This is why the snapshotter
ships before any modelling code.

Run at 22:45 UTC, ahead of the 23:00 price-change deadlines that
`game_config.settings.price_change_deadlines` advertises.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://fantasy.premierleague.com/api"

ENDPOINTS = {
    "bootstrap": f"{API}/bootstrap-static/",
    "fixtures": f"{API}/fixtures/",
}

# Fields that move within a season and are unrecoverable if not snapshotted.
VOLATILE = [
    "id", "web_name", "team", "element_type", "now_cost", "selected_by_percent",
    "transfers_in_event", "transfers_out_event", "cost_change_event",
    "cost_change_start", "status", "chance_of_playing_this_round",
    "chance_of_playing_next_round", "news", "news_added", "form", "ep_this",
    "ep_next", "penalties_order", "corners_and_indirect_freekicks_order",
    "direct_freekicks_order", "price_change_hourly_rate", "opta_code",
]


def fetch(url: str) -> dict | list:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def snapshot(out_dir: Path = Path("data/raw/snapshots")) -> dict:
    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    day = ts.strftime("%Y-%m-%d")

    result = {"timestamp": ts.isoformat(), "files": []}

    for name, url in ENDPOINTS.items():
        payload = fetch(url)
        raw_dir = out_dir / name / f"date={day}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{stamp}.json.gz"

        import gzip
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        result["files"].append(str(path))
        result[name] = len(payload) if isinstance(payload, list) else None

    # Flatten the volatile element fields to Parquet -- this is the table the
    # ownership/price time series gets built from, and it stays small.
    boot = fetch(ENDPOINTS["bootstrap"])
    els = pd.DataFrame(boot["elements"])
    cols = [c for c in VOLATILE if c in els.columns]
    snap = els[cols].copy()
    snap["snapshot_ts"] = ts
    # Identify which GW this snapshot precedes, so it is point-in-time joinable.
    nxt = [e["id"] for e in boot["events"] if e.get("is_next")]
    cur = [e["id"] for e in boot["events"] if e.get("is_current")]
    snap["next_gw"] = nxt[0] if nxt else None
    snap["current_gw"] = cur[0] if cur else 0

    pq_dir = out_dir / "elements_volatile"
    pq_dir.mkdir(parents=True, exist_ok=True)
    pq = pq_dir / f"{stamp}.parquet"
    snap.to_parquet(pq, index=False)
    result["files"].append(str(pq))
    result["elements"] = len(snap)
    result["next_gw"] = snap["next_gw"].iloc[0]
    return result


if __name__ == "__main__":
    r = snapshot()
    print(f"snapshot {r['timestamp']}")
    print(f"  elements: {r['elements']}  fixtures: {r['fixtures']}  next_gw: {r['next_gw']}")
    for f in r["files"]:
        print(f"  wrote {f}")
