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
import sys
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


def check_gap(out_dir: Path = Path("data/raw/snapshots"), max_days: int = 1) -> dict:
    """Warn if the snapshot series has a hole in it.

    Silent snapshot loss is the one failure here with no recovery path -- no
    upstream source backfills price or ownership. A failure that goes unnoticed
    for a week costs a week of history permanently, so the check runs on every
    invocation and shouts rather than logging quietly.
    """
    pq_dir = out_dir / "elements_volatile"
    files = sorted(pq_dir.glob("*.parquet")) if pq_dir.exists() else []
    if not files:
        return {"ok": False, "reason": "no snapshots on disk", "gap_days": None}

    stamps = []
    for f in files:
        try:
            stamps.append(datetime.strptime(f.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))
        except ValueError:
            continue
    stamps.sort()
    now = datetime.now(timezone.utc)
    since_last = (now - stamps[-1]).total_seconds() / 86400.0

    # Interior holes matter as much as staleness: a run that died for three days
    # and recovered still lost three days.
    holes = [(a.date().isoformat(), b.date().isoformat())
             for a, b in zip(stamps, stamps[1:])
             if (b - a).total_seconds() / 86400.0 > max_days + 0.25]

    ok = since_last <= max_days + 0.25 and not holes
    return {"ok": ok, "n_snapshots": len(stamps), "gap_days": round(since_last, 2),
            "last": stamps[-1].isoformat(), "holes": holes}


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
    gap = check_gap()
    if not gap["ok"]:
        print(f"!! SNAPSHOT GAP: last {gap.get('last')} "
              f"({gap.get('gap_days')}d ago), holes={gap.get('holes')}", file=sys.stderr)
    else:
        print(f"  gap check ok: {gap['n_snapshots']} snapshots, "
              f"newest {gap['gap_days']}d old")
    print(f"snapshot {r['timestamp']}")
    print(f"  elements: {r['elements']}  fixtures: {r['fixtures']}  next_gw: {r['next_gw']}")
    for f in r["files"]:
        print(f"  wrote {f}")
