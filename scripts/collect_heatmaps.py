"""Collect season touch heatmaps for every dashboard player FotMob knows."""
import gzip, glob, json
import pandas as pd
from fpl.ingest.fotmob_heatmap import fetch, zone_shares

m = pd.read_parquet("data/raw/fotmob/id_map.parquet")
e = pd.read_parquet("data/raw/fotmob/element_code_2026_27.parquet")
d = json.load(open("data/features/combined.json"))
ids = {p["id"] for p in d["players"]}
t = m.merge(e, on="code").query("id in @ids").drop_duplicates("player_id")
print(f"collecting {len(t)} players", flush=True)

touches = fetch(t.player_id.astype(int).tolist())
touches.to_parquet("data/raw/fotmob/touches.parquet", index=False)
z = zone_shares(touches)
z = z.merge(t[["player_id", "code", "name"]], on="player_id", how="left")
z.to_parquet("data/features/heatmap_zones.parquet", index=False)
print(f"done: {len(z)} players, {len(touches)} touches", flush=True)
