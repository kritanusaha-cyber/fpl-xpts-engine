#!/usr/bin/env python3
"""Inject the simulation JSON into the graphs dashboard template."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "scripts" / "simulation_template.html"
DATA = ROOT / "data" / "features" / "simulation.json"
OUT = ROOT / "simulation.html"

def main() -> None:
    tpl = TPL.read_text(encoding="utf-8")
    data = json.dumps(json.loads(DATA.read_text(encoding="utf-8")), ensure_ascii=True)
    out = tpl.replace("__DATA__", data)
    assert "__DATA__" not in out
    bad = sorted({c for c in out if ord(c) > 127})
    assert not bad, f"non-ASCII leaked: {bad}"
    OUT.write_text(out, encoding="utf-8")
    print(f"built {OUT} ({len(out)//1024} KB)")

if __name__ == "__main__":
    main()
