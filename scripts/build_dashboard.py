#!/usr/bin/env python3
"""Inject the projection JSON into the dashboard template.

The page must be fully self-contained (the Artifact CSP blocks every external
request), so the data is embedded rather than fetched. `ensure_ascii` keeps the
output pure ASCII, which means the page renders identically regardless of what
charset the host declares -- accented player names come through as \\uXXXX
escapes instead of mojibake.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "scripts" / "dashboard_template.html"
DATA = ROOT / "data" / "features" / "dashboard.json"
OUT = ROOT / "dashboard.html"

def main() -> None:
    tpl = TPL.read_text(encoding="utf-8")
    data = json.dumps(json.loads(DATA.read_text(encoding="utf-8")), ensure_ascii=True)
    out = tpl.replace("__DATA__", data)
    assert "__DATA__" not in out
    bad = sorted({c for c in out if ord(c) > 127})
    assert not bad, f"non-ASCII leaked into output: {bad}"
    OUT.write_text(out, encoding="utf-8")
    print(f"built {OUT} ({len(out)//1024} KB)")

if __name__ == "__main__":
    main()
