#!/usr/bin/env python3
"""Inject the combined JSON into the dashboard template."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "scripts" / "combined_template.html"
DATA = ROOT / "data" / "features" / "combined.json"
OUT = ROOT / "dashboard.html"

def main() -> None:
    tpl = TPL.read_text(encoding="utf-8")
    method = (ROOT / "scripts" / "methodology.html").read_text(encoding="utf-8")
    # Wide tables must scroll inside their own container, never push the page
    # sideways. Wrapping at build time keeps the source readable as plain markup.
    method = re.sub(r"<table>(.*?)</table>",
                    r'<div class="tscroll"><table>\1</table></div>',
                    method, flags=re.S)
    tpl = tpl.replace("__METHOD__", method)
    data = json.dumps(json.loads(DATA.read_text(encoding="utf-8")), ensure_ascii=True)
    out = tpl.replace("__DATA__", data)
    assert "__DATA__" not in out
    bad = sorted({c for c in out if ord(c) > 127})
    assert not bad, f"non-ASCII leaked: {bad}"
    OUT.write_text(out, encoding="utf-8")
    print(f"built {OUT} ({len(out)//1024} KB)")

if __name__ == "__main__":
    main()
