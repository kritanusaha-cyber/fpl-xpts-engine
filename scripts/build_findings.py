"""Render FINDINGS.md as a page on the site.

The file is 1,500 lines of results that were only readable by opening the
repo. Rendering it puts the evidence next to the thing it is evidence about.
"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "FINDINGS.md"
OUT = ROOT / "findings.html"

CSS = """
:root{--ground:#F6F7F3;--surface:#FFF;--ink:#171A13;--ink-2:#3D4436;--muted:#6C7364;
  --line:#E2E5DC;--accent:#9A6A11;--pos:#2F6B3A;--neg:#9A3412;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#121410;--surface:#191C15;--ink:#E7EAE0;--ink-2:#BFC5B5;--muted:#8A9180;
  --line:#2B3025;--accent:#D9A33A;--pos:#7FB069;--neg:#E07A5F}}
:root[data-theme="dark"]{--ground:#121410;--surface:#191C15;--ink:#E7EAE0;--ink-2:#BFC5B5;
  --muted:#8A9180;--line:#2B3025;--accent:#D9A33A;--pos:#7FB069;--neg:#E07A5F}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:14.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:44px 22px 90px}
.back{font-size:12.5px;color:var(--muted);text-decoration:none;font-family:var(--mono)}
.back:hover{color:var(--accent)}
h1{font-family:var(--mono);font-size:21px;font-weight:600;letter-spacing:-.01em;
  margin:26px 0 6px;padding-bottom:10px;border-bottom:1px solid var(--line)}
h2{font-size:17px;margin:38px 0 8px;letter-spacing:-.01em}
h3{font-size:14.5px;margin:26px 0 6px;color:var(--ink-2)}
p{margin:10px 0}
strong{color:var(--ink);font-weight:600}
em{color:var(--ink-2)}
code{font-family:var(--mono);font-size:12.5px;background:var(--surface);
  border:1px solid var(--line);border-radius:2px;padding:1px 4px}
pre{background:var(--surface);border:1px solid var(--line);border-radius:3px;
  padding:12px 14px;overflow-x:auto}
pre code{border:0;padding:0;background:none}
hr{border:0;border-top:1px solid var(--line);margin:40px 0}
blockquote{margin:12px 0;padding:2px 0 2px 14px;border-left:2px solid var(--line);
  color:var(--ink-2)}
ul,ol{padding-left:20px;margin:10px 0}
li{margin:3px 0}
/* Wide tables scroll inside their own box so the page never scrolls sideways. */
.tscroll{overflow-x:auto;margin:14px 0;border:1px solid var(--line);border-radius:3px;
  background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 11px;text-align:left;border-bottom:1px solid var(--line);
  white-space:nowrap}
th{font-family:var(--mono);font-size:11.5px;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0}
/* Numeric columns read better right-aligned and monospaced. */
td:not(:first-child){font-family:var(--mono);font-size:12.5px;text-align:right}
th:not(:first-child){text-align:right}
"""


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    # every table gets its own horizontal scroll container
    html = re.sub(r"<table>", '<div class="tscroll"><table>', html)
    html = re.sub(r"</table>", "</table></div>", html)
    OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Findings &middot; FPL xPts Engine</title>
<style>{CSS}</style></head><body>
<div class="wrap">
<a class="back" href="index.html">&larr; FPL xPts Engine</a>
<span style="color:var(--muted);opacity:.5;margin:0 5px">&middot;</span>
<a class="back" href="dashboard.html">Dashboard</a>
<span style="color:var(--muted);opacity:.5;margin:0 5px">&middot;</span>
<a class="back" href="paper/aiaa_paper.html">Technical paper</a>
{html}
</div></body></html>""", encoding="utf-8")
    print(f"built {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
