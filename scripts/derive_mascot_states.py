#!/usr/bin/env python
"""Derive per-state animated mascot assets from the editable source.

The source (static/assets/lare-concept.svg) is the full animated mascot in
demo mode.  Each runtime asset keeps the source's shared defs and animation
CSS plus exactly one state group, pinned visible by the source's own
`svg[data-state=...]` selector.  Usage:

    .venv/bin/python scripts/derive_mascot_states.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "static" / "assets" / "lare-concept.svg"
TARGETS = {"idle": "idle", "listening": "listen", "thinking": "think",
           "speaking": "speak", "error": "error"}

# In runtime mode the source's reduced-motion rules would hide every state but
# idle; drop the two opacity lines so the data-state selector always wins.
_REDUCED_MOTION_FIX = (
    "  #idle-state { opacity: 1; }\n"
    "  #listen-state, #think-state, #speak-state, #error-state { opacity: 0; }\n"
)


def extract_group(text: str, design: str) -> str:
    """Return the balanced <g id="{design}-state">...</g> block."""
    marker = f'<g id="{design}-state"'
    start = text.index(marker)
    depth = 0
    for match in re.finditer(r"</?g(?:\s[^>]*)?>", text[start:]):
        tag = match.group(0)
        if tag.startswith("</"):
            depth -= 1
            if depth == 0:
                return text[start:start + match.end()]
        elif not tag.endswith("/>"):
            depth += 1
    raise ValueError(f"unbalanced group {design}-state")


def derive(source: str, design: str) -> str:
    head = source[:source.index('<g id="idle-state"')].rstrip()
    head = head.replace('class="demo" data-state="idle"', f'data-state="{design}"')
    head = head.replace(_REDUCED_MOTION_FIX, "")
    group = extract_group(source, design)
    return f"{head}\n  {group}\n  <!-- Runtime asset: state {design} -->\n</svg>\n"


def main() -> int:
    source = SOURCE.read_text()
    for state, design in TARGETS.items():
        target = SOURCE.parent / f"lare-{state}.svg"
        target.write_text(derive(source, design))
        print(f"{target.name}: {target.stat().st_size} bytes (state={design})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
