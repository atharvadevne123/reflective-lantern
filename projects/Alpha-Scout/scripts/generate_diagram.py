"""Generate Alpha-Scout architecture diagram as SVG."""
from __future__ import annotations

import os
from pathlib import Path


def generate_svg_diagram() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<svg width="900" height="520" xmlns="http://www.w3.org/2000/svg" font-family="monospace,sans-serif">
  <rect width="900" height="520" fill="#0f172a" rx="12"/>
  <text x="450" y="38" text-anchor="middle" fill="#f1f5f9" font-size="20" font-weight="bold">Alpha-Scout — System Architecture</text>
</svg>"""


def main() -> None:
    os.makedirs("screenshots", exist_ok=True)
    svg_content = generate_svg_diagram()
    Path("screenshots/architecture.svg").write_text(svg_content)
    print("Architecture diagram saved to screenshots/architecture.svg")


if __name__ == "__main__":
    main()
