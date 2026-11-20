"""Generate Alpha-Scout architecture diagram as SVG."""
from __future__ import annotations

import os
from pathlib import Path


def generate_svg_diagram() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<svg width="900" height="520" xmlns="http://www.w3.org/2000/svg" font-family="monospace,sans-serif">
  <rect width="900" height="520" fill="#0f172a" rx="12"/>
  <text x="450" y="38" text-anchor="middle" fill="#f1f5f9" font-size="20" font-weight="bold">Alpha-Scout -- System Architecture</text>
  <rect x="30" y="65" width="150" height="50" rx="8" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="105" y="85" text-anchor="middle" fill="#93c5fd" font-size="11" font-weight="bold">Client</text>
  <text x="105" y="103" text-anchor="middle" fill="#bfdbfe" font-size="10">REST / JSON</text>
  <rect x="250" y="55" width="180" height="70" rx="8" fill="#1e3a5f" stroke="#6366f1" stroke-width="1.5"/>
  <text x="340" y="78" text-anchor="middle" fill="#a5b4fc" font-size="12" font-weight="bold">FastAPI (v1)</text>
  <text x="340" y="96" text-anchor="middle" fill="#c7d2fe" font-size="9">/predict /health /metrics /drift</text>
  <line x1="180" y1="90" x2="250" y2="90" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>
  <rect x="250" y="165" width="180" height="55" rx="8" fill="#134e3a" stroke="#34d399" stroke-width="1.5"/>
  <text x="340" y="186" text-anchor="middle" fill="#6ee7b7" font-size="12" font-weight="bold">Feature Pipeline</text>
  <text x="340" y="204" text-anchor="middle" fill="#a7f3d0" font-size="9">StockFeatureEngineer (20+)</text>
  <rect x="510" y="145" width="180" height="75" rx="8" fill="#3b1f61" stroke="#a855f7" stroke-width="1.5"/>
  <text x="600" y="168" text-anchor="middle" fill="#d8b4fe" font-size="12" font-weight="bold">XGBoost Ensemble</text>
  <text x="600" y="186" text-anchor="middle" fill="#e9d5ff" font-size="9">sklearn Pipeline + Scaler</text>
  <text x="600" y="200" text-anchor="middle" fill="#e9d5ff" font-size="9">5-fold CV AUC scoring</text>
  <rect x="510" y="255" width="180" height="55" rx="8" fill="#3b2e0a" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="600" y="275" text-anchor="middle" fill="#fde68a" font-size="12" font-weight="bold">Drift Monitor</text>
  <text x="600" y="293" text-anchor="middle" fill="#fef3c7" font-size="9">KS-test per feature</text>
  <rect x="250" y="340" width="180" height="55" rx="8" fill="#1e293b" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="340" y="360" text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="bold">SQLAlchemy ORM</text>
  <rect x="510" y="340" width="180" height="55" rx="8" fill="#1a3a1a" stroke="#22c55e" stroke-width="1.5"/>
  <text x="600" y="360" text-anchor="middle" fill="#86efac" font-size="12" font-weight="bold">Retrain DAG</text>
  <text x="600" y="378" text-anchor="middle" fill="#bbf7d0" font-size="9">Airflow @weekly</text>
  <rect x="720" y="65" width="155" height="55" rx="8" fill="#1a1a2e" stroke="#64748b" stroke-width="1.5"/>
  <text x="797" y="85" text-anchor="middle" fill="#94a3b8" font-size="12" font-weight="bold">GitHub Actions</text>
  <text x="797" y="103" text-anchor="middle" fill="#cbd5e1" font-size="9">ruff lint + pytest</text>
  <rect x="720" y="145" width="155" height="55" rx="8" fill="#0c2340" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="797" y="165" text-anchor="middle" fill="#7dd3fc" font-size="12" font-weight="bold">Docker Compose</text>
  <text x="797" y="183" text-anchor="middle" fill="#bae6fd" font-size="9">API + PostgreSQL</text>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
    </marker>
  </defs>
  <line x1="340" y1="125" x2="340" y2="165" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="430" y1="192" x2="510" y2="185" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="340" y1="220" x2="340" y2="340" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="600" y1="220" x2="600" y2="255" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="600" y1="310" x2="600" y2="340" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="430" y1="367" x2="510" y2="367" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="450" y="490" text-anchor="middle" fill="#475569" font-size="10">Alpha-Scout v1.0.0 FastAPI XGBoost sklearn SQLAlchemy Docker GitHub Actions</text>
</svg>"""


def main() -> None:
    os.makedirs("screenshots", exist_ok=True)
    svg_content = generate_svg_diagram()
    Path("screenshots/architecture.svg").write_text(svg_content)
    print("Architecture diagram saved to screenshots/architecture.svg")


if __name__ == "__main__":
    main()
