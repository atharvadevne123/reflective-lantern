"""Generate the Ops-Vision architecture diagram as an SVG file."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).parent.parent / "screenshots" / "architecture.svg"

SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 560" width="900" height="560">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
    </marker>
    <style>
      .box {{ fill: #f1f5f9; stroke: #475569; stroke-width: 1.5; rx: 6; }}
      .ml   {{ fill: #ede9fe; stroke: #7c3aed; stroke-width: 1.5; rx: 6; }}
      .data {{ fill: #dbeafe; stroke: #2563eb; stroke-width: 1.5; rx: 6; }}
      .obs  {{ fill: #fef3c7; stroke: #d97706; stroke-width: 1.5; rx: 6; }}
      .lbl  {{ font: 600 13px system-ui, sans-serif; fill: #0f172a; text-anchor: middle; }}
      .sub  {{ font: 400 11px system-ui, sans-serif; fill: #475569; text-anchor: middle; }}
      .ttl  {{ font: 700 19px system-ui, sans-serif; fill: #0f172a; }}
      .sect {{ font: 700 12px system-ui, sans-serif; fill: #64748b; }}
      .edge {{ stroke: #64748b; stroke-width: 1.5; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>

  <text x="30" y="36" class="ttl">Ops-Vision — SRE Incident Prediction Architecture</text>

  <text x="30" y="76" class="sect">INGEST</text>
  <rect x="30" y="86" width="170" height="62" class="box"/>
  <text x="115" y="110" class="lbl">SRE Metrics</text>
  <text x="115" y="128" class="sub">cpu / mem / errors</text>
  <text x="115" y="142" class="sub">latency / rps / disk</text>

  <text x="250" y="76" class="sect">API LAYER</text>
  <rect x="250" y="86" width="190" height="62" class="box"/>
  <text x="345" y="106" class="lbl">FastAPI /api/v1</text>
  <text x="345" y="124" class="sub">predict · health · metrics</text>
  <text x="345" y="138" class="sub">forecast · runbooks/search</text>

  <rect x="250" y="168" width="190" height="46" class="box"/>
  <text x="345" y="188" class="lbl">Middleware</text>
  <text x="345" y="204" class="sub">correlation-id · rate limit</text>

  <text x="490" y="76" class="sect">FEATURE + MODEL</text>
  <rect x="490" y="86" width="180" height="62" class="ml"/>
  <text x="580" y="106" class="lbl">Feature Pipeline</text>
  <text x="580" y="124" class="sub">10 engineered features</text>
  <text x="580" y="138" class="sub">RobustScaler</text>

  <rect x="490" y="168" width="180" height="62" class="ml"/>
  <text x="580" y="188" class="lbl">Voting Ensemble</text>
  <text x="580" y="206" class="sub">XGBoost · LightGBM</text>
  <text x="580" y="220" class="sub">RandomForest (soft)</text>

  <text x="720" y="76" class="sect">RETRIEVAL</text>
  <rect x="720" y="86" width="150" height="62" class="ml"/>
  <text x="795" y="110" class="lbl">FAISS Index</text>
  <text x="795" y="128" class="sub">runbook RAG</text>
  <text x="795" y="142" class="sub">top-k similarity</text>

  <text x="30" y="290" class="sect">OBSERVABILITY</text>
  <rect x="30" y="300" width="200" height="62" class="obs"/>
  <text x="130" y="324" class="lbl">Drift Monitor</text>
  <text x="130" y="342" class="sub">KS-test, p &lt; 0.05</text>
  <text x="130" y="356" class="sub">sliding windows</text>

  <rect x="260" y="300" width="200" height="62" class="obs"/>
  <text x="360" y="324" class="lbl">Forecaster</text>
  <text x="360" y="342" class="sub">Holt exponential smoothing</text>
  <text x="360" y="356" class="sub">24h incident rate</text>

  <rect x="490" y="300" width="180" height="62" class="obs"/>
  <text x="580" y="324" class="lbl">Retrain DAG</text>
  <text x="580" y="342" class="sub">Airflow nightly 02:00</text>
  <text x="580" y="356" class="sub">AUC gate 0.70</text>

  <text x="30" y="420" class="sect">PERSISTENCE</text>
  <rect x="30" y="430" width="640" height="70" class="data"/>
  <text x="350" y="456" class="lbl">PostgreSQL via SQLAlchemy</text>
  <text x="350" y="476" class="sub">incidents · predictions · drift_alerts</text>
  <text x="350" y="492" class="sub">connection pooling (size 10, overflow 20) · indexed timestamps</text>

  <path d="M 200 117 L 246 117" class="edge"/>
  <path d="M 440 117 L 486 117" class="edge"/>
  <path d="M 670 117 L 716 117" class="edge"/>
  <path d="M 580 148 L 580 164" class="edge"/>
  <path d="M 345 148 L 345 164" class="edge"/>
  <path d="M 580 230 L 580 296" class="edge"/>
  <path d="M 345 214 L 345 296" class="edge"/>
  <path d="M 130 362 L 130 426" class="edge"/>
  <path d="M 360 362 L 360 426" class="edge"/>
  <path d="M 580 362 L 580 426" class="edge"/>
</svg>
"""


def generate_diagram(output_path: Path = OUTPUT_PATH) -> Path:
    """Write the architecture diagram SVG to disk.

    Args:
        output_path: Destination path for the SVG file.

    Returns:
        The path the diagram was written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(SVG_TEMPLATE, encoding="utf-8")
    logger.info("Architecture diagram written to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_diagram()
    print(f"Diagram generated: {path}")
