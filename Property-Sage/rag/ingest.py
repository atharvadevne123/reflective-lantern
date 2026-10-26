"""Ingest neighbourhood market reports into the RAG knowledge base.

Each document is a short paragraph describing a neighbourhood's character,
typical price band, and rental demand. In production these would be loaded
from a database or scraped from market reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NEIGHBOURHOOD_DOCS: list[dict[str, str]] = [
    {
        "id": "downtown",
        "text": (
            "Downtown is the most densely developed neighbourhood with the highest price per sqft. "
            "Properties are predominantly condos and apartments built between 1980 and 2020. "
            "Rental demand is very strong due to office proximity, yielding 4.5–6% gross annually."
        ),
    },
    {
        "id": "waterfront",
        "text": (
            "Waterfront commands a 80% premium over the city average. Luxury houses and high-rise "
            "condos dominate. Rental yields are lower (3–4.5%) but capital appreciation averages "
            "6% per year. Strong demand from high-income renters and short-term letting platforms."
        ),
    },
    {
        "id": "suburb",
        "text": (
            "Suburb is the reference neighbourhood for family homes. Median price sits near the "
            "city average. Three-bedroom houses built in the 1990s make up the majority of stock. "
            "Rental yields of 4.8–5.5% are the most stable in the market."
        ),
    },
    {
        "id": "midtown",
        "text": (
            "Midtown is a mixed-use area popular with young professionals. A 30% premium over suburb "
            "reflects proximity to transit and amenities. Townhouses and condos are typical. "
            "Gross rental yields range from 4.2–5.2%."
        ),
    },
    {
        "id": "uptown",
        "text": (
            "Uptown is an established residential area with well-maintained detached houses. "
            "Properties command a 20% premium over the city average. Rental yields are moderate "
            "at 4–5%, but vacancy rates are very low at under 2%."
        ),
    },
    {
        "id": "historic",
        "text": (
            "Historic district properties carry a heritage premium of 10% over comparable houses "
            "elsewhere. Renovation costs can be high due to preservation requirements. Rental "
            "demand from academics and professionals is consistent, yielding 4.5–5.5%."
        ),
    },
    {
        "id": "university",
        "text": (
            "University neighbourhood is dominated by student accommodation. Studios and 1-bedroom "
            "apartments are priced 5% below the city average but generate the highest gross yields "
            "in the market at 6.5–8% due to consistent occupancy during academic terms."
        ),
    },
    {
        "id": "airport",
        "text": (
            "Airport zone offers the lowest entry prices at 15% below average. It attracts "
            "logistics workers and short-term corporate lettings. Gross rental yields are "
            "6–7% but capital appreciation is muted. Noise blight must be factored in."
        ),
    },
    {
        "id": "industrial",
        "text": (
            "Industrial fringe has seen recent residential development targeting first-time buyers. "
            "Prices are 25% below the city average. Rental yields of 6.5–7.5% are attractive, "
            "though tenants are sensitive to the industrial character of adjacent lots."
        ),
    },
    {
        "id": "rural",
        "text": (
            "Rural properties offer the most affordable entry point at 35% below city average. "
            "Land and lot sizes are large. Rental demand is thin and seasonal, with gross yields "
            "of 3.5–5%. Best suited to owner-occupiers or holiday-let investors."
        ),
    },
]

CORPUS_PATH: Path = Path("rag/corpus.json")


def ingest_documents(output_path: Path = CORPUS_PATH) -> int:
    """Write neighbourhood documents to the corpus JSON file.

    Args:
        output_path: Destination file for the serialised corpus.

    Returns:
        Number of documents ingested.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(NEIGHBOURHOOD_DOCS, indent=2))
    logger.info("Ingested %d neighbourhood documents to %s", len(NEIGHBOURHOOD_DOCS), output_path)
    return len(NEIGHBOURHOOD_DOCS)


def load_documents(corpus_path: Path = CORPUS_PATH) -> list[dict[str, str]]:
    """Load the corpus JSON from disk.

    Args:
        corpus_path: Path to the corpus JSON file.

    Returns:
        List of document dicts with 'id' and 'text' keys.
    """
    if not corpus_path.exists():
        logger.info("Corpus not found — running ingest")
        ingest_documents(corpus_path)
    return json.loads(corpus_path.read_text())
