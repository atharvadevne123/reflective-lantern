"""Shared fixtures: a small corpus exercising every pipeline stage."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.pipeline import RagPipeline  # noqa: E402

NOW = datetime.now(UTC)


CORPUS = [
    {
        "source": "solar-policy.md",
        "text": (
            "# Solar Energy Policy\n\n"
            "The rooftop solar rebate program launched in March 2024. "
            "Households receive a rebate of 900 dollars per installed kilowatt. "
            "The maximum rebate per household is 5400 dollars. "
            "Applications are processed within 30 business days. "
            "The program is administered by the Department of Energy Transition."
        ),
        "created_at": (NOW - timedelta(days=30)).isoformat(),
    },
    {
        "source": "battery-guide.txt",
        "text": (
            "Home battery storage systems qualify for an additional incentive. "
            "The battery incentive pays 250 dollars per kilowatt hour of usable capacity. "
            "Batteries must be installed by an accredited installer to qualify. "
            "The battery incentive is capped at 3500 dollars per property. "
            "Warranty requirements mandate a minimum of ten years of coverage."
        ),
        "created_at": (NOW - timedelta(days=90)).isoformat(),
    },
    {
        "source": "grid-rules.txt",
        "text": (
            "Grid connection rules require an export limit of five kilowatts for "
            "single phase connections. Three phase connections may export up to "
            "fifteen kilowatts. Smart inverters must comply with the 2020 standard. "
            "Export limits are reviewed annually by the network operator."
        ),
        "created_at": (NOW - timedelta(days=200)).isoformat(),
    },
]


@pytest.fixture
def config() -> Settings:
    return Settings()


@pytest.fixture
def pipeline(config: Settings) -> RagPipeline:
    """A pipeline pre-loaded with the test corpus (extractive generator)."""
    pipe = RagPipeline(config=config)
    for doc in CORPUS:
        report = pipe.ingest(
            doc["text"].encode("utf-8"),
            source=doc["source"],
            created_at=doc["created_at"],
        )
        assert report["status"] == "ingested", report
    return pipe


@pytest.fixture
def empty_pipeline(config: Settings) -> RagPipeline:
    return RagPipeline(config=config)
