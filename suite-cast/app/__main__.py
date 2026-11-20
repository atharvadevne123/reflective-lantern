"""Run the Suite-Cast API with `python -m app`."""

from __future__ import annotations

import uvicorn

from .config import settings


def main() -> None:
    """Start uvicorn with settings-driven configuration."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # noqa: S104
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
