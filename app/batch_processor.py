"""Configurable batch processing pipeline for large dataset operations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchResult(Generic[R]):
    """Result of processing a single batch.

    Attributes:
        batch_index: Zero-based batch number.
        items_processed: Count of items in this batch.
        results: Output produced by the processor function.
        errors: Exceptions caught during processing (if error_handling='collect').
    """

    batch_index: int
    items_processed: int
    results: list[R]
    errors: list[Exception] = field(default_factory=list)


@dataclass
class RunSummary:
    """Aggregate summary for a complete batch run.

    Attributes:
        total_items: Total number of input items.
        total_batches: Number of batches processed.
        total_results: Number of output items produced.
        total_errors: Count of errors across all batches.
    """

    total_items: int
    total_batches: int
    total_results: int
    total_errors: int


class BatchProcessor(Generic[T, R]):
    """Processes an iterable in fixed-size batches.

    Args:
        processor: Callable that transforms a list of items into results.
        batch_size: Maximum items per batch.
        error_handling: ``'raise'`` re-raises immediately; ``'collect'`` captures
            errors per batch and continues.
        on_batch_done: Optional callback invoked after each :class:`BatchResult`.
    """

    def __init__(
        self,
        processor: Callable[[list[T]], list[R]],
        batch_size: int = 100,
        error_handling: str = "raise",
        on_batch_done: Callable[[BatchResult[R]], None] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if error_handling not in ("raise", "collect"):
            raise ValueError("error_handling must be 'raise' or 'collect'")
        self.processor = processor
        self.batch_size = batch_size
        self.error_handling = error_handling
        self.on_batch_done = on_batch_done

    @staticmethod
    def _chunk(items: list[T], size: int) -> Iterator[list[T]]:
        """Yield successive non-overlapping slices of *items* each of length *size*."""
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def run(self, items: list[T]) -> RunSummary:
        """Process all items in batches.

        Args:
            items: Input items to process.

        Returns:
            :class:`RunSummary` with aggregate counts.
        """
        total_results = 0
        total_errors = 0
        batch_index = 0

        for batch in self._chunk(items, self.batch_size):
            errors: list[Exception] = []
            results: list[R] = []
            try:
                results = self.processor(batch)
            except Exception as exc:
                if self.error_handling == "raise":
                    raise
                errors.append(exc)
                logger.error("Batch %d failed: %s", batch_index, exc)

            br = BatchResult(
                batch_index=batch_index,
                items_processed=len(batch),
                results=results,
                errors=errors,
            )
            total_results += len(results)
            total_errors += len(errors)
            if self.on_batch_done:
                try:
                    self.on_batch_done(br)
                except Exception as cb_exc:
                    logger.warning("on_batch_done callback failed: %s", cb_exc)
            batch_index += 1
            logger.debug(
                "Batch %d: %d in, %d out, %d errors",
                batch_index,
                len(batch),
                len(results),
                len(errors),
            )

        return RunSummary(
            total_items=len(items),
            total_batches=batch_index,
            total_results=total_results,
            total_errors=total_errors,
        )


__all__ = ["BatchProcessor", "BatchResult", "RunSummary"]
