"""Data compression and decompression utilities using zlib and gzip."""

from __future__ import annotations

import gzip
import io
import json
import logging
import zlib
from typing import Any

logger = logging.getLogger(__name__)


def zlib_compress(data: bytes, level: int = 6) -> bytes:
    """Compress bytes using zlib deflate.

    Args:
        data: Raw bytes to compress.
        level: Compression level 0-9 (default 6).

    Returns:
        Compressed bytes.
    """
    return zlib.compress(data, level)


def zlib_decompress(data: bytes) -> bytes:
    """Decompress zlib-compressed bytes.

    Args:
        data: Compressed bytes.

    Returns:
        Decompressed bytes.

    Raises:
        ValueError: On decompression failure.
    """
    try:
        return zlib.decompress(data)
    except zlib.error as exc:
        raise ValueError(f"zlib decompression failed: {exc}") from exc


def gzip_compress(data: bytes, level: int = 6) -> bytes:
    """Compress bytes using gzip format.

    Args:
        data: Raw bytes to compress.
        level: Compression level 0-9 (default 6).

    Returns:
        gzip-compressed bytes.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as gz:
        gz.write(data)
    return buf.getvalue()


def gzip_decompress(data: bytes) -> bytes:
    """Decompress gzip-compressed bytes.

    Args:
        data: gzip-compressed bytes.

    Returns:
        Decompressed bytes.

    Raises:
        ValueError: On decompression failure.
    """
    try:
        buf = io.BytesIO(data)
        with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
            return gz.read()
    except (OSError, EOFError) as exc:
        raise ValueError(f"gzip decompression failed: {exc}") from exc


def compress_json(obj: Any, method: str = "gzip", level: int = 6) -> bytes:
    """Serialise obj to JSON and compress the result.

    Args:
        obj: Any JSON-serialisable object.
        method: ``'gzip'`` or ``'zlib'``.
        level: Compression level 0-9.

    Returns:
        Compressed bytes.

    Raises:
        ValueError: For unknown method.
    """
    raw = json.dumps(obj, separators=(",", ":"), default=str).encode()
    if method == "gzip":
        compressed = gzip_compress(raw, level)
    elif method == "zlib":
        compressed = zlib_compress(raw, level)
    else:
        raise ValueError(f"Unknown compression method: {method}")
    ratio = len(compressed) / len(raw) if raw else 1.0
    logger.debug("%s compressed %d -> %d bytes (%.1f%%)", method, len(raw), len(compressed), ratio * 100)
    return compressed


def decompress_json(data: bytes, method: str = "gzip") -> Any:
    """Decompress bytes and deserialise from JSON.

    Args:
        data: Compressed bytes.
        method: ``'gzip'`` or ``'zlib'``.

    Returns:
        The deserialised Python object.
    """
    if method == "gzip":
        raw = gzip_decompress(data)
    elif method == "zlib":
        raw = zlib_decompress(data)
    else:
        raise ValueError(f"Unknown compression method: {method}")
    return json.loads(raw)


def compression_ratio(original: bytes | str, method: str = "gzip") -> float:
    """Return the compression ratio for the given data.

    Args:
        original: Original data (bytes or str).
        method: Compression method to use.

    Returns:
        Float ratio compressed_size / original_size (lower is better).
    """
    raw = original.encode() if isinstance(original, str) else original
    if not raw:
        return 1.0
    compressed = compress_json(raw.decode(errors="replace"), method=method)
    return len(compressed) / len(raw)


__all__ = [
    "compress_json",
    "compression_ratio",
    "decompress_json",
    "gzip_compress",
    "gzip_decompress",
    "zlib_compress",
    "zlib_decompress",
]
