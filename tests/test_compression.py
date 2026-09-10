"""Tests for app.compression module."""

from __future__ import annotations

import pytest

from app.compression import (
    compress_json,
    compression_ratio,
    decompress_json,
    gzip_compress,
    gzip_decompress,
    zlib_compress,
    zlib_decompress,
)

SAMPLE = b"hello world " * 100
SAMPLE_OBJ = {"key": "value", "numbers": list(range(100))}


class TestZlib:
    def test_roundtrip(self):
        assert zlib_decompress(zlib_compress(SAMPLE)) == SAMPLE

    def test_compressed_smaller(self):
        assert len(zlib_compress(SAMPLE)) < len(SAMPLE)

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError, match="zlib"):
            zlib_decompress(b"not-compressed")

    @pytest.mark.parametrize("level", [1, 6, 9])
    def test_compression_levels(self, level):
        compressed = zlib_compress(SAMPLE, level)
        assert zlib_decompress(compressed) == SAMPLE


class TestGzip:
    def test_roundtrip(self):
        assert gzip_decompress(gzip_compress(SAMPLE)) == SAMPLE

    def test_compressed_smaller(self):
        assert len(gzip_compress(SAMPLE)) < len(SAMPLE)

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError, match="gzip"):
            gzip_decompress(b"not-gzip-data")

    @pytest.mark.parametrize("level", [1, 6, 9])
    def test_compression_levels(self, level):
        compressed = gzip_compress(SAMPLE, level)
        assert gzip_decompress(compressed) == SAMPLE


class TestCompressJson:
    @pytest.mark.parametrize("method", ["gzip", "zlib"])
    def test_json_roundtrip(self, method):
        compressed = compress_json(SAMPLE_OBJ, method=method)
        result = decompress_json(compressed, method=method)
        assert result == SAMPLE_OBJ

    def test_unknown_method_raises_compress(self):
        with pytest.raises(ValueError, match="Unknown"):
            compress_json({}, method="brotli")

    def test_unknown_method_raises_decompress(self):
        with pytest.raises(ValueError, match="Unknown"):
            decompress_json(b"", method="brotli")

    def test_empty_object_roundtrip(self):
        compressed = compress_json({})
        assert decompress_json(compressed) == {}

    def test_nested_structure(self):
        obj = {"nested": {"a": [1, 2, 3]}, "flag": True}
        assert decompress_json(compress_json(obj)) == obj


class TestCompressionRatio:
    def test_ratio_below_one_for_repetitive(self):
        ratio = compression_ratio("hello " * 500)
        assert ratio < 1.0

    def test_ratio_positive(self):
        assert compression_ratio("data") > 0

    def test_empty_bytes_returns_one(self):
        assert compression_ratio(b"") == 1.0

    @pytest.mark.parametrize("method", ["gzip", "zlib"])
    def test_both_methods(self, method):
        ratio = compression_ratio("test " * 100, method=method)
        assert 0 < ratio < 1.0


class TestEdgeCases:
    def test_zlib_single_byte_roundtrip(self):
        data = b"\xff"
        assert zlib_decompress(zlib_compress(data)) == data

    def test_gzip_single_byte_roundtrip(self):
        data = b"\x00"
        assert gzip_decompress(gzip_compress(data)) == data

    def test_zlib_level_zero_no_compression(self):
        compressed = zlib_compress(SAMPLE, level=0)
        assert zlib_decompress(compressed) == SAMPLE

    def test_gzip_level_zero_no_compression(self):
        compressed = gzip_compress(SAMPLE, level=0)
        assert gzip_decompress(compressed) == SAMPLE

    def test_compress_json_with_list(self):
        obj = list(range(500))
        assert decompress_json(compress_json(obj)) == obj

    def test_compress_json_with_unicode(self):
        obj = {"emoji": "\U0001f600", "arabic": "مرحبا"}
        assert decompress_json(compress_json(obj)) == obj

    def test_compress_json_bool_values(self):
        obj = {"flag": True, "other": False, "none": None}
        result = decompress_json(compress_json(obj))
        assert result["flag"] is True
        assert result["other"] is False
        assert result["none"] is None

    def test_zlib_level9_smallest(self):
        c1 = zlib_compress(SAMPLE, level=1)
        c9 = zlib_compress(SAMPLE, level=9)
        assert len(c9) <= len(c1)

    def test_compression_ratio_string_input(self):
        ratio = compression_ratio("aaaa" * 500)
        assert ratio < 0.1  # highly compressible

    @pytest.mark.parametrize("obj", [[], {}, 0, False, ""])
    def test_compress_json_falsy_primitives(self, obj):
        assert decompress_json(compress_json(obj)) == obj


@pytest.mark.parametrize(
    "obj",
    [
        {"key": "value"},
        [1, 2, 3],
        {"nested": {"a": 1}},
        42,
        "string",
    ],
)
def test_compress_json_roundtrip_various_types(obj) -> None:
    """compress_json / decompress_json round-trips any JSON-serialisable value."""
    assert decompress_json(compress_json(obj)) == obj


@pytest.mark.parametrize("level", [1, 5, 9])
def test_gzip_compress_produces_smaller_output(level: int) -> None:
    """gzip_compress reduces the size of highly repetitive data at any level."""
    data = b"a" * 10_000
    compressed = gzip_compress(data, level=level)
    assert len(compressed) < len(data)


@pytest.mark.parametrize("n", [1, 10, 100])
def test_zlib_roundtrip_various_lengths(n: int) -> None:
    """zlib_compress / zlib_decompress round-trips data of arbitrary length."""
    data = b"x" * n
    assert zlib_decompress(zlib_compress(data)) == data


@pytest.mark.parametrize("level", [1, 5, 9])
def test_zlib_compress_produces_smaller_output_for_repetitive_data(level: int) -> None:
    """zlib compression reduces size of repetitive data at all levels."""
    data = b"a" * 5_000
    assert len(zlib_compress(data, level=level)) < len(data)


class TestCompressionRatioEdgeCases:
    def test_ratio_single_byte(self) -> None:
        ratio = compression_ratio(b"x")
        assert ratio > 0.0

    def test_ratio_unicode_string(self) -> None:
        ratio = compression_ratio("hello world " * 200)
        assert ratio < 1.0
