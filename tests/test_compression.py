"""Tests for app.compression module."""

from __future__ import annotations

import pytest

from app.compression import (
    compress_and_measure,
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


class TestCompressAndMeasure:
    def test_returns_expected_keys(self):
        result = compress_and_measure(b"hello world " * 100)
        assert set(result.keys()) == {"original_bytes", "compressed_bytes", "ratio", "savings_pct"}

    def test_savings_positive_for_repetitive(self):
        result = compress_and_measure(b"abcabc" * 200)
        assert result["savings_pct"] > 0

    def test_original_bytes_matches_input(self):
        data = b"x" * 500
        result = compress_and_measure(data)
        assert result["original_bytes"] == 500

    @pytest.mark.parametrize("method", ["gzip", "zlib"])
    def test_both_methods_accepted(self, method):
        result = compress_and_measure(b"data " * 100, method=method)
        assert result["ratio"] < 1.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            compress_and_measure(b"data", method="bzip2")
