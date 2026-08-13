"""Tests for app/cli.py."""
from __future__ import annotations

import pytest


def test_build_parser_returns_parser():
    from app.cli import build_parser
    parser = build_parser()
    assert parser is not None


def test_parser_train_subcommand():
    from app.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["train", "--data", "/tmp/data.csv"])
    assert args.command == "train"
    assert args.data == "/tmp/data.csv"


def test_parser_serve_subcommand():
    from app.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["serve", "--port", "9000"])
    assert args.command == "serve"
    assert args.port == 9000


def test_parser_no_subcommand_fails():
    from app.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
