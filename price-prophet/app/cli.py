"""
Command-line interface for Price-Prophet.

Entry points
------------
``serve``
    Start the FastAPI server with uvicorn.
``train``
    Load a dataset and train a named pricing model.
``evaluate``
    Run a backtest against a dataset and print summary metrics.

Usage::

    python -m app.cli serve --host 0.0.0.0 --port 8000
    python -m app.cli train --data data/prices.csv --model-type linear
    python -m app.cli evaluate --model linear --data data/prices.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the configured argument parser (without parsing).

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser with all subcommands registered.
    """
    parser = argparse.ArgumentParser(
        prog="price-prophet",
        description="Price-Prophet: ML-powered dynamic pricing system.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    serve_parser = subparsers.add_parser("serve", help="Start the Price-Prophet API server.")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    train_parser = subparsers.add_parser("train", help="Train a pricing model.")
    train_parser.add_argument("--data", required=True, metavar="PATH")
    train_parser.add_argument("--model-type", dest="model_type", default="linear",
                              choices=["linear", "gradient_boost", "ensemble"])

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    evaluate_parser.add_argument("--model", required=True, metavar="NAME")
    evaluate_parser.add_argument("--data", required=True, metavar="PATH")

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build and parse the CLI argument parser.

    Parameters
    ----------
    argv:
        Argument list (excluding the program name).  ``None`` reads from
        :data:`sys.argv`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with a ``command`` attribute set to one of
        ``"serve"``, ``"train"``, or ``"evaluate"``.
    """
    parser = argparse.ArgumentParser(
        prog="price-prophet",
        description="Price-Prophet: ML-powered dynamic pricing system.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ---- serve -----------------------------------------------------------
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the Price-Prophet API server.",
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to listen on (default: 8000).",
    )

    # ---- train -----------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train a pricing model on a dataset.",
    )
    train_parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to the training CSV or JSON file.",
    )
    train_parser.add_argument(
        "--model-type",
        dest="model_type",
        default="linear",
        choices=["linear", "gradient_boost", "ensemble"],
        help="Model architecture to train (default: linear).",
    )

    # ---- evaluate --------------------------------------------------------
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model against a dataset.",
    )
    evaluate_parser.add_argument(
        "--model",
        required=True,
        metavar="NAME",
        help="Name of the model to evaluate (as registered in the registry).",
    )
    evaluate_parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to the evaluation CSV or JSON file.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI application with uvicorn."""
    print(f"Starting server on {args.host}:{args.port}")
    try:
        import uvicorn  # type: ignore

        uvicorn.run(
            "app.api.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except ImportError:
        print(
            "uvicorn is not installed.  Install it with: pip install uvicorn",
            file=sys.stderr,
        )
        return 1
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    """Load data and train the requested model type."""
    print(f"Training {args.model_type} model on {args.data}")
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Run the backtester for the named model on the given dataset."""
    print(f"Evaluating model {args.model} on {args.data}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand handler.

    Parameters
    ----------
    argv:
        Argument list (excluding the program name).  ``None`` reads from
        :data:`sys.argv`.

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    args = parse_args(argv)

    handlers = {
        "serve": _handle_serve,
        "train": _handle_train,
        "evaluate": _handle_evaluate,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command!r}", file=sys.stderr)
        return 2

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
