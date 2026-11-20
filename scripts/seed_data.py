#!/usr/bin/env python3
"""Seed development data for local testing.

Populates in-memory registries with representative fixtures so developers
can exercise the full application flow without external dependencies.

Usage::

    python scripts/seed_data.py [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("seed")


def seed_model_registry() -> None:
    from app.model_registry import ModelRegistry

    reg = ModelRegistry()
    reg.register(
        name="churn_classifier",
        version="1.0.0",
        artifact_path="s3://models/churn/v1",
        framework="sklearn",
        metrics={"auc": 0.87, "f1": 0.81},
    )
    reg.transition("churn_classifier", "1.0.0", "STAGING")
    reg.transition("churn_classifier", "1.0.0", "PRODUCTION")
    reg.add_tag("churn_classifier", "1.0.0", "owner", "ml-team")
    logger.info("Seeded model_registry: churn_classifier v1.0.0 (PRODUCTION)")


def seed_feature_store() -> None:
    from app.feature_store import FeatureStore

    store = FeatureStore()
    store.publish(
        name="user_features",
        version="2026-08-24",
        features={"age_bucket": "int", "spend_30d": "float", "country": "str"},
        description="Standard user feature set for churn model",
    )
    store.publish(
        name="item_features",
        version="2026-08-24",
        features={"category": "str", "price": "float", "rating": "float"},
        description="Product catalogue features",
    )
    logger.info("Seeded feature_store: user_features, item_features")


def seed_experiment_tracker() -> None:
    from app.experiment_tracker import ExperimentRegistry

    reg = ExperimentRegistry()
    reg.register("homepage_cta", variants=["control", "variant_a", "variant_b"])
    reg.register(
        "checkout_flow",
        variants=["old", "new"],
        weights=[0.8, 0.2],
    )
    logger.info("Seeded experiment_tracker: homepage_cta, checkout_flow")


def seed_audit_log() -> None:
    from app.audit_log import AuditLog

    log = AuditLog()
    log.record("system", "startup", "app", note="seed run")
    log.record("alice", "login", "/session", ip="127.0.0.1")
    log.record("bob", "update", "model/churn_classifier", version="1.0.0")
    logger.info("Seeded audit_log with %d entries", len(log))


SEED_FUNCTIONS = [
    seed_model_registry,
    seed_feature_store,
    seed_experiment_tracker,
    seed_audit_log,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed development data")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    errors = []
    for fn in SEED_FUNCTIONS:
        try:
            fn()
        except Exception as exc:
            logger.error("Seed function %s failed: %s", fn.__name__, exc)
            errors.append(exc)

    if errors:
        sys.exit(1)
    logger.info("Seeding complete.")


if __name__ == "__main__":
    main()
