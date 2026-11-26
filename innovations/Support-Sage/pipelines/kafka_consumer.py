"""Kafka consumer for real-time ticket ingestion into Support-Sage."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaConsumer

    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False


def consume_tickets(
    topic: str = "helpdesk-tickets",
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "support-sage",
) -> None:
    """Consume ticket events from Kafka and run inference."""
    if not _KAFKA_AVAILABLE:
        logger.warning("kafka-python not installed - Kafka consumer disabled")
        return
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[bootstrap_servers],
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    logger.info("Kafka consumer started on topic '%s'", topic)
    for message in consumer:
        ticket: dict[str, Any] = message.value
        logger.info("Received ticket: %s", ticket.get("ticket_id", "unknown"))
