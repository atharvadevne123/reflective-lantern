"""Feature engineering pipeline for Ticket-Oracle.

Builds a ColumnTransformer that combines TF-IDF text features from ticket
descriptions with ordinal-encoded categorical fields and scaled numeric
workload metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

PRIORITY_MAP: dict[str, int] = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
PRIORITY_INV: dict[int, str] = {v: k for k, v in PRIORITY_MAP.items()}

TIER_ORDER = [["Bronze", "Standard", "Silver", "Gold"]]
CHANNEL_ORDER = [["portal", "email", "chat", "phone"]]

CATEGORICAL_COLS = ["department", "incident_type", "channel", "customer_tier", "product_area"]
NUMERIC_COLS = [
    "open_tickets_count",
    "agent_load",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_month_end",
    "ticket_age_minutes",
    "same_user_last_7d",
]
TEXT_COL = "description"

DEPARTMENTS = [
    "Engineering", "Finance", "HR", "IT", "Legal",
    "Marketing", "Operations", "Sales", "Support", "Unknown",
]
INCIDENT_TYPES = [
    "Access", "Application", "Hardware", "Network",
    "Security", "Service Request", "Software", "Unknown",
]
CHANNELS = ["chat", "email", "phone", "portal"]
TIERS = ["Bronze", "Gold", "Silver", "Standard"]
PRODUCT_AREAS = [
    "CRM", "Database", "ERP", "Infrastructure",
    "Networking", "Security", "Workstation", "Unknown",
]


def make_feature_pipeline() -> ColumnTransformer:
    """Build the ColumnTransformer that encodes all feature groups.

    Returns:
        A fitted-ready ColumnTransformer combining TF-IDF, ordinal encoding,
        and standard scaling for the three feature groups.
    """
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=80, stop_words="english", ngram_range=(1, 2))),
    ])

    cat_encoder = OrdinalEncoder(
        categories=[
            DEPARTMENTS,
            INCIDENT_TYPES,
            CHANNELS,
            TIERS,
            PRODUCT_AREAS,
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    transformer = ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_COL),
            ("cat", cat_encoder, CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )
    return transformer


def engineer_features(raw: dict[str, Any]) -> dict[str, Any]:
    """Derive lag/rolling/ratio features from raw ticket payload.

    Args:
        raw: Dictionary of raw ticket fields.

    Returns:
        Enriched dictionary ready for the ColumnTransformer.
    """
    out = dict(raw)

    dow = int(raw.get("day_of_week", 0))
    day = int(raw.get("day_of_month", 15))

    out["is_weekend"] = int(dow >= 5)
    out["is_month_end"] = int(day >= 28)
    out.setdefault("ticket_age_minutes", 0)
    out.setdefault("same_user_last_7d", 0)
    out.setdefault("open_tickets_count", 10)
    out.setdefault("agent_load", 5)
    out.setdefault("description", "")
    out.setdefault("department", "Unknown")
    out.setdefault("incident_type", "Unknown")
    out.setdefault("channel", "email")
    out.setdefault("customer_tier", "Standard")
    out.setdefault("product_area", "Unknown")

    return out


def payload_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a single prediction payload into a one-row DataFrame.

    Args:
        payload: Enriched ticket dictionary from engineer_features.

    Returns:
        One-row DataFrame with all expected columns.
    """
    enriched = engineer_features(payload)
    cols = [TEXT_COL] + CATEGORICAL_COLS + NUMERIC_COLS
    row = {c: enriched.get(c, 0 if c in NUMERIC_COLS else "") for c in cols}
    df = pd.DataFrame([row])
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    return df


def make_synthetic_dataset(n_samples: int = 2000, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a synthetic IT helpdesk dataset with realistic label signal.

    Priority is driven by a weighted latent score over tier, incident_type,
    open_tickets_count, and hour_of_day.  SLA breach is correlated with
    priority and workload.

    Args:
        n_samples: Number of synthetic tickets to generate.
        random_state: NumPy random seed for reproducibility.

    Returns:
        Tuple of (X DataFrame, y_priority ndarray, y_sla ndarray).
    """
    rng = np.random.default_rng(random_state)

    departments = rng.choice(DEPARTMENTS, n_samples)
    incident_types = rng.choice(INCIDENT_TYPES, n_samples)
    channels = rng.choice(CHANNELS, n_samples, p=[0.15, 0.45, 0.20, 0.20])
    tiers = rng.choice(TIERS, n_samples, p=[0.25, 0.15, 0.30, 0.30])
    product_areas = rng.choice(PRODUCT_AREAS, n_samples)
    hour_of_day = rng.integers(0, 24, n_samples).astype(float)
    day_of_week = rng.integers(0, 7, n_samples).astype(float)
    open_tickets = rng.poisson(30, n_samples).astype(float)
    agent_load = rng.integers(1, 15, n_samples).astype(float)
    ticket_age = rng.exponential(60, n_samples)
    same_user = rng.poisson(1.5, n_samples).astype(float)

    descriptions = [
        _generate_description(inc, dep, rng)
        for inc, dep in zip(incident_types, departments, strict=True)
    ]

    tier_score = np.where(np.isin(tiers, ["Gold"]), 1.5,
                 np.where(np.isin(tiers, ["Silver"]), 0.8, 0.0))
    inc_score = np.where(np.isin(incident_types, ["Security", "Network"]), 1.5,
                np.where(np.isin(incident_types, ["Application", "Access"]), 0.8, 0.0))
    workload_score = (open_tickets / 50.0) + (agent_load / 15.0)
    hour_score = np.where((hour_of_day >= 8) & (hour_of_day <= 18), 0.5, 1.0)

    latent = tier_score + inc_score + workload_score + hour_score
    latent += rng.normal(0, 0.5, n_samples)

    # Map latent score to 4-class priority (0=P1 critical, 3=P4 low)
    thresholds = np.percentile(latent, [20, 50, 75])
    y_priority = np.where(latent >= thresholds[2], 0,
                 np.where(latent >= thresholds[1], 1,
                 np.where(latent >= thresholds[0], 2, 3)))

    # SLA breach: P1/P2 tickets + high workload raises breach probability
    breach_prob = (
        np.where(y_priority == 0, 0.85,
        np.where(y_priority == 1, 0.60,
        np.where(y_priority == 2, 0.30, 0.10)))
        + 0.1 * workload_score
    ).clip(0.0, 1.0)
    y_sla = (rng.uniform(0, 1, n_samples) < breach_prob).astype(int)

    is_weekend = (day_of_week >= 5).astype(float)
    is_month_end = rng.integers(0, 2, n_samples).astype(float)

    X = pd.DataFrame({
        TEXT_COL: descriptions,
        "department": departments,
        "incident_type": incident_types,
        "channel": channels,
        "customer_tier": tiers,
        "product_area": product_areas,
        "open_tickets_count": open_tickets,
        "agent_load": agent_load,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_month_end": is_month_end,
        "ticket_age_minutes": ticket_age,
        "same_user_last_7d": same_user,
    })

    logger.info(
        "synthetic_dataset_generated",
        extra={"n_samples": n_samples, "priority_dist": dict(zip(*np.unique(y_priority, return_counts=True), strict=True))},
    )
    return X, y_priority, y_sla


_DESCRIPTION_TEMPLATES: dict[str, list[str]] = {
    "Security": [
        "Unauthorized access attempt detected on {dep} systems",
        "Password reset required for {dep} service account after breach",
        "SSL certificate expired on {dep} production server",
    ],
    "Network": [
        "VPN connectivity failure affecting {dep} remote workers",
        "Network latency spike on {dep} subnet degrading performance",
        "Firewall blocking legitimate {dep} traffic to external service",
    ],
    "Application": [
        "{dep} application crashing on startup after latest update",
        "Authentication error in {dep} portal for multiple users",
        "API timeout errors in {dep} integration with payment gateway",
    ],
    "Hardware": [
        "Laptop screen flickering and overheating in {dep} department",
        "Printer offline affecting multiple {dep} team members",
        "Workstation BSOD after memory upgrade in {dep}",
    ],
    "Access": [
        "New employee in {dep} requires system access provisioning",
        "User locked out of {dep} Active Directory account",
        "Permission denied error accessing {dep} shared drive",
    ],
    "Software": [
        "{dep} team unable to install required software due to admin policy",
        "License expiry warning for {dep} design tool",
        "Office 365 activation failing for {dep} users after device refresh",
    ],
    "Service Request": [
        "{dep} team requesting software upgrade to latest version",
        "Request to provision cloud storage bucket for {dep} project",
        "New monitor setup required for {dep} workstation",
    ],
    "Unknown": [
        "General IT issue reported by {dep} team member",
        "Intermittent problem observed in {dep} workflow",
    ],
}


def _generate_description(incident_type: str, department: str, rng: np.random.Generator) -> str:
    templates = _DESCRIPTION_TEMPLATES.get(incident_type, _DESCRIPTION_TEMPLATES["Unknown"])
    tpl = templates[int(rng.integers(0, len(templates)))]
    return tpl.format(dep=department)
