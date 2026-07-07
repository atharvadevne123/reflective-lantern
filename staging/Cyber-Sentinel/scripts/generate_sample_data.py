"""Generate sample network event data for testing and demonstration."""
from __future__ import annotations

import json
import logging
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROTOCOLS = ["tcp", "udp", "icmp"]
FLAG_COMBOS = ["SA", "S", "F", "R", "FA", "RSTO"]
ATTACK_PATTERNS = {
    "normal": {"src_bytes": (100, 5000), "duration_ms": (10, 500), "serror_rate": (0.0, 0.05)},
    "dos": {"src_bytes": (50000, 500000), "duration_ms": (0, 10), "serror_rate": (0.8, 1.0)},
    "probe": {"src_bytes": (50, 500), "duration_ms": (5, 50), "serror_rate": (0.1, 0.5)},
}


def generate_event(pattern: str = "normal") -> dict[str, object]:
    """Generate a synthetic network event dict for a given attack pattern."""
    p = ATTACK_PATTERNS.get(pattern, ATTACK_PATTERNS["normal"])
    src_lo, src_hi = p["src_bytes"]
    dur_lo, dur_hi = p["duration_ms"]
    err_lo, err_hi = p["serror_rate"]

    return {
        "protocol": random.choice(PROTOCOLS),
        "payload_bytes": random.randint(src_lo, src_hi),
        "duration_ms": round(random.uniform(dur_lo, dur_hi), 2),
        "src_bytes": random.randint(src_lo, src_hi),
        "dst_bytes": random.randint(0, 10000),
        "flags": random.choice(FLAG_COMBOS),
        "wrong_fragment": random.randint(0, 3),
        "urgent": random.randint(0, 1),
        "hot": random.randint(0, 20),
        "num_failed_logins": random.randint(0, 5),
        "logged_in": random.randint(0, 1),
        "num_compromised": random.randint(0, 10),
        "root_shell": random.randint(0, 1),
        "count": random.randint(1, 511),
        "srv_count": random.randint(1, 511),
        "serror_rate": round(random.uniform(err_lo, err_hi), 4),
        "rerror_rate": round(random.uniform(0, 0.3), 4),
        "same_srv_rate": round(random.uniform(0.5, 1.0), 4),
        "diff_srv_rate": round(random.uniform(0.0, 0.5), 4),
    }


def main() -> None:
    """Generate and print 10 sample events."""
    samples = []
    for pattern in ["normal", "dos", "probe"]:
        for _ in range(3):
            ev = generate_event(pattern)
            ev["_label"] = pattern
            samples.append(ev)
    print(json.dumps(samples, indent=2))
    logger.info("Generated %d sample events", len(samples))


if __name__ == "__main__":
    main()
