"""FAISS-backed threat intelligence retriever for CVE and MITRE ATT&CK context."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

THREAT_INTEL_CORPUS: list[dict[str, str]] = [
    {
        "id": "CVE-2021-44228",
        "text": "Log4Shell: Remote code execution in Apache Log4j 2.x. "
                "Attackers send JNDI lookup strings via HTTP headers to exploit "
                "LDAP/RMI callbacks. Class: U2R. Mitigation: upgrade to 2.17+.",
    },
    {
        "id": "MITRE-T1190",
        "text": "Exploit Public-Facing Application (T1190): Adversaries exploit "
                "weaknesses in internet-facing software to gain initial access. "
                "Commonly associated with R2L and U2R attack classes.",
    },
    {
        "id": "MITRE-T1046",
        "text": "Network Service Discovery (T1046): Adversaries perform port scanning "
                "to enumerate services. Produces high diff_srv_rate and elevated "
                "serror_rate. Maps to Probe attack class.",
    },
    {
        "id": "MITRE-T1498",
        "text": "Network Denial of Service (T1498): Flooding a target to degrade "
                "availability. Characterised by very high connection count, near-zero "
                "dst_bytes and high serror_rate. Maps to DoS class.",
    },
    {
        "id": "CVE-2017-0144",
        "text": "EternalBlue: SMB protocol exploit enabling remote code execution. "
                "Large src_bytes bursts, SMB service, SF flag. Class: R2L/U2R.",
    },
    {
        "id": "CVE-2019-0708",
        "text": "BlueKeep: Remote Desktop Protocol pre-auth RCE on Windows. "
                "Abnormal RDP traffic with many failed logins. Class: R2L.",
    },
    {
        "id": "MITRE-T1071",
        "text": "Application Layer Protocol (T1071): C2 communication over legitimate "
                "protocols (HTTP, DNS, SMTP). Elevated dst_bytes, normal-looking flags. "
                "Hard to distinguish from benign traffic without entropy analysis.",
    },
    {
        "id": "MITRE-T1110",
        "text": "Brute Force (T1110): Repeated login attempts. High num_failed_logins, "
                "consistent destination, low src/dst bytes ratio. Class: R2L.",
    },
    {
        "id": "MITRE-T1595",
        "text": "Active Scanning (T1595): Systematic port and vulnerability scanning. "
                "High srv_count with high diff_srv_rate, ICMP or TCP SYN traffic. "
                "Maps to Probe attack class.",
    },
    {
        "id": "CVE-2014-6271",
        "text": "Shellshock: Bash vulnerability allowing remote command injection "
                "via environment variables in HTTP headers. Elevated hot count. "
                "Class: U2R/R2L. Affects web servers running CGI scripts.",
    },
]


class ThreatIntelRetriever:
    """Simple TF-IDF + cosine-similarity retriever over threat intel corpus."""

    def __init__(self) -> None:
        self._index: np.ndarray | None = None
        self._corpus = THREAT_INTEL_CORPUS
        self._vocab: dict[str, int] = {}

    def build_index(self) -> None:
        """Build TF-IDF vectors for the threat corpus."""
        texts = [d["text"].lower() for d in self._corpus]
        words = set()
        for t in texts:
            words.update(t.split())
        self._vocab = {w: i for i, w in enumerate(sorted(words))}

        matrix = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.split():
                if word in self._vocab:
                    matrix[i, self._vocab[word]] += 1.0

        # TF normalisation
        row_sums = matrix.sum(axis=1, keepdims=True) + 1e-9
        matrix = matrix / row_sums

        # IDF weighting
        df = (matrix > 0).sum(axis=0) + 1.0
        idf = np.log(len(texts) / df)
        self._index = matrix * idf
        logger.info("Threat intel index built: %d docs, %d terms", len(texts), len(self._vocab))

    def _vectorise(self, query: str) -> np.ndarray:
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for word in query.lower().split():
            if word in self._vocab:
                vec[self._vocab[word]] += 1.0
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top-k threat intel entries most similar to query.

        Args:
            query: Free-text query (attack class, CVE ID, technique name).
            top_k: Number of results to return.

        Returns:
            List of dicts with id, text, and similarity score.
        """
        if self._index is None:
            self.build_index()
        q_vec = self._vectorise(query)
        norms = np.linalg.norm(self._index, axis=1) + 1e-9
        sims = (self._index @ q_vec) / norms
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "id": self._corpus[i]["id"],
                "text": self._corpus[i]["text"],
                "similarity": round(float(sims[i]), 4),
            }
            for i in top_idx
        ]
